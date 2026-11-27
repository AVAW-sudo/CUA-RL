"""Four auditable SFT pre-runs per official RL task; never invent missing rates."""
import argparse
import asyncio
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

from aiohttp import ClientError, ClientResponseError, ClientSession, ClientTimeout
from ais_frontier_shards import write_shard_rates
from build_initial_rates import TERMINAL_AGENT_STATES
from inference_pool import load_services
from run_benchmark import ROOT, write_json


def load_image_filter(source):
    agent = source / 'scalecua_rl/scripts/server_agent.py'
    if agent.is_file():
        spec = importlib.util.spec_from_file_location('original_server_agent', agent)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
    else:
        # The AIS worker archive includes the same upstream filter in the OSWorld adapter.
        evaluator = source / 'osworld_eval'
        adapter = evaluator / 'mm_agents/scalecua-osworld/main.py'
        if not adapter.is_file():
            raise FileNotFoundError(f'ScaleCUA image filter missing from both {agent} and {adapter}')
        sys.path.insert(0, str(evaluator))
        try:
            upstream = importlib.import_module('mm_agents.scalecua-osworld.main')
        finally:
            sys.path.pop(0)
    return upstream._maybe_filter_to_n_most_recent_images


def task_request(name, index):
    return {'name': name, 'index': -1, 'custom_task': {'index': index, 'with_record': False,
            'image_width': 1280, 'image_height': 720, 'screen_width': 1920, 'screen_height': 1080,
            'agent_version': 'scalecua_osworld', 'env_version': 'anthropic',
            'agent_config': {'relative_coordinate': True}, 'max_rounds': 50}}


async def initial_state(rpc, name, index, receipt):
    """End a fresh guest session before any action and let its official judge score it."""
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        if record['task_index'] != index:
            raise ValueError(f'Misplaced control at {receipt}')
        return record['status'] == 'passed'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {'task_index': index, 'status': 'unavailable', 'errors': []}
    for attempt in range(2):
        sid = None
        try:
            data, sid = await rpc('/start_sample', task_request(name, index))
            if not sid or data.get('finish') or not data.get('messages'):
                raise ValueError('No live initial observation')
            data, _ = await rpc('/interact', {'messages': [{'role': 'assistant', 'content': 'Done.'}]}, sid)
            reward = data.get('reward')
            if (not data.get('finish') or data.get('status') != 'completed' or
                    type(reward) not in (float, int) or not 0 <= reward <= 1):
                raise ValueError(f'Initial state had no verified reward ({data.get("status")})')
            record.update(status='passed' if reward == 0 else 'nonzero_initial_reward', reward=reward, session_id=sid)
            break
        except (ClientError, asyncio.TimeoutError, ValueError) as exc:
            record['errors'].append(str(exc))
        finally:
            if sid:
                try:
                    await rpc('/cancel', sid=sid)
                except (ClientError, asyncio.TimeoutError):
                    pass
        if attempt == 0:
            await asyncio.sleep(3)
    write_json(receipt, record)
    return record['status'] == 'passed'


async def release_failed_start(client, worker, exc):
    # The controller forgets a session after a /start_sample error. Its response
    # still identifies the exact session the private worker may be initializing.
    sid = (exc.headers or {}).get('session_id', '')
    if not sid.isdigit():
        return None
    async with client.post(worker + '/cancel', json={'session_id': int(sid)}, timeout=240) as response:
        if response.status == 400 and 'No such session' in await response.text():
            return {'session_id': sid, 'status': 'already_released'}
        response.raise_for_status()
        return {'session_id': sid, 'status': 'released'}


async def collect_all(args):
    source = Path(os.getenv('CUA_SOURCE_DIR', ROOT / 'SCALE-CUA'))
    filter_images = load_image_filter(source)
    primary = json.loads(args.primary_service.read_text())
    services = load_services(args.service_pool, primary)
    indices = json.loads(args.indices.read_text())
    if args.probe_one is not None and args.probe_one not in indices:
        raise ValueError(f'Probe task is outside the selected indices: {args.probe_one}')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = {'model_revision': primary['model_revision'], 'tasks': len(indices), 'task_name': args.task_name, 'runs_per_task': 4,
                'temperature': 0.8, 'max_tokens': 4096, 'max_turns': 50, 'max_total_len': 30000,
                'generation_transport_retries': 1,
                'images_to_keep': 5, 'min_removal_threshold': 5, 'agent_version': 'scalecua_osworld',
                'agent_note': 'Released public ScaleCUA agent adapter; anthropic_rl is absent from the pinned public source.',
                'indices_sha256': hashlib.sha256(args.indices.read_bytes()).hexdigest(),
                'top_p': 1.0, 'top_k': -1,
                'services': [{k: s.get(k) for k in ['model_revision', 'model_dir', 'served_model_name',
                                                  'base_url', 'command', 'runtime_compatibility']} for s in services]}
    if args.reused_protocol:
        from ais_frontier_shards import require_equivalent_protocol
        require_equivalent_protocol(json.loads(args.reused_protocol.read_text()), protocol)
    protocol_path = directory / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Frontier pre-run protocol changed; preserve this run and use a new directory')
    write_json(protocol_path, protocol)
    selection = ROOT / 'runs/original-rl-pipeline/secondary-worker-selection.json'
    with (directory / 'launches.jsonl').open('a') as log:
        log.write(json.dumps({'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'concurrency': args.concurrency, 'controller': args.controller,
                              'secondary_worker': json.loads(selection.read_text()) if selection.exists() else None}) + '\n')
    selected = [args.probe_one] if args.probe_one else indices

    async with ClientSession(timeout=ClientTimeout(total=900)) as client:
        async def rpc(route, body=None, sid=None):
            headers = {'session_id': str(sid)} if sid is not None else {}
            async with client.post(args.controller + route, json=body, headers=headers) as response:
                response.raise_for_status()
                return await response.json(), response.headers.get('session_id')

        async def episode(index, run, path, service):
            path.mkdir(parents=True, exist_ok=True)
            for number in range(1, args.max_attempts + 1):
                attempt = path / f'attempt-{number:02d}.json'
                if attempt.exists():
                    previous = json.loads(attempt.read_text())
                    if previous.get('model_calls', 0) or previous.get('result') is not None:
                        return
                    continue
                record = {'task_index': index, 'run_number': run, 'attempt': number, 'model_calls': 0,
                          'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'base_url': service['base_url'], 'model_revision': primary['model_revision'],
                          'timings': {'initialization_seconds': 0, 'inference_seconds': 0,
                                      'environment_seconds': 0}, 'turn_timings': []}
                sid = None
                # Write before the first network request so a crash cannot silently resample.
                write_json(attempt, record)
                try:
                    initialized = time.monotonic()
                    data, sid = await rpc('/start_sample', task_request(args.task_name, index))
                    record['session_id'] = sid
                    record['timings']['initialization_seconds'] = time.monotonic() - initialized
                    if data.get('finish') or not data.get('messages'):
                        raise RuntimeError('Initialization returned no live observation: ' + str(data.get('status')))
                    messages = data['messages']; tools = data.get('tools', [])
                    for turn in range(60):
                        messages = filter_images(messages, 5, 5)
                        record['model_calls'] += 1
                        write_json(attempt, record)
                        payload = {'model': service['served_model_name'], 'messages': messages,
                                   'temperature': 0.8, 'top_p': 1.0, 'top_k': -1, 'max_completion_tokens': 4096}
                        if tools:
                            payload['tools'] = tools
                        generated = time.monotonic()
                        for transport_attempt in range(2):
                            try:
                                async with client.post(service['base_url'] + '/chat/completions', json=payload) as response:
                                    response.raise_for_status()
                                    answer = await response.json()
                                break
                            except (ClientError, asyncio.TimeoutError) as exc:
                                record.setdefault('transport_errors', []).append({'turn': turn, 'attempt': transport_attempt, 'error': str(exc)})
                                write_json(attempt, record)
                                if transport_attempt:
                                    raise
                                await asyncio.sleep(2)
                        inference_seconds = time.monotonic() - generated
                        record['timings']['inference_seconds'] += inference_seconds
                        message = {k: v for k, v in answer['choices'][0]['message'].items() if v is not None}
                        message.setdefault('content', '')
                        with (path / f'trace-{number:02d}.jsonl').open('a') as log:
                            log.write(json.dumps({'turn': turn, 'message': message, 'usage': answer.get('usage'),
                                                  'inference_seconds': inference_seconds}) + '\n')
                        if answer.get('usage', {}).get('total_tokens', 0) > 30000:
                            data = {'finish': True, 'status': 'agent context limit', 'reward': 0}
                            break
                        messages.append(message)
                        acted = time.monotonic()
                        data, _ = await rpc('/interact', {'messages': [message]}, sid)
                        environment_seconds = time.monotonic() - acted
                        record['timings']['environment_seconds'] += environment_seconds
                        record['turn_timings'].append({'turn': turn, 'inference_seconds': inference_seconds,
                                                       'environment_seconds': environment_seconds})
                        write_json(attempt, record)
                        if data.get('finish'):
                            break
                        incoming = data['messages']
                        messages = incoming if incoming and incoming[0].get('role') == 'system' else messages + incoming
                    if not data.get('finish') or data.get('status') not in TERMINAL_AGENT_STATES:
                        raise RuntimeError('Unscored terminal state: ' + str(data.get('status')))
                    reward = data['reward']
                    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
                        raise RuntimeError(f'Invalid verifier reward: {reward!r}')
                    result = {'model': primary['served_model_name'], 'task_index': index, 'run_number': run,
                              'result': int(reward == 1), 'raw_reward': reward, 'terminal_status': data['status'],
                              'model_revision': primary['model_revision'], 'attempt': number}
                    record.update(result=result, status='completed', terminal={
                        key: data[key] for key in ('finish', 'status', 'reward')})
                    write_json(attempt, record)
                    write_json(path / 'result.json', result)
                    return
                except Exception as exc:
                    record.update(status='infrastructure_error', error=str(exc))
                    if args.private_worker and sid is None and isinstance(exc, ClientResponseError):
                        try:
                            record['worker_cleanup'] = await release_failed_start(client, args.private_worker, exc)
                        except (ClientError, asyncio.TimeoutError) as cleanup_error:
                            record['worker_cleanup'] = {'error': str(cleanup_error)}
                    write_json(attempt, record)
                    if record['model_calls']:
                        return
                finally:
                    if sid:
                        try:
                            await rpc('/cancel', sid=sid)
                        except Exception:
                            pass
                if number < args.max_attempts:
                    await asyncio.sleep(min(30 * number, 90))

        queue = asyncio.Queue()
        group_size = args.concurrency if args.screening else max(1, len(selected))
        controls = {}
        for start in range(0, len(selected), group_size):
            for run in ([1] if args.probe_one else range(1, 5)):
                for index in selected[start:start + group_size]:
                    path = directory / index / str(run)
                    if not (path / 'result.json').exists():
                        queue.put_nowait((index, run, path))

        async def lane(slot):
            service = services[slot % len(services)]
            while not queue.empty():
                index, run, path = queue.get_nowait()
                if args.check_initial_state:
                    if index not in controls:
                        controls[index] = asyncio.create_task(initial_state(
                            rpc, args.task_name, index, directory / index / 'initial-state.json'))
                    if not await controls[index]:
                        queue.task_done()
                        continue
                await episode(index, run, path, service)
                queue.task_done()
        await asyncio.gather(*(lane(slot) for slot in range(args.concurrency)))
    if args.probe_one:
        path = directory / args.probe_one / '1/result.json'
        if not path.is_file():
            raise RuntimeError('Frontier probe produced no scored terminal result')
        write_json(directory / 'probe.json', {'scope': 'single AIS integration probe',
                                               'result': json.loads(path.read_text())})
        return
    # The dataset screener audits each slot and publishes partial coverage itself.
    if not args.screening:
        write_shard_rates(directory, args.rates_output, protocol, indices, primary['served_model_name'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service-pool', type=Path, required=True)
    p.add_argument('--primary-service', type=Path, default=ROOT / 'runs/inference-service.json')
    p.add_argument('--probe-one', metavar='TASK', help='Collect only round one of this selected task')
    p.add_argument('--screening', action='store_true', help='Leave per-slot evidence for the screening audit')
    p.add_argument('--check-initial-state', action='store_true', help='Require a zero-reward untouched control before SFT trials')
    p.add_argument('--indices', type=Path, default=ROOT / 'configs/rl-indices.json')
    p.add_argument('--task-name', default='osworld-rl')
    p.add_argument('--rates-output', type=Path, default=ROOT / 'configs/initial_pass_rate.osworld-rl.json')
    p.add_argument('--controller', default='http://127.0.0.1:18520/api')
    p.add_argument('--reused-protocol', type=Path, help='Require equivalent sampling and service identity before collection')
    p.add_argument('--private-worker', help='Direct API for this single collector’s exclusively owned desktop worker')
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/frontier-initialization')
    args = p.parse_args()
    if args.check_initial_state and not args.screening:
        p.error('--check-initial-state requires --screening')
    if not 1 <= args.concurrency <= 64 or not 1 <= args.max_attempts <= 5:
        p.error('Invalid concurrency or retry budget')
    if args.private_worker and args.concurrency != 1:
        p.error('A private desktop worker requires a single collector lane')
    asyncio.run(collect_all(args))
