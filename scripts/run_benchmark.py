#!/usr/bin/env python3
"""Durable, resumable SFT campaign. One immutable attempt directory per episode."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from inference_pool import load_services, validate_routing_upgrade
from evaluation_recovery import classify_attempt, retry_eligible
from checkpoint_specs import CHECKPOINTS, checkpoint_spec

ROOT = Path(os.environ.get('CUA_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
ADAPTERS = ['evaluate_sft.py', 'eval_qemu_provider.py', 'qemu_runtime.py', 'run_benchmark.py', 'scienceboard_compat.py', 'inference_pool.py', 'evaluation_recovery.py', 'checkpoint_specs.py']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def fingerprint(directory=None, names=None):
    directory = directory or ROOT / 'scripts'
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in (names or ADAPTERS)}


def official_tasks(benchmark):
    name = 'test_nogdrive' if benchmark == 'osworld' else 'test_scienceboard'
    path = ROOT / 'SCALE-CUA/osworld_eval/evaluation_examples' / (name + '.json')
    mapping = json.loads(path.read_text())
    # Deterministic interleaving exposes app-specific setup faults early.
    tasks = [f'{domain}/{ids[i]}' for i in range(max(map(len, mapping.values())))
             for domain, ids in mapping.items() if i < len(ids)]
    if len(tasks) != len(set(tasks)):
        raise ValueError('Duplicate official task IDs')
    return tasks, hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows):
    completed = [r for r in rows if r['status'] == 'completed']
    n, valid = len(rows), len(completed)
    successes = sum(r['score'] == 1 for r in completed)
    errors = sum(r['status'] == 'infrastructure_error' for r in rows)
    return {'total': n, 'completed': valid, 'successes': successes,
            'infrastructure_or_verifier_errors': errors,
            'running': sum(r['status'] == 'running' for r in rows),
            'pending': sum(r['status'] == 'pending' for r in rows),
            'coverage': valid / n if n else None,
            'success_rate': successes / n if n and valid == n else None,
            'mean_score': sum(r['score'] for r in completed) / n if n and valid == n else None,
            'note': 'Full-denominator metrics remain null until every task has a valid result.'}


def snapshot(state):
    rows = list(state['tasks'].values())
    domains = sorted({r['domain'] for r in rows})
    return {'updated_at_utc': now(), 'status': state['status'], 'checkpoint': state['protocol']['checkpoint'],
            'benchmark': state['protocol']['benchmark'], 'protocol': state['protocol'],
            'overall': aggregate(rows),
            'by_domain': {d: aggregate([r for r in rows if r['domain'] == d]) for d in domains},
            'errors': {k: v.get('error') for k, v in state['tasks'].items() if v['status'] == 'infrastructure_error'},
            'workers': state.get('workers'), 'inference_pool': state.get('inference_pool'),
            'protocol_migrations': state.get('protocol_migrations', []),
            'pause_reason': state.get('pause_reason')}


def needs_run(row, retry_errors=False):
    return row['status'] in {'pending', 'running'} or (retry_errors and row['status'] == 'infrastructure_error')


def select_queue(tasks, rows, retry_errors=False, only_tasks=None):
    selected = set(only_tasks) if only_tasks is not None else set(tasks)
    if not selected.issubset(tasks):
        raise ValueError('Selected tasks must belong to the official campaign')
    return [t for t in tasks if t in selected and needs_run(rows[t], retry_errors)]


def check_endpoint(service):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(service['base_url'] + '/models', timeout=15) as response:
        models = json.load(response)['data']
    if not any(m['id'] == service['served_model_name'] and m.get('root') == service['model_dir'] for m in models):
        raise RuntimeError('Pinned SFT endpoint identity mismatch')


def check_concurrency(workers, cpus, memory_mb, benchmark, report_path=None):
    """Require a matching host/image/resource acceptance run above four VMs."""
    if not 1 <= workers <= 32:
        raise ValueError('Campaign concurrency must be between one and thirty-two')
    if workers <= 4:
        return None
    if report_path is None:
        raise ValueError('More than four workers requires --concurrency-report')
    content = Path(report_path).read_bytes()
    report = json.loads(content)
    image = ROOT / ('data/osworld-vm/Ubuntu.qcow2' if benchmark == 'osworld'
                    else 'data/scienceboard-vm/Ubuntu2_scienceboard.qcow2')
    if not (report.get('status') == 'passed' and report.get('hostname') == socket.gethostname()
            and report.get('accelerator') == 'tcg' and report.get('vcpus_per_vm') == cpus
            and report.get('memory_mb_per_vm') == memory_mb
            and report.get('base_image') == str(image.resolve())
            and any(stage.get('concurrency', 0) >= workers and stage.get('passed')
                    and stage.get('cleanup_passed') for stage in report.get('stages', []))):
        raise ValueError('Concurrency acceptance report does not match this host/image/resource configuration')
    return {'source': str(Path(report_path).resolve()), 'sha256': hashlib.sha256(content).hexdigest(),
            'report': report}


def stop_process(process, output):
    # Each evaluation owns its process group; QEMU has its own session and is
    # cleaned separately using exact run/host/UUID identities.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    manifest = output / 'manifest.json'
    if manifest.exists():
        from eval_qemu_provider import cleanup_run
        cleanup_run(json.loads(manifest.read_text())['run_id'])


def episode(args, task, output, service, expected_hashes, stopped):
    log_path = output.parent / (output.name + '.log')
    result = {'attempt_dir': str(output), 'log': str(log_path), 'started_at_utc': now()}
    process = None
    try:
        if fingerprint(args.runtime_dir, expected_hashes) != expected_hashes:
            raise RuntimeError('Adapter changed during campaign; refusing mixed protocols')
        if service.get('_state_path') and hashlib.sha256(Path(service['_state_path']).read_bytes()).hexdigest() != service['_state_sha256']:
            raise RuntimeError('Pinned replica state changed during campaign')
        check_endpoint(service)
        command = [sys.executable, str(args.runtime_dir / 'evaluate_sft.py'),
                   '--benchmark', args.benchmark, '--checkpoint', args.checkpoint,
                   '--accelerator', 'tcg', '--workers', '1', '--cpus', str(args.cpus),
                   '--memory-mb', str(args.memory_mb), '--recording-mode', 'steps-only',
                   '--proxy-policy', args.proxy_policy, '--tasks', task, '--run-dir', str(output)]
        if service.get('_state_path'):
            command += ['--service-state', service['_state_path']]
            result['inference_service_state'] = service['_state_path']
        with log_path.open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, 'CUA_PROJECT_ROOT': str(ROOT)})
            result['pid'] = process.pid
            write_json(output.parent / (output.name + '.process.json'), result)
            deadline = time.monotonic() + args.task_timeout
            while process.poll() is None:
                if stopped() or time.monotonic() > deadline:
                    stop_process(process, output)
                    raise RuntimeError('Campaign interrupted' if stopped() else 'Episode wall time limit exceeded')
                time.sleep(1)
        result['exit_code'] = process.returncode
        summary_path = output / 'summary.json'
        if not summary_path.exists():
            raise RuntimeError('Evaluation exited without a summary; inspect attempt log')
        summary = json.loads(summary_path.read_text())
        row = summary['tasks'][0]
        result['action_records'] = row.get('action_records', 0)
        score = row['score']
        if row['status'] != 'completed' or score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError('; '.join(row['errors']) or 'Episode has no valid verifier result')
        result.update(status='completed', score=score, action_records=row['action_records'], directory=row['directory'])
    except Exception as exc:
        result.update(status='infrastructure_error', score=None, error=str(exc))
    finally:
        if process is not None:
            stop_process(process, output)
        if result.get('status') == 'infrastructure_error':
            result['recovery'] = classify_attempt(result)
        result['finished_at_utc'] = now()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'run', 'status'])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--benchmark', choices=['osworld', 'scienceboard'], default='osworld')
    p.add_argument('--checkpoint', choices=CHECKPOINTS, default='osworld-sft')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--concurrency-report', type=Path, help='Matching VM acceptance report, required above four workers')
    p.add_argument('--service-pool', type=Path, help='Pool of identical checkpoint replicas with equivalent inference flags')
    p.add_argument('--upgrade-service-routing', action='store_true', help='Audited routing-only launcher upgrade on a drained legacy campaign')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--memory-mb', type=int, default=8192)
    p.add_argument('--proxy-policy', choices=['upstream', 'direct'], default='upstream')
    p.add_argument('--task-timeout', type=int, default=3600)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--auto-recover', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-infra-attempts', type=int, default=3)
    p.add_argument('--retry-backoff-seconds', type=int, default=30)
    p.add_argument('--launch-interval', type=float, default=1.0,
                   help='Pace new episode process launches to reduce simultaneous guest boot load')
    p.add_argument('--only-tasks', nargs='+', help='Schedule only these domain/id entries; retain the full benchmark denominator')
    args = p.parse_args()
    if not 1 <= args.max_infra_attempts <= 5 or not 0 <= args.retry_backoff_seconds <= 300:
        p.error('Recovery requires 1..5 attempts and a backoff of 0..300 seconds')
    if not 0 <= args.launch_interval <= 10:
        p.error('launch-interval must be in 0..10 seconds')
    output = args.run_dir.resolve()
    if args.action == 'status':
        print((output / 'summary.json').read_text())
        return
    if socket.gethostname() not in {'research-a-worker-0', 'research-a-worker-1'}:
        p.error('Campaigns run only on worker-0/1')
    try:
        concurrency_evidence = check_concurrency(args.workers, args.cpus, args.memory_mb,
                                               args.benchmark, args.concurrency_report)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    if args.action == 'start':
        cmd = [sys.executable, str(Path(__file__).resolve()), 'run', *sys.argv[2:]]
        with (output / 'campaign.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError('Campaign failed to launch; see ' + str(output / 'campaign.log'))
        print(json.dumps({'pid': proc.pid, 'hostname': socket.gethostname(), 'directory': str(output)}))
        return
    lease = (output / 'campaign.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    service_file = checkpoint_spec(ROOT, args.checkpoint)['state_file']
    service = json.loads((ROOT / 'runs' / service_file).read_text())
    check_endpoint(service)
    services = load_services(args.service_pool, service) if args.service_pool else [service]
    for item in services:
        check_endpoint(item)
    tasks, task_hash = official_tasks(args.benchmark)
    path = output / 'campaign.json'
    previous = json.loads(path.read_text()) if path.exists() else None
    args.runtime_dir = output / (previous.get('runtime_subdirectory', 'runtime') if previous else 'runtime')
    migration = None
    if args.upgrade_service_routing:
        if previous is None or not args.resume or not args.service_pool or previous['status'] == 'running':
            p.error('Routing upgrade requires --resume, --service-pool and a drained existing campaign')
        old_hashes = previous['protocol']['adapter_sha256']
        if fingerprint(args.runtime_dir, old_hashes) != old_hashes:
            p.error('Original runtime differs from its recorded hashes; routing upgrade refused')
        if any(row['status'] == 'running' for row in previous['tasks'].values()):
            p.error('Routing upgrade requires every active episode to finish first')
        migration = validate_routing_upgrade(args.runtime_dir, ROOT / 'scripts')
        args.runtime_dir = output / 'runtimes' / f'service-routing-{time.time_ns()}'
        args.runtime_dir.mkdir(parents=True, exist_ok=False)
        for name in ADAPTERS:
            (args.runtime_dir / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    if not path.exists():
        args.runtime_dir.mkdir(exist_ok=False)
        for name in ADAPTERS:
            (args.runtime_dir / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    elif not args.runtime_dir.is_dir():
        p.error('Legacy campaign has no immutable runtime snapshot; start a new version')
    # A legacy campaign must keep exactly its original runtime file set.
    # New campaigns include all current adapters; missing pinned files always fail.
    pinned_names = previous['protocol']['adapter_sha256'] if previous and not migration else ADAPTERS
    protocol = {'benchmark': args.benchmark, 'checkpoint': args.checkpoint,
                'model_revision': service['model_revision'], 'server_command': service['command'],
                'task_list_sha256': task_hash, 'adapter_sha256': fingerprint(args.runtime_dir, pinned_names),
                'accelerator': 'tcg', 'recording_mode': 'steps-only', 'cpus': args.cpus,
                'memory_mb': args.memory_mb, 'proxy_policy': args.proxy_policy,
                'task_timeout_seconds': args.task_timeout, 'samples_per_task': 1}
    if path.exists():
        if not args.resume:
            p.error('Existing campaign requires --resume; nothing is overwritten')
        state = previous
        if migration:
            old_semantics = {k: v for k, v in state['protocol'].items() if k != 'adapter_sha256'}
            new_semantics = {k: v for k, v in protocol.items() if k != 'adapter_sha256'}
            if old_semantics != new_semantics:
                p.error('Routing upgrade must preserve checkpoint, generation, task list and VM protocol')
            state.setdefault('protocol_migrations', []).append({
                'at_utc': now(), 'kind': 'inference_service_routing_only', 'validation': migration,
                'previous_protocol': state['protocol'], 'new_protocol': protocol,
                'preserved_completed_results': {k: r['score'] for k, r in state['tasks'].items() if r['status'] == 'completed'}})
            state['protocol'] = protocol
            state['runtime_subdirectory'] = str(args.runtime_dir.relative_to(output))
        if state['protocol'] != protocol:
            p.error('Resume protocol differs from the original campaign')
        if state['hostname'] != socket.gethostname():
            p.error('Resume on the original worker so orphan cleanup is reliable')
        # Reconcile completed attempts after a coordinator crash before scheduling.
        for task, row in state['tasks'].items():
            if row['status'] != 'running':
                continue
            last = row['attempts'][-1]
            attempt = Path(last['attempt_dir'])
            process_file = attempt.parent / (attempt.name + '.process.json')
            if process_file.exists():
                pid = json.loads(process_file.read_text()).get('pid')
                cmdline = Path(f'/proc/{pid}/cmdline')
                if cmdline.exists() and str(attempt).encode() in cmdline.read_bytes().split(b'\0'):
                    p.error(f'Previous episode process {pid} is still live; do not duplicate it')
            from eval_qemu_provider import cleanup_run
            if (attempt / 'manifest.json').exists():
                cleanup_run(json.loads((attempt / 'manifest.json').read_text())['run_id'])
            if (attempt / 'summary.json').exists():
                measured = json.loads((attempt / 'summary.json').read_text())['tasks'][0]
                if measured['status'] == 'completed':
                    last.update(measured)
                    row.update(status='completed', score=measured['score'])
                    continue
            recovery = classify_attempt(last)
            last.update(status='infrastructure_error', score=None, recovery=recovery,
                        error='Coordinator or episode process exited before a valid result was recorded')
            row.update(status='infrastructure_error', score=None, error=last['error'])
    else:
        state = {'created_at_utc': now(), 'hostname': socket.gethostname(), 'protocol': protocol,
                 'tasks': {t: {'domain': t.split('/')[0], 'status': 'pending', 'score': None, 'attempts': []} for t in tasks}}
    queue = select_queue(tasks, state['tasks'], args.retry_errors, args.only_tasks)
    if args.auto_recover:
        allowed = set(args.only_tasks or tasks)
        queue += [t for t in tasks if t in allowed and t not in queue
                  and retry_eligible(state['tasks'][t], args.max_infra_attempts)]
    state.update(status='running', pid=os.getpid(), workers=args.workers, pause_reason=None)
    coordinator_code = Path(__file__).read_bytes()
    coordinator_dir = output / 'coordinators'
    coordinator_dir.mkdir(exist_ok=True)
    coordinator_path = coordinator_dir / f'{time.time_ns()}-{os.getpid()}.py'
    coordinator_path.write_bytes(coordinator_code)
    if args.service_pool:
        if '--service-state' not in (args.runtime_dir / 'evaluate_sft.py').read_text():
            p.error('Legacy runtime needs the explicit routing-only upgrade before replica selection')
        service_dir = coordinator_path.with_suffix('.services')
        service_dir.mkdir()
        pinned_services = []
        for index, item in enumerate(services):
            pinned = service_dir / f'{index:02d}.json'
            write_json(pinned, item)
            pinned_services.append({**item, '_state_path': str(pinned),
                                    '_state_sha256': hashlib.sha256(pinned.read_bytes()).hexdigest()})
        services = pinned_services
        state['inference_pool'] = {'source': str(args.service_pool.resolve()),
                                   'services': [{k: item[k] for k in ['hostname', 'gpus', 'base_url', 'model_revision', '_state_path', '_state_sha256']} for item in services]}
    elif state.get('inference_pool'):
        p.error('Resuming a pooled campaign requires --service-pool')
    state.setdefault('launch_history', []).append({
        'at_utc': now(), 'pid': os.getpid(), 'retry_errors': args.retry_errors,
        'workers': args.workers, 'concurrency_acceptance': concurrency_evidence,
        'recovery_policy': {'auto_recover': args.auto_recover, 'max_attempts': args.max_infra_attempts,
                            'backoff_seconds': args.retry_backoff_seconds,
                            'launch_interval_seconds': args.launch_interval,
                            'scope': 'Explicit zero-action initialization failures only; valid scores and acted episodes never resampled'},
        'inference_pool': state.get('inference_pool'),
        'scheduled_tasks': queue.copy(), 'coordinator_snapshot': str(coordinator_path),
        'coordinator_sha256': hashlib.sha256(coordinator_code).hexdigest()})
    stopped = [False]
    draining = [False]
    def interrupted(signum, frame):
        stopped[0] = True
        state['pause_reason'] = f'Received signal {signum}'
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def drain(signum, frame):
        draining[0] = True
        state['pause_reason'] = 'Draining active episodes for a coordinator handoff'
    signal.signal(signal.SIGUSR1, drain)
    def save():
        state['updated_at_utc'] = now()
        write_json(path, state)
        write_json(output / 'summary.json', snapshot(state))
    save()
    retry_at = {}
    endpoint_retry_at = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        free_slots = list(range(args.workers))
        while queue or active:
            while queue and len(active) < args.workers and not stopped[0] and not draining[0]:
                ready = next((t for t in queue if retry_at.get(t, 0) <= time.monotonic()), None)
                if ready is None:
                    break
                available = [s for s in free_slots if endpoint_retry_at.get(services[s % len(services)]['base_url'], 0) <= time.monotonic()]
                if not available:
                    break
                slot = available[0]
                selected_service = services[slot % len(services)]
                try:
                    check_endpoint(selected_service)
                except Exception as exc:
                    endpoint_retry_at[selected_service['base_url']] = time.monotonic() + 30
                    state.setdefault('service_faults', []).append({'at_utc': now(), 'base_url': selected_service['base_url'], 'error': str(exc)})
                    save()
                    continue
                task = ready
                queue.remove(task)
                row = state['tasks'][task]
                attempt = output / 'attempts' / task.replace('/', '__') / f'{len(row["attempts"]) + 1:03d}'
                attempt.parent.mkdir(parents=True, exist_ok=True)
                row['attempts'].append({'attempt_dir': str(attempt), 'status': 'running'})
                row['status'] = 'running'
                free_slots.remove(slot)
                active[pool.submit(episode, args, task, attempt, selected_service, protocol['adapter_sha256'], lambda: stopped[0])] = (task, slot)
                save()
                if args.launch_interval:
                    time.sleep(args.launch_interval)
            if not active:
                if not queue or stopped[0] or draining[0]:
                    break
                time.sleep(2)
                continue
            done, _ = concurrent.futures.wait(active, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task, slot = active.pop(future)
                free_slots.append(slot)
                result = future.result()
                row = state['tasks'][task]
                row['attempts'][-1] = result
                row.update(status=result['status'], score=result['score'], error=result.get('error'))
                print(json.dumps({'task': task, **result}), flush=True)
                if args.auto_recover and retry_eligible(row, args.max_infra_attempts):
                    queue.append(task)
                    retry_at[task] = time.monotonic() + args.retry_backoff_seconds * 2 ** (len(row['attempts']) - 1)
                save()
            save()
    summary = aggregate(list(state['tasks'].values()))
    state['status'] = 'complete' if summary['completed'] == summary['total'] else ('paused_for_review' if stopped[0] or draining[0] else 'finished_with_errors')
    save()


if __name__ == '__main__':
    main()
