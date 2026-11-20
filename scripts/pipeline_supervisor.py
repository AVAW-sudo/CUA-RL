"""Persistently recover evaluation queues, publish reports, and run original-RL stages."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from persistent_job import identity, alive
from evaluation_recovery import retry_eligible
from run_benchmark import ROOT, now, write_json
from write_evaluation_report import generate

PYTHON = '/aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python'


def remote(host, command, timeout=60):
    if host != socket.gethostname():
        command = [PYTHON, str(ROOT / 'scripts/worker_exec.py'), host, '--', *command]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)


def campaign_live(state, directory):
    code = ('from pathlib import Path; import json; '
            f'p=Path("/proc/{state["pid"]}"); '
            f'print(json.dumps(p.exists() and {str(directory)!r}.encode() in (p/"cmdline").read_bytes().split(b"\\0")))')
    return json.loads(remote(state['hostname'], [PYTHON, '-c', code]).stdout)


def resume(directory, state):
    p = state['protocol']
    cmd = [PYTHON, str(ROOT / 'scripts/run_benchmark.py'), 'start', '--resume',
           '--run-dir', str(directory), '--benchmark', p['benchmark'], '--checkpoint', p['checkpoint'],
           '--workers', str(state['workers']), '--cpus', str(p['cpus']), '--memory-mb', str(p['memory_mb']),
           '--proxy-policy', p['proxy_policy'], '--task-timeout', str(p['task_timeout_seconds'])]
    if p.get('subset_file'):
        cmd += ['--task-subset', p['subset_file']]
    acceptance = state['launch_history'][-1].get('concurrency_acceptance')
    if acceptance:
        cmd += ['--concurrency-report', acceptance['source']]
    if state.get('inference_pool'):
        cmd += ['--service-pool', state['inference_pool']['source']]
    return json.loads(remote(state['hostname'], cmd).stdout)


def maintain_campaigns(paths, supervisor, upgrade=False):
    busy = False
    for path in paths:
        directory = Path(path)
        state = json.loads((directory / 'campaign.json').read_text())
        live = campaign_live(state, directory)
        if live:
            busy = True
            if (upgrade and not state['launch_history'][-1].get('recovery_policy')
                    and not supervisor.get('draining_campaign')):
                code = f'import os,signal; os.kill({state["pid"]}, signal.SIGUSR1)'
                remote(state['hostname'], [PYTHON, '-c', code])
                supervisor['draining_campaign'] = str(directory)
                supervisor.setdefault('events', []).append({'at_utc': now(), 'event': 'drain_legacy_coordinator', 'campaign': str(directory), 'pid': state['pid']})
            continue
        if supervisor.get('draining_campaign') == str(directory):
            supervisor.pop('draining_campaign')
        needed = any(row['status'] in {'pending', 'running'} or retry_eligible(row)
                     for row in state['tasks'].values())
        if needed:
            launched = resume(directory, state)
            supervisor.setdefault('events', []).append({'at_utc': now(), 'event': 'resume_campaign', **launched})
            busy = True
    return busy


def maintain_services(paths, supervisor):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health = supervisor.setdefault('services', {})
    for path in paths:
        path = Path(path)
        state = json.loads(path.read_text())
        item = health.setdefault(str(path), {'failures': 0, 'restarts': 0})
        try:
            with opener.open(state['base_url'] + '/models', timeout=5) as response:
                models = json.load(response)['data']
            if not any(m['id'] == state['served_model_name'] and m.get('root') == state['model_dir'] for m in models):
                raise RuntimeError('Endpoint checkpoint identity mismatch')
            item.update(failures=0, healthy_at_utc=now())
            continue
        except Exception as exc:
            item.update(failures=item['failures'] + 1, error=str(exc))
        if time.time() < item.get('startup_grace_until', 0):
            continue
        if item['failures'] < 3 or item['restarts'] >= 3:
            continue
        args = [PYTHON, str(ROOT / 'scripts/serve_model.py')]
        selection = ['--checkpoint', state['checkpoint'], '--gpus', ','.join(state['gpus'])]
        command = state['command']
        port = command[command.index('--port') + 1]
        selection += ['--port', port]
        if state.get('replica'):
            selection += ['--replica', state['replica']]
        if not item.get('stop_requested'):
            remote(state['hostname'], [*args, 'stop', *selection])
            item['stop_requested'] = time.time()
            continue
        if time.time() - item['stop_requested'] < 30:
            continue
        settings = []
        for flag in ['--max-model-len', '--max-num-seqs', '--gpu-memory-utilization']:
            settings += [flag, command[command.index(flag) + 1]]
        settings += ['--max-images', str(json.loads(command[command.index('--limit-mm-per-prompt') + 1])['image'])]
        for key, value in state.get('generation_overrides', {}).items():
            settings += ['--' + key.replace('_', '-') + '-default', str(value)]
        try:
            remote(state['hostname'], [*args, 'start', *selection, *settings])
            item.update(failures=0, restarts=item['restarts'] + 1, startup_grace_until=time.time() + 600)
            item.pop('stop_requested', None)
            supervisor.setdefault('events', []).append({'at_utc': now(), 'event': 'restart_owned_replica', 'state_file': str(path), 'previous_pid': state['pid']})
        except Exception as exc:
            item['restart_error'] = str(exc)


def find_stage(stage, config):
    expected = [str(ROOT / 'scripts/original_rl_pipeline.py'), stage, '--config', str(config)]
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            argv = [x.decode() for x in (directory / 'cmdline').read_bytes().split(b'\0') if x]
            if any(argv[i:i + len(expected)] == expected for i in range(len(argv))):
                candidate = identity(int(directory.name))
                if candidate:
                    return candidate
        except (OSError, UnicodeError, ValueError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/original-rl-pipeline.json')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'research-a-worker-1':
        parser.error('Pipeline supervisor belongs on worker-1')
    initial_config = json.loads(args.config.read_text())
    directory = ROOT / initial_config.get('state_directory', 'runs/original-rl-pipeline')
    directory.mkdir(exist_ok=True)
    lease = (directory / 'supervisor.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'phase': 'sft_evaluation', 'stage_index': 0, 'events': []}
    state.update(pid=os.getpid(), hostname=socket.gethostname(), process_identity=identity(os.getpid()))
    state['updated_at_utc'] = now()
    write_json(state_path, state)
    while True:
        cfg = json.loads(args.config.read_text())
        try:
            if not cfg['enabled']:
                state['control'] = 'disabled_by_configuration'
            elif state['phase'] == 'sft_evaluation':
                maintain_services(cfg['sft_service_states'], state)
                busy = maintain_campaigns(cfg['sft_campaigns'], state, upgrade=True)
                report = generate(cfg['sft_campaigns'], ROOT / 'reports/sft-evaluation-report', final=not busy)
                state['sft_report_status'] = report['status']
                if not busy:
                    state['phase'] = 'original_rl_stages'
                    state['events'].append({'at_utc': now(), 'event': 'sft_phase_finalized', 'report_status': report['status']})
            elif state['phase'] == 'original_rl_stages':
                stages = cfg['stages']
                if state['stage_index'] >= len(stages):
                    state['phase'] = 'complete'
                else:
                    stage = stages[state['stage_index']]
                    receipt_path = directory / (stage + '.receipt.json')
                    process_path = directory / (stage + '.process.json')
                    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                    if receipt and receipt['exit_code'] == 0:
                        state['events'].append({'at_utc': now(), 'event': 'stage_completed', 'stage': stage})
                        state['stage_index'] += 1
                        state.pop('stage_error', None)
                    elif receipt:
                        state['stage_error'] = {'stage': stage, 'receipt': str(receipt_path), 'exit_code': receipt['exit_code']}
                        failures = state.setdefault('stage_failures', {}).get(stage, 0)
                        ended = receipt.get('finished_at_utc')
                        import datetime
                        elapsed = time.time() - datetime.datetime.fromisoformat(ended).timestamp() if ended else 0
                        if failures < 2 and elapsed >= 60 * 2 ** failures:
                            archive = directory / 'failed-stage-attempts'
                            archive.mkdir(exist_ok=True)
                            receipt_path.rename(archive / f'{stage}-{failures + 1}.receipt.json')
                            if process_path.exists():
                                process_path.rename(archive / f'{stage}-{failures + 1}.process.json')
                            state['stage_failures'][stage] = failures + 1
                            state['events'].append({'at_utc': now(), 'event': 'retry_stage_from_persisted_state', 'stage': stage, 'attempt': failures + 2})
                    elif process_path.exists():
                        process = json.loads(process_path.read_text())
                        live_path = Path(f'/proc/{process["pid"]}/cmdline')
                        if ((process.get('identity') and not alive(process['identity'])) or not live_path.exists()
                                or str(ROOT / 'scripts/original_rl_pipeline.py').encode() not in live_path.read_bytes()):
                            write_json(receipt_path, {'stage': stage, 'exit_code': 1, 'finished_at_utc': now(),
                                'error': 'Stage process disappeared; resume via persisted command receipts and campaign states'})
                    else:
                        owned = find_stage(stage, args.config)
                        if owned:
                            write_json(process_path, {'pid': owned['pid'], 'identity': owned, 'stage': stage, 'adopted_at_utc': now()})
                            state['events'].append({'at_utc': now(), 'event': 'adopt_existing_stage', 'stage': stage, 'pid': owned['pid']})
                        else:
                            with (directory / (stage + '.log')).open('ab', buffering=0) as log:
                                proc = subprocess.Popen([PYTHON, str(ROOT / 'scripts/original_rl_pipeline.py'), stage, '--config', str(args.config)],
                                                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                        start_new_session=True)
                            write_json(process_path, {'pid': proc.pid, 'identity': identity(proc.pid), 'stage': stage, 'started_at_utc': now()})
                            state['events'].append({'at_utc': now(), 'event': 'stage_started', 'stage': stage, 'pid': proc.pid})
            if cfg['enabled']:
                state.pop('control', None)
            state.pop('loop_error', None)
        except Exception as exc:
            state['loop_error'] = {'at_utc': now(), 'type': type(exc).__name__, 'message': str(exc)}
        state['updated_at_utc'] = now()
        write_json(state_path, state)
        if args.once or state['phase'] == 'complete':
            print(json.dumps(state))
            break
        time.sleep(20)


if __name__ == '__main__':
    main()
