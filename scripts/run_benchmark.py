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

ROOT = Path(os.environ.get('CUA_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
ADAPTERS = ['evaluate_sft.py', 'eval_qemu_provider.py', 'qemu_runtime.py', 'run_benchmark.py', 'scienceboard_compat.py']


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
        check_endpoint(service)
        command = [sys.executable, str(args.runtime_dir / 'evaluate_sft.py'),
                   '--benchmark', args.benchmark, '--checkpoint', args.checkpoint,
                   '--accelerator', 'tcg', '--workers', '1', '--cpus', str(args.cpus),
                   '--memory-mb', str(args.memory_mb), '--recording-mode', 'steps-only',
                   '--proxy-policy', args.proxy_policy, '--tasks', task, '--run-dir', str(output)]
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
        score = row['score']
        if row['status'] != 'completed' or score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError('; '.join(row['errors']) or 'Episode has no valid verifier result')
        result.update(status='completed', score=score, action_records=row['action_records'], directory=row['directory'])
    except Exception as exc:
        result.update(status='infrastructure_error', score=None, error=str(exc))
    finally:
        if process is not None:
            stop_process(process, output)
        result['finished_at_utc'] = now()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'run', 'status'])
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--benchmark', choices=['osworld', 'scienceboard'], default='osworld')
    p.add_argument('--checkpoint', choices=['osworld-sft', 'scienceboard-sft'], default='osworld-sft')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--memory-mb', type=int, default=8192)
    p.add_argument('--proxy-policy', choices=['upstream', 'direct'], default='upstream')
    p.add_argument('--task-timeout', type=int, default=3600)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--only-tasks', nargs='+', help='Schedule only these domain/id entries; retain the full benchmark denominator')
    args = p.parse_args()
    output = args.run_dir.resolve()
    if args.action == 'status':
        print((output / 'summary.json').read_text())
        return
    if socket.gethostname() not in {'research-a-worker-0', 'research-a-worker-1'}:
        p.error('Campaigns run only on worker-0/1')
    if not 1 <= args.workers <= 4:
        p.error('Initial campaign concurrency is limited to four; higher concurrency needs validation')
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
    service_file = 'inference-service.json' if args.checkpoint == 'osworld-sft' else 'scienceboard-inference-service.json'
    service = json.loads((ROOT / 'runs' / service_file).read_text())
    check_endpoint(service)
    tasks, task_hash = official_tasks(args.benchmark)
    path = output / 'campaign.json'
    args.runtime_dir = output / 'runtime'
    if not path.exists():
        args.runtime_dir.mkdir(exist_ok=False)
        for name in ADAPTERS:
            (args.runtime_dir / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    elif not args.runtime_dir.is_dir():
        p.error('Legacy campaign has no immutable runtime snapshot; start a new version')
    # A legacy campaign must keep exactly its original runtime file set.
    # New campaigns include all current adapters; missing pinned files always fail.
    pinned_names = json.loads(path.read_text())['protocol']['adapter_sha256'] if path.exists() else ADAPTERS
    protocol = {'benchmark': args.benchmark, 'checkpoint': args.checkpoint,
                'model_revision': service['model_revision'], 'server_command': service['command'],
                'task_list_sha256': task_hash, 'adapter_sha256': fingerprint(args.runtime_dir, pinned_names),
                'accelerator': 'tcg', 'recording_mode': 'steps-only', 'cpus': args.cpus,
                'memory_mb': args.memory_mb, 'proxy_policy': args.proxy_policy,
                'task_timeout_seconds': args.task_timeout, 'samples_per_task': 1}
    if path.exists():
        if not args.resume:
            p.error('Existing campaign requires --resume; nothing is overwritten')
        state = json.loads(path.read_text())
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
            last.update(status='interrupted', score=None)
            row.update(status='pending', score=None)
    else:
        state = {'created_at_utc': now(), 'hostname': socket.gethostname(), 'protocol': protocol,
                 'tasks': {t: {'domain': t.split('/')[0], 'status': 'pending', 'score': None, 'attempts': []} for t in tasks}}
    queue = select_queue(tasks, state['tasks'], args.retry_errors, args.only_tasks)
    state.update(status='running', pid=os.getpid(), workers=args.workers, pause_reason=None)
    coordinator_code = Path(__file__).read_bytes()
    coordinator_dir = output / 'coordinators'
    coordinator_dir.mkdir(exist_ok=True)
    coordinator_path = coordinator_dir / f'{time.time_ns()}-{os.getpid()}.py'
    coordinator_path.write_bytes(coordinator_code)
    state.setdefault('launch_history', []).append({
        'at_utc': now(), 'pid': os.getpid(), 'retry_errors': args.retry_errors,
        'scheduled_tasks': queue.copy(), 'coordinator_snapshot': str(coordinator_path),
        'coordinator_sha256': hashlib.sha256(coordinator_code).hexdigest()})
    stopped = [False]
    def interrupted(signum, frame):
        stopped[0] = True
        state['pause_reason'] = f'Received signal {signum}'
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def save():
        state['updated_at_utc'] = now()
        write_json(path, state)
        write_json(output / 'summary.json', snapshot(state))
    save()
    consecutive_errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        while queue or active:
            while queue and len(active) < args.workers and not stopped[0]:
                task = queue.pop(0)
                row = state['tasks'][task]
                attempt = output / 'attempts' / task.replace('/', '__') / f'{len(row["attempts"]) + 1:03d}'
                attempt.parent.mkdir(parents=True, exist_ok=True)
                row['attempts'].append({'attempt_dir': str(attempt), 'status': 'running'})
                row['status'] = 'running'
                active[pool.submit(episode, args, task, attempt, service, protocol['adapter_sha256'], lambda: stopped[0])] = task
                save()
            if not active:
                break
            done, _ = concurrent.futures.wait(active, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                result = future.result()
                row = state['tasks'][task]
                row['attempts'][-1] = result
                row.update(status=result['status'], score=result['score'], error=result.get('error'))
                consecutive_errors = consecutive_errors + 1 if result['status'] != 'completed' else 0
                print(json.dumps({'task': task, **result}), flush=True)
                if consecutive_errors >= 3:
                    stopped[0] = True
                    state['pause_reason'] = 'Three consecutive infrastructure/verifier errors; inspect before resuming'
                save()
            save()
    summary = aggregate(list(state['tasks'].values()))
    state['status'] = 'complete' if summary['completed'] == summary['total'] else ('paused_for_review' if stopped[0] else 'finished_with_errors')
    save()


if __name__ == '__main__':
    main()
