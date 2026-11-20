#!/usr/bin/env python3
"""Drain an old coordinator without interrupting its episodes, then resize it.

The legacy coordinator predates SIGUSR1 draining. Suspend only that coordinator
while its independent evaluation subprocesses finish, then terminate/resume it
so it reaps results and releases its lease. A timeout or error resumes normal
scheduling. Never signal the evaluation process groups or the model service.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from run_benchmark import ROOT, check_concurrency, fingerprint, now, write_json


def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def episode_pids(pid):
    children = set()
    for path in Path(f'/proc/{pid}/task').glob('*/children'):
        try:
            children.update(map(int, path.read_text().split()))
        except FileNotFoundError:
            pass
    return {child for child in children if live(child)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--concurrency-report', type=Path)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    directory = args.run_dir.resolve()
    state_path = directory / 'campaign.json'
    state = json.loads(state_path.read_text())
    protocol = state['protocol']
    if state['hostname'] != socket.gethostname() or state['status'] != 'running':
        parser.error('Resize a running campaign on its original worker')
    check_concurrency(args.workers, protocol['cpus'], protocol['memory_mb'],
                      protocol['benchmark'], args.concurrency_report)
    pid = state['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if str(directory).encode() not in command or not any(c.endswith(b'/run_benchmark.py') for c in command):
        raise RuntimeError('Coordinator PID identity mismatch')
    runtime_hashes = fingerprint(directory / 'runtime', protocol['adapter_sha256'])
    if runtime_hashes != protocol['adapter_sha256']:
        raise RuntimeError('Pinned runtime changed')
    report_path = directory / f'resize-{time.time_ns()}.json'
    report = {'started_at_utc': now(), 'old_pid': pid, 'old_workers': state['workers'],
              'new_workers': args.workers, 'status': 'draining', 'runtime_sha256': runtime_hashes}
    write_json(report_path, report)
    suspended = False
    try:
        os.kill(pid, signal.SIGSTOP)
        suspended = True
        deadline = time.monotonic() + args.timeout
        while Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in {'T', 't'}:
            if time.monotonic() > deadline:
                raise TimeoutError('Coordinator did not suspend')
            time.sleep(.1)
        state = json.loads(state_path.read_text())
        preserved = {k: r['score'] for k, r in state['tasks'].items() if r['status'] == 'completed'}
        running = {k: r['attempts'][-1]['attempt_dir'] for k, r in state['tasks'].items() if r['status'] == 'running'}
        expected_pids = set()
        for output in running.values():
            path = Path(output)
            process = path.parent / (path.name + '.process.json')
            if not process.exists():
                raise RuntimeError('Caught a launch in progress; safely resume and retry later')
            child = json.loads(process.read_text())['pid']
            if live(child):
                argv = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
                if str(path).encode() not in argv:
                    raise RuntimeError('Episode PID identity mismatch')
                expected_pids.add(child)
        if episode_pids(pid) != expected_pids:
            raise RuntimeError('Campaign state and direct children differ; safely resume and retry later')
        report.update(active_tasks=running, active_pids=sorted(expected_pids), preserved_results=preserved)
        write_json(report_path, report)
        print(json.dumps({'status': 'draining', 'active': len(expected_pids), 'report': str(report_path)}), flush=True)
        while any(live(child) for child in expected_pids):
            if time.monotonic() > deadline:
                raise TimeoutError('Drain deadline exceeded; original scheduling restored')
            time.sleep(3)
        # All children have naturally exited. No model trajectory is interrupted.
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        suspended = False
        deadline = time.monotonic() + 90
        while live(pid):
            if time.monotonic() > deadline:
                raise TimeoutError('Old coordinator did not finish reaping results')
            time.sleep(.5)
        after = json.loads(state_path.read_text())
        if after['protocol'] != protocol or any(after['tasks'][k]['score'] != score or
                after['tasks'][k]['status'] != 'completed' for k, score in preserved.items()):
            raise RuntimeError('Previously completed results or pinned protocol changed')
        if any(after['tasks'][k]['status'] == 'running' for k in running):
            raise RuntimeError('Old coordinator did not reconcile all drained episodes')
        report['drained_outcomes'] = {k: {'status': after['tasks'][k]['status'],
                                        'score': after['tasks'][k]['score']} for k in running}
        cmd = [sys.executable, str(ROOT / 'scripts/run_benchmark.py'), 'start', '--run-dir', str(directory),
               '--benchmark', protocol['benchmark'], '--checkpoint', protocol['checkpoint'],
               '--workers', str(args.workers), '--cpus', str(protocol['cpus']),
               '--memory-mb', str(protocol['memory_mb']), '--proxy-policy', protocol['proxy_policy'],
               '--task-timeout', str(protocol['task_timeout_seconds']), '--resume']
        if args.concurrency_report:
            cmd += ['--concurrency-report', str(args.concurrency_report.resolve())]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        report.update(status='resumed', resumed_at_utc=now(), launch=json.loads(result.stdout),
                      interrupted_episodes=0, preserved_result_count=len(preserved))
        print(result.stdout, end='', flush=True)
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        if suspended and live(pid):
            os.kill(pid, signal.SIGCONT)
            report['original_coordinator_resumed'] = True
        write_json(report_path, report)


if __name__ == '__main__':
    main()
