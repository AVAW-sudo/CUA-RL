"""Host-local outer watchdog: revive only the owned supervisor, never its children."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from persistent_job import alive, identity, write
from run_benchmark import ROOT, now


def tick(directory, config, command, stale_seconds=600):
    state_path = directory / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Completion/explicit disable must not cause an endless restart loop.
    enabled = json.loads(config.read_text()).get('enabled', False)
    result = {'at_utc': now(), 'watchdog': identity(os.getpid()), 'enabled': enabled,
              'phase': state.get('phase'), 'action': 'observe'}
    if not enabled or state.get('phase') == 'complete':
        return result
    owned = state.get('process_identity')
    if alive(owned):
        stamp = state.get('updated_at_utc')
        age = time.time() - datetime.datetime.fromisoformat(stamp).timestamp() if stamp else 0
        result['heartbeat_age_seconds'] = age
        if age > stale_seconds:
            # Signal this PID alone. Stage and episode processes keep running.
            os.kill(owned['pid'], signal.SIGTERM)
            result['action'] = 'terminate_stale_supervisor_only'
        return result
    # The supervisor lock also closes the launch-before-state-write window.
    lease = (directory / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        result['action'] = 'supervisor_lock_held'
        return result
    # Release before spawn; the child itself holds the authoritative lock.
    lease.close()
    with (directory / 'supervisor.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result.update(action='restart_supervisor', launched=identity(proc.pid))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/original-rl-pilot-pipeline.json')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'research-a-worker-1':
        parser.error('Run on worker-1')
    cfg = json.loads(args.config.read_text())
    directory = ROOT / cfg.get('state_directory', 'runs/original-rl-pipeline')
    directory.mkdir(parents=True, exist_ok=True)
    lease = (directory / 'watchdog.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [os.sys.executable, str(ROOT / 'scripts/pipeline_supervisor.py'), '--config', str(args.config)]
    while True:
        try:
            result = tick(directory, args.config, command)
        except Exception as exc:
            result = {'at_utc': now(), 'error': repr(exc), 'action': 'retry_next_poll'}
        write(directory / 'watchdog.json', result)
        if result.get('action') not in {'observe', 'supervisor_lock_held'}:
            with (directory / 'watchdog-events.jsonl').open('a') as log:
                log.write(json.dumps(result) + '\n')
        if args.once:
            print(json.dumps(result))
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
