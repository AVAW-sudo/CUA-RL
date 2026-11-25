"""Supervise a private OSWorld service and the GRPO trainer in one AIS allocation."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from ais_desktop_service import stop


ROOT = Path(__file__).resolve().parents[1]


def local_controller(directory):
    """Called by the trainer after its allocation, model and rates preflight."""
    (directory / 'start').touch()
    ready = directory / 'ready.json'
    deadline = time.monotonic() + 3600
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('Local OSWorld service did not register within one hour. See AIS desktop logs.')
        time.sleep(1)
    return json.loads(ready.read_text())['controller']


def start_desktop(directory):
    print('AIS RL: starting the local OSWorld controller and worker', flush=True)
    return subprocess.Popen(
        ['bash', str(ROOT / 'AIS_ENTRY.txt'), 'rl-desktop', '--host', '127.0.0.1',
         '--advertise-host', '127.0.0.1', '--port', '0', '--worker-port', '0',
         '--ready-file', str(directory / 'ready.json')],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), start_new_session=True)


def run_job(command):
    # Stop the trainer while its desktop service is still available for cleanup.
    with tempfile.TemporaryDirectory(prefix='cua-rl-job-') as tmp, ExitStack() as desktop_cleanup, ExitStack() as trainer_cleanup:
        directory = Path(tmp)
        env = dict(os.environ, CUA_LOCAL_DESKTOP_DIR=tmp)
        # Both the trainer and its Ray workers must reach the local controller directly.
        bypass = ','.join(filter(None, ('127.0.0.1', 'localhost', env.get('no_proxy'), env.get('NO_PROXY'))))
        env.update(NO_PROXY=bypass, no_proxy=bypass)
        trainer = subprocess.Popen(command, env=env, start_new_session=True)
        trainer_cleanup.callback(stop, trainer, 60)
        while not (directory / 'start').exists():
            if trainer.poll() is not None:
                return trainer.returncode
            time.sleep(1)
        desktop = start_desktop(directory)
        desktop_cleanup.callback(stop, desktop, 65)
        while trainer.poll() is None:
            if desktop.poll() is not None:
                raise RuntimeError(f'Local OSWorld service exited during the RL job (code {desktop.returncode})')
            time.sleep(1)
        return trainer.returncode


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    code = run_job(sys.argv[1:])
    sys.exit(code if code >= 0 else 128 - code)
