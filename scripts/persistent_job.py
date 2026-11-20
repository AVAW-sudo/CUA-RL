"""Detached command receipts and exact process identities for stage recovery.

The runner survives its caller. A resumed stage attaches to the same runner;
it never infers success from a missing PID or starts over a surviving child.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'hostname': socket.gethostname()}
    except (OSError, ValueError, IndexError):
        return None


def alive(value):
    return bool(value and identity(value['pid']) == value)


class Job:
    def __init__(self, directory):
        self.directory = Path(directory)

    def state(self):
        path = self.directory / 'process.json'
        return json.loads(path.read_text()) if path.exists() else None

    @property
    def pid(self):
        return (self.state() or {}).get('runner', {}).get('pid')

    def poll(self):
        if hasattr(self, '_process'):
            self._process.poll()  # reap our own finished runner
        receipt = self.directory / 'receipt.json'
        if receipt.exists():
            if hasattr(self, '_process'):
                self._process.wait(timeout=5)
            return json.loads(receipt.read_text())['exit_code']
        state = self.state()
        if state and alive(state.get('runner')):
            return None
        if state and alive(state.get('child')):
            # The parent alone was lost. The child still owns its resources.
            # Await a result, never duplicate it or claim exit status zero.
            return None
        raise RuntimeError(f'Job disappeared without a receipt: {self.directory}')


def attach(directory):
    job = Job(directory)
    if (job.directory / 'process.json').exists() or (job.directory / 'receipt.json').exists():
        return job
    return None


def start(directory, command, cwd):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = {'command': [str(x) for x in command], 'cwd': str(cwd)}
    lease = (directory / 'launch.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX)
    path = directory / 'spec.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError('Cannot change a persisted job command')
    existing = attach(directory)
    if existing:
        lease.close()
        return existing
    write(path, spec)
    with (directory / 'output.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--directory', str(directory)],
                                cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 15
    while not (directory / 'process.json').exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('Persistent runner did not register; inspect ' + str(directory))
        time.sleep(.05)
    lease.close()
    job = Job(directory)
    job._process = proc
    return job


def run(directory):
    lease = (directory / 'runner.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if attach(directory):
        raise RuntimeError('A prior runner exists; do not duplicate its child')
    spec = json.loads((directory / 'spec.json').read_text())
    record = {'runner': identity(os.getpid()), 'started_unix': time.time(), 'child': None}
    write(directory / 'process.json', record)
    try:
        proc = subprocess.Popen(spec['command'], cwd=spec['cwd'], stdin=subprocess.DEVNULL)
        record['child'] = identity(proc.pid)
        write(directory / 'process.json', record)
        code = proc.wait()
    except Exception as exc:
        record['error'] = repr(exc)
        code = 1
    write(directory / 'receipt.json', {**record, 'exit_code': code, 'finished_unix': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args().directory)
