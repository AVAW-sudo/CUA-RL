"""Repeat trusted GUI probes in an already staged, pinned udocker image on worker-0.

This is a capability/latency experiment. PRoot provides no strong security
isolation, and these results must not be treated as OSWorld benchmark scores.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import uuid

from qemu_runtime import ROOT
from run_benchmark import now, write_json


def run_owned(command, env):
    import psutil
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=240)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            # Track only descendants of this exact probe before stopping it.
            children = psutil.Process(process.pid).children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--udocker', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--container-id', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gnome', action='store_true')
    parser.add_argument('--office', action='store_true')
    parser.add_argument('--applications', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if socket.gethostname() != 'research-a-worker-0':
        parser.error('Only worker-0 CPU resources are used')
    if str(uuid.UUID(args.container_id)) != args.container_id:
        parser.error('Use the exact container UUID')
    if not 1 <= args.repeats <= 5 or args.report.exists():
        parser.error('Use 1..5 repeats and a new report path')
    lease = (ROOT / 'runs/userspace-desktop-probe.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    for address, port in [('127.0.0.1', 15994), ('127.0.4.92', 19594), ('127.0.0.1', 19595)]:
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind((address, port))
    container = args.repository / 'containers' / args.container_id / 'ROOT'
    if not container.is_dir() or not container.resolve().is_relative_to(Path('/tmp')):
        parser.error('Expected a disposable container filesystem under /tmp')
    provenance = json.loads((ROOT / 'reports/container-image-provenance.json').read_text())
    local_manifest = json.loads((args.repository / 'repos/ghcr.io/cua-lite/lite.osworld/latest/manifest').read_text())
    for key in ['config', 'layers']:
        if local_manifest[key] != provenance['manifest'][key]:
            raise RuntimeError('Staged image differs from the inspected pinned manifest')
    source = ROOT / 'scripts/userspace_desktop_guest.py'
    destination = container / 'tmp/cua-desktop-guest.py'
    shutil.copyfile(source, destination)
    report = {'status': 'running', 'started_at_utc': now(), 'hostname': socket.gethostname(),
              'gpu_allocation': 0, 'image_digest': provenance['digest'], 'image_reference': provenance['reference'],
              'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'gnome_requested': args.gnome,
              'office_requested': args.office, 'applications_requested': args.applications,
              'scope': 'Scripted capability probe and microbenchmark, not full desktop/task/verifier parity',
              'isolation': 'PRoot P1; not a security sandbox; no model-generated actions',
              'runs': []}
    output = ROOT / 'runs/userspace-desktop-probe' / args.report.stem
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / source.name)
    env = dict(os.environ, UDOCKER_DIR=str(args.repository))
    write_json(args.report, report)
    try:
        for iteration in range(args.repeats):
            name = 'cua-desktop-' + uuid.uuid4().hex
            guest_output = '/tmp/' + name
            command = [str(args.udocker), 'run', '--nobanner', '--user=user', '--containerauth', '--nosysdirs',
                '--volume=/proc:/proc', '--volume=/dev/null:/dev/null', '--volume=/dev/zero:/dev/zero',
                '--volume=/dev/urandom:/dev/urandom', '--volume=/dev/random:/dev/random', '--workdir=/tmp',
                '--env=CUA_USERSPACE_PROBE=1',
                args.container_id, '/usr/bin/python3', '/tmp/cua-desktop-guest.py', '--output', guest_output]
            if args.gnome:
                command.append('--gnome')
            if args.office:
                command.append('--office')
            if args.applications:
                command.append('--applications')
            row = {'iteration': iteration + 1, 'command': command, 'started_at_utc': now(), 'status': 'running'}
            report['runs'].append(row)
            write_json(args.report, report)
            started = time.monotonic()
            result = run_owned(command, env)
            row.update(exit_code=result.returncode, process_seconds=time.monotonic() - started)
            directory = output / str(iteration + 1)
            directory.mkdir()
            (directory / 'stdout.log').write_text(result.stdout)
            (directory / 'stderr.log').write_text(result.stderr)
            guest_dir = container / 'tmp' / name
            if guest_dir.is_dir():
                shutil.copytree(guest_dir, directory / 'guest', ignore=shutil.ignore_patterns('runtime', 'browser-profile', 'calc-profile'))
            data = guest_dir / 'report.json'
            if not data.is_file():
                raise RuntimeError(f'Guest produced no report, see {directory}')
            row['result'] = json.loads(data.read_text())
            row['status'] = row['result']['status']
            row['finished_at_utc'] = now()
            write_json(args.report, report)
        report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['runs']) else 'completed_with_failures'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['finished_at_utc'] = now()
        write_json(args.report, report)
    print(json.dumps({'status': report['status'], 'report': str(args.report)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
