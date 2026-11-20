"""Isolated CPU-only comparison of GPFS/local disks and full QEMU snapshot reset.

Keeps production VM allocation unchanged. This measures infrastructure and reset
isolation; it does not establish task/verifier parity or full rollout throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time
import urllib.request

from check_vm_concurrency import exercise
from qemu_runtime import LocalVM, ROOT
from run_benchmark import now, write_json


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class ProbeVM(LocalVM):
    """Storage override confined to this experimental process and its own UUID."""
    def __init__(self, base, local_root=None, address='127.0.4.90'):
        super().__init__(address=address, cpus=4, memory_mb=8192,
                         accelerator='tcg', base_image=base)
        self.audit_directory = self.directory
        if local_root is not None:
            self.directory = local_root / self.id
            self.directory.mkdir(parents=True)

    def start(self):
        state = super().start()
        write_json(self.audit_directory / 'state.json', state)
        return state

    def monitor(self, command, timeout=300):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(self.directory / 'monitor.sock'))

            def receive():
                content = b''
                while b'(qemu)' not in content:
                    block = connection.recv(65536)
                    if not block:
                        raise RuntimeError('Monitor closed before command completion')
                    content += block
                return content.decode(errors='replace')

            receive()
            connection.sendall(command.encode() + b'\n')
            return receive()


def guest(vm, program):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f'http://{vm.address}:5000/execute',
                                     data=json.dumps({'command': ['python', '-c', program]}).encode(),
                                     headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=90) as response:
        result = json.load(response)
    if result.get('returncode') != 0:
        raise RuntimeError('Guest probe failed: ' + str(result))
    return result.get('output', '').strip()


def cpu_seconds(vm):
    try:
        values = Path(f'/proc/{vm.process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        return (int(values[11]) + int(values[12])) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError):
        return None


def stage_base(source, local, report, save):
    target = local / source.name
    record = {'source': str(source), 'target': str(target), 'bytes': source.stat().st_size,
              'started_at_utc': now()}
    report.setdefault('staging', []).append(record)
    save()
    start = time.monotonic()
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        temporary = target.with_suffix('.copying')
        subprocess.run(['cp', '--reflink=auto', '--sparse=always', str(source), str(temporary)], check=True)
        temporary.rename(target)
    record['copy_seconds'] = time.monotonic() - start
    record['source_sha256'] = digest(source)
    record['local_sha256'] = digest(target)
    if record['source_sha256'] != record['local_sha256']:
        raise RuntimeError('Local VM image is not byte-identical to the frozen source')
    record['finished_at_utc'] = now()
    save()
    return target


def probe(base, storage, arm, image_name, repeat, report, save, snapshots):
    row = {'image': image_name, 'arm': arm, 'repeat': repeat, 'started_at_utc': now(),
           'status': 'running', 'base_image': str(base)}
    report['runs'].append(row)
    save()
    vm = ProbeVM(base, storage)
    try:
        started = time.monotonic()
        row['vm'] = vm.start()
        save()
        vm.wait_ready(600)
        row['cold_boot_seconds'] = time.monotonic() - started
        row['guest_control'] = exercise(vm, 3)
        if snapshots:
            guest(vm, "from pathlib import Path; import pyautogui; Path('/tmp/cua-envbench-baseline').write_text('clean'); pyautogui.moveTo(400,300)")
            baseline = guest(vm, "import json,pyautogui; print(json.dumps(list(pyautogui.position())))")
            started = time.monotonic()
            output = vm.monitor('savevm cua_envbench_clean')
            if any(word in output.lower() for word in ['error', 'failed', 'not supported']):
                raise RuntimeError('Snapshot save failed: ' + output[-1000:])
            row['snapshot_save_seconds'] = time.monotonic() - started
            row['snapshot_resets'] = []
            save()
            for iteration in range(3):
                guest(vm, "from pathlib import Path; import pyautogui; Path('/tmp/cua-envbench-dirty').write_text('dirty'); Path('/tmp/cua-envbench-baseline').write_text('changed'); pyautogui.moveTo(100,100)")
                started = time.monotonic()
                output = vm.monitor('loadvm cua_envbench_clean')
                if any(word in output.lower() for word in ['error', 'failed', 'not supported']):
                    raise RuntimeError('Snapshot restore failed: ' + output[-1000:])
                vm.monitor('cont')
                vm.wait_ready(120)
                reset = {'iteration': iteration + 1, 'restore_and_ready_seconds': time.monotonic() - started}
                state = json.loads(guest(vm, "from pathlib import Path; import json,pyautogui; print(json.dumps({'dirty_exists':Path('/tmp/cua-envbench-dirty').exists(),'baseline':Path('/tmp/cua-envbench-baseline').read_text(),'pointer':list(pyautogui.position())}))"))
                reset['state'] = state
                reset['isolation_passed'] = not state['dirty_exists'] and state['baseline'] == 'clean' and state['pointer'] == json.loads(baseline)
                reset['guest_control'] = exercise(vm, 3)
                row['snapshot_resets'].append(reset)
                save()
                if not reset['isolation_passed']:
                    raise RuntimeError('Snapshot did not restore disk and GUI state')
        row['status'] = 'passed'
    except Exception as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        row['qemu_cpu_seconds'] = cpu_seconds(vm) if vm.process else None
        vm.stop()
        row['cleanup_passed'] = vm.process is None or vm.process.poll() is not None
        row['finished_at_utc'] = now()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', choices=['osworld', 'scienceboard'], default=['osworld', 'scienceboard'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/environment-runtime-comparison.json')
    args = parser.parse_args()
    if socket.gethostname() != 'research-a-worker-0':
        parser.error('Run probes on worker-0, leaving the worker-1 evaluation VM pool alone')
    if not 1 <= args.repeats <= 5:
        parser.error('repeats must be 1..5')
    import fcntl
    lease = (ROOT / 'runs/environment-runtime-comparison.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.report.exists():
        parser.error('Preserve earlier experiment reports; select a new --report path')
    local = Path(f'/tmp/cua-rl-envbench-{os.getuid()}')
    local.mkdir(exist_ok=True)
    if shutil.disk_usage(local).free < 150 * 1024**3:
        parser.error('Require 150 GiB free local space before staging both images and snapshots')
    report = {'status': 'running', 'started_at_utc': now(), 'hostname': socket.gethostname(),
              'cpu_per_vm': 4, 'memory_mb_per_vm': 8192, 'gpu_allocation': 0,
              'local_root': str(local), 'repeats': args.repeats, 'runs': [],
              'scope': 'Alternating sequential cold boots plus three full snapshot restores per local-disk guest; same images and resources. Infrastructure microbenchmark, not full rollout throughput or verifier equivalence.',
              'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    def save():
        write_json(args.report, report)
    save()
    bases = {'osworld': ROOT / 'data/osworld-vm/Ubuntu.qcow2',
             'scienceboard': ROOT / 'data/scienceboard-vm/Ubuntu2_scienceboard.qcow2'}
    try:
        for image_name in args.images:
            shared = bases[image_name]
            local_base = stage_base(shared, local, report, save)
            for repeat in range(1, args.repeats + 1):
                arms = [('shared_disk', shared, None), ('local_disk', local_base, local / 'vms')]
                if repeat % 2 == 0:
                    arms.reverse()
                for arm, base, storage in arms:
                    probe(base, storage, arm, image_name, repeat, report, save, snapshots=arm == 'local_disk')
        report['status'] = 'passed' if all(row['status'] == 'passed' for row in report['runs']) else 'completed_with_failures'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        summary = {}
        for image_name in args.images:
            summary[image_name] = {}
            for arm in ['shared_disk', 'local_disk']:
                rows = [r for r in report['runs'] if r['image'] == image_name and r['arm'] == arm]
                cold = [r['cold_boot_seconds'] for r in rows if 'cold_boot_seconds' in r]
                resets = [s['restore_and_ready_seconds'] for r in rows for s in r.get('snapshot_resets', []) if s['isolation_passed']]
                summary[image_name][arm] = {'attempts': len(rows), 'failures': sum(r['status'] != 'passed' for r in rows),
                    'cold_boot_median_seconds': statistics.median(cold) if cold else None,
                    'verified_snapshot_resets': len(resets), 'snapshot_reset_median_seconds': statistics.median(resets) if resets else None}
        report.update(summary=summary, finished_at_utc=now())
        save()
    print(json.dumps({'status': report['status'], 'report': str(args.report), 'summary': report['summary']}))


if __name__ == '__main__':
    main()
