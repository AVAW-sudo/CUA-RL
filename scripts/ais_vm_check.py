"""Foreground AIS entry for testing disposable OSWorld guests on the allocated node."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import functools
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPException

from qemu_runtime import require_kvm

ROOT = Path(os.environ.get('CUA_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
GUEST = Path(__file__).with_name('ais_vm_guest_check.py')
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


class Qemu:
    def __init__(self, prefix):
        self.env = os.environ.copy()
        self.loader = []
        self.firmware = []
        self.vga = 'virtio-vga'
        if prefix is None and not all(shutil.which(p) for p in ('qemu-img', 'qemu-system-x86_64')):
            cache = ROOT.parent.parent / 'ENV/cache'
            prefix = cache / 'ais-qemu-jammy/root'
            if not (prefix / 'usr/bin/qemu-system-x86_64').is_file():
                prefix = cache / 'cua-rl/qemu-system/root'
        self.bin = prefix / 'usr/bin' if prefix else None
        if prefix:
            libs = [prefix / p for p in ('usr/lib/x86_64-linux-gnu', 'lib/x86_64-linux-gnu')]
            search = ':'.join(map(str, libs))
            ld = libs[1] / 'ld-linux-x86-64.so.2'
            if ld.is_file():
                self.loader = [str(ld), '--library-path', search]
            else:
                previous = self.env.get('LD_LIBRARY_PATH')
                self.env['LD_LIBRARY_PATH'] = search + ':' + previous if previous else search
            self.env['QEMU_MODULE_DIR'] = str(libs[0] / 'qemu')
            bios = prefix / 'usr/share/seabios'
            self.firmware = ['-L', str(prefix / 'usr/share/qemu'), '-bios', str(bios / 'bios-256k.bin')]
            self.vga += ',romfile=' + str(bios / 'vgabios-virtio.bin')

    def cmd(self, name, *args):
        return [*self.loader, str(self.bin / name) if self.bin else name, *map(str, args)]

    def run(self, name, *args):
        return subprocess.check_output(self.cmd(name, *args), env=self.env, text=True,
                                       stderr=subprocess.STDOUT, timeout=60).strip()


class VM:
    def __init__(self, tools, scratch, output, index):
        self.tools = tools
        self.work = scratch / str(index)
        self.logs = output / f'vm-{index}'
        self.work.mkdir()
        self.logs.mkdir()
        self.process = None
        self.token = uuid.uuid4().hex
        self.steps = {}

    def start(self, base, accel, cpus, ram, worker_address=None, template=None):
        disk = self.work / 'disk.qcow2'
        if template is None:
            self.tools.run('qemu-img', 'create', '-f', 'qcow2', '-F', 'qcow2', '-b', base, disk)
        else:
            metadata = json.loads((template / 'template.json').read_text())
            if metadata['machine'] != [str(Path(base).resolve()), accel, cpus, ram]:
                raise ValueError('SFT snapshot does not match the requested guest')
            subprocess.run(['cp', '--reflink=auto', '--sparse=always', template / 'clean.qcow2', disk], check=True)
            disk.chmod(0o600)
        self.address = worker_address or '127.0.0.1'
        if worker_address:
            self.ports = {}
            with ExitStack() as stack:
                for guest_port in (5000, 9222, 8080, 8000, 8006):
                    sock = stack.enter_context(socket.socket())
                    sock.bind((self.address, 0))
                    self.ports[guest_port] = sock.getsockname()[1]
            self.port = self.ports[5000]
            forwards = ','.join(f'hostfwd=tcp:{self.address}:{host}-:{guest}' for guest, host in self.ports.items())
        else:
            with socket.socket() as port:
                port.bind((self.address, 0))
                self.port = port.getsockname()[1]
            forwards = f'hostfwd=tcp:{self.address}:{self.port}-:5000'
        args = self.tools.cmd('qemu-system-x86_64', '-machine', 'q35', '-accel',
                              'kvm' if accel == 'kvm' else 'tcg,thread=multi',
                              '-cpu', 'host' if accel == 'kvm' else 'max', '-smp', cpus, '-m', ram,
                              *self.tools.firmware, '-drive', f'file={disk},if=virtio,format=qcow2',
                              '-device', self.tools.vga, '-device', 'qemu-xhci', '-device', 'usb-tablet',
                              '-netdev', 'user,id=n0,' + forwards,
                              '-device', 'virtio-net-pci,netdev=n0,romfile=', '-display', 'none',
                              '-vnc', 'unix:' + str(self.work / 'vnc'), '-qmp',
                              f'unix:{self.work}/qmp,server=on,wait=off', '-serial',
                              'file:' + str(self.work / 'serial.log'), '-no-reboot')
        if template is not None:
            args += ['-loadvm', 'sft-clean']
        with (self.work / 'qemu.log').open('wb') as log:
            self.process = subprocess.Popen(args, env=self.tools.env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=log, start_new_session=not worker_address)
        self.started = time.monotonic()

    def alive(self):
        if self.process.poll() is not None:
            raise RuntimeError('QEMU exited: ' + (self.work / 'qemu.log').read_text(errors='replace')[-1500:])

    def qmp(self, command, arguments=None):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(180)
            sock.connect(str(self.work / 'qmp'))
            with sock.makefile('rwb', buffering=0) as stream:
                if 'QMP' not in json.loads(stream.readline()):
                    raise RuntimeError('Invalid QMP greeting')
                for payload in [{'execute': 'qmp_capabilities'},
                                {'execute': command, 'arguments': arguments or {}}]:
                    sock.sendall(json.dumps(payload).encode() + b'\r\n')
                    while True:
                        response = json.loads(stream.readline())
                        if 'error' in response:
                            raise RuntimeError(str(response['error']))
                        if 'return' in response:
                            break
                return response['return']

    def machine_ready(self):
        deadline = time.monotonic() + 20
        while True:
            self.alive()
            try:
                status = self.qmp('query-status')
                with socket.socket(socket.AF_UNIX) as vnc:
                    vnc.settimeout(5)
                    vnc.connect(str(self.work / 'vnc'))
                    greeting = vnc.recv(12, socket.MSG_WAITALL).decode().strip()
                if not status['running'] or not greeting.startswith('RFB '):
                    raise RuntimeError(f'Unexpected machine state: {status}, VNC {greeting}')
                return {'qmp': status['status'], 'vnc': greeting}
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)

    def guest(self, code, *args):
        req = urllib.request.Request(f'http://{self.address}:{self.port}/execute',
                                    data=json.dumps({'command': ['python', '-c', code, *args]}).encode(),
                                    headers={'Content-Type': 'application/json'})
        with HTTP.open(req, timeout=125) as response:
            result = json.load(response)
        if result.get('returncode') != 0:
            raise RuntimeError(str(result.get('error', result))[-1200:])
        return result['output'].strip()

    def screenshot(self, name):
        from PIL import Image, ImageStat
        with HTTP.open(f'http://{self.address}:{self.port}/screenshot', timeout=8) as response:
            png = response.read()
        with Image.open(io.BytesIO(png)) as im:
            if im.format != 'PNG' or max(ImageStat.Stat(im.convert('RGB').resize((64, 36))).stddev) < 5:
                raise ValueError('Empty or blank guest screenshot')
            dimensions = im.size
        (self.logs / name).write_bytes(png)
        return {'size': dimensions, 'png_bytes': len(png)}

    def wait_desktop(self, timeout, name):
        deadline = time.monotonic() + timeout
        next_log = time.monotonic() + 30
        while True:
            self.alive()
            try:
                return self.screenshot(name)
            except (OSError, ValueError, HTTPException) as exc:
                if time.monotonic() >= deadline:
                    try:
                        from PIL import Image
                        frame = self.work / 'console.ppm'
                        self.qmp('screendump', {'filename': str(frame)})
                        with Image.open(frame) as image:
                            image.save(self.logs / 'console-at-timeout.png')
                    except (OSError, ValueError, RuntimeError):
                        pass
                    raise TimeoutError(f'Desktop not ready after {timeout}s: {exc}') from exc
                if time.monotonic() >= next_log:
                    print(f'{self.logs.name} waiting for guest screenshot ({deadline - time.monotonic():.0f}s remaining)', flush=True)
                    next_log = time.monotonic() + 30
                time.sleep(2)

    def step(self, name, fn):
        started = time.monotonic()
        try:
            result = {'status': 'passed', 'result': fn()}
        except Exception as exc:
            result = {'status': 'failed', 'error': str(exc)[-1600:]}
            if isinstance(exc, subprocess.CalledProcessError):
                result['error'] += ': ' + str(exc.output)[-800:]
        result['seconds'] = round(time.monotonic() - started, 3)
        self.steps[name] = result
        save(self.logs / 'checks.json', self.steps)
        print(f'{self.logs.name} {name}: {result["status"]} ({result["seconds"]}s) {result.get("error", "")}', flush=True)
        return result['status'] == 'passed'

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for name in ('qemu.log', 'serial.log'):
            if (self.work / name).exists():
                shutil.copyfile(self.work / name, self.logs / name)
        disk = self.work / 'disk.qcow2'
        return {'qemu_exited': self.process is None or self.process.poll() is not None,
                'overlay_allocated_bytes': disk.stat().st_blocks * 512 if disk.exists() else 0}


class Fixture(BaseHTTPRequestHandler):
    def do_GET(self):
        token = self.path.removeprefix('/')
        if token not in self.server.tokens:
            self.send_error(404)
            return
        body = f'<html><head><title>{token}</title></head><body>{token}</body></html>'.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def snapshot(vm, marker):
    def hmp(cmd):
        message = vm.qmp('human-monitor-command', {'command-line': cmd})
        if message.strip():
            raise RuntimeError(message)
    vm.guest('import os; os.sync()')
    hmp('savevm clean')
    dirty = vm.guest('from pathlib import Path; import os,sys; p = Path(sys.argv[1]); '
                     'p.write_text("dirty"); os.sync(); print(p.read_text())', marker)
    if dirty != 'dirty':
        raise RuntimeError('Guest marker did not change before snapshot restore')
    hmp('loadvm clean')
    if not vm.qmp('query-status')['running']:
        vm.qmp('cont')
    vm.wait_desktop(120, 'after-restore.png')
    restored = json.loads(vm.guest('from pathlib import Path; import sys,time,json; '
                                  'print(json.dumps([Path(sys.argv[1]).read_text(),time.time()]))', marker))
    if restored[0] != vm.token:
        raise RuntimeError('Snapshot did not restore the original guest file')
    return {'file_restored': True, 'guest_clock_skew_s': round(time.time() - restored[1], 2)}


def build_sft_template(image, local, evidence, test_gui=False, prepare_guest=None):
    """Snapshot a ready guest before any task setup. Each episode boots an independent writable copy."""
    started = time.monotonic()
    logs = evidence / 'template'
    logs.mkdir()
    prefix = os.environ.get('CUA_QEMU_PREFIX')
    tools = Qemu(Path(prefix) if prefix else None)
    attempts = []
    vm = None
    try:
        for index in range(2):
            vm = VM(tools, local, logs, index)
            vm.start(image, 'tcg', 4, 8192, worker_address='127.0.0.1')
            try:
                vm.wait_desktop(600, 'clean.png')
                attempts.append('ready')
                break
            except TimeoutError as exc:
                attempts.append(str(exc))
                save(logs / 'startup-attempts.json', attempts)
                if index:
                    raise
                vm.stop()
                print('Clean template guest timed out before task setup, retrying on a fresh disk', flush=True)
        save(logs / 'startup-attempts.json', attempts)
        if prepare_guest is not None:
            prepare_guest(vm)
        vm.guest("from pathlib import Path; import os,sys; Path('/tmp/cua-sft-baseline').write_text(sys.argv[1]); os.sync()", vm.token)
        message = vm.qmp('human-monitor-command', {'command-line': 'savevm sft-clean'})
        if message.strip():
            raise RuntimeError(message)
        # The live disk intentionally differs from the saved point. Clones must lose this marker on restore.
        vm.guest("from pathlib import Path; import os; Path('/tmp/cua-sft-after-save').touch(); os.sync()")
        checked_seconds = 0
        if test_gui:
            from sft_gui_input import compare_input
            check_start = time.monotonic()
            compare_input(vm, logs)
            checked_seconds = time.monotonic() - check_start
    finally:
        if vm is not None:
            vm.stop()
    disk = local / 'clean.qcow2'
    (vm.work / 'disk.qcow2').rename(disk)
    disk.chmod(0o400)
    record = {'machine': [str(image.resolve()), 'tcg', 4, 8192], 'baseline': vm.token,
              'build_seconds': time.monotonic() - started - checked_seconds, 'test_seconds': checked_seconds,
              'allocated_bytes': disk.stat().st_blocks * 512}
    save(local / 'template.json', record)
    save(logs / 'template.json', record)
    return local


def check_sft_restore(vm, template):
    """Check rollback and correct the guest clock before the upstream task reset."""
    token = json.loads((template / 'template.json').read_text())['baseline']
    vm.guest("from pathlib import Path; import sys,subprocess; "
             "assert Path('/tmp/cua-sft-baseline').read_text()==sys.argv[1]; "
             "assert not Path('/tmp/cua-sft-after-save').exists(); "
             "cmd=['date','-u','-s','@'+sys.argv[2]]; "
             "r=subprocess.run(['sudo','-n',*cmd],capture_output=True); "
             "r = r if r.returncode==0 else subprocess.run(['sudo','-S','-p','',*cmd],input=b'password\\n',capture_output=True); "
             "assert r.returncode==0,r.stderr", token, str(time.time()))
    skew = abs(time.time() - float(vm.guest('import time; print(time.time())')))
    if skew > 30:
        raise RuntimeError(f'Restored guest clock differs by {skew:.1f}s')
    return {'baseline': token, 'clock_skew_seconds': skew}


def exercise(vm, args, port, marker):
    if not vm.step('qmp_vnc', vm.machine_ready) or args.host_only:
        return
    def first_screenshot():
        result = vm.wait_desktop(args.timeout, 'desktop.png')
        result['seconds_since_qemu_start'] = round(time.monotonic() - vm.started, 3)
        return result
    if not vm.step('desktop_boot', first_screenshot):
        return
    url = f'http://10.0.2.2:{port}/{vm.token}'
    program = GUEST.read_text()
    apps = ('science',) if args.suite == 'scienceboard' else ('browser', 'office')
    for action in ('desktop', 'network', *apps):
        vm.step(action, lambda a=action: json.loads(vm.guest(program, a, vm.token, url)))
    def observation_rounds():
        seconds = []
        for _ in range(args.rounds):
            start = time.monotonic()
            vm.screenshot('observation.png')
            reply = vm.guest('import pyautogui as p; print(p.size().width)')
            if int(reply) <= 0:
                raise ValueError('Guest display dimensions unavailable')
            seconds.append(round(time.monotonic() - start, 3))
        return {'screenshot_and_control_seconds': seconds}
    vm.step('observation_rounds', observation_rounds)
    vm.step('write_marker', lambda: vm.guest('import sys; from pathlib import Path; '
                                             'Path(sys.argv[1]).write_text(sys.argv[2])', marker, vm.token))
    if not args.skip_snapshot and vm.steps['write_marker']['status'] == 'passed':
        vm.step('snapshot_reset', lambda: snapshot(vm, marker))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-only', action='store_true', help='Start QEMU with a small empty disk; test QMP/VNC')
    parser.add_argument('--image', type=Path, default=ROOT / 'data/osworld-vm/Ubuntu.qcow2')
    parser.add_argument('--suite', choices=['auto', 'osworld', 'scienceboard'], default='auto',
                        help='Application checks; auto uses the image filename')
    parser.add_argument('--qemu-prefix', type=Path, help='Root of an unpacked Ubuntu QEMU package tree')
    parser.add_argument('--accelerator', choices=['auto', 'tcg', 'kvm'], default='auto')
    parser.add_argument('--vms', type=int, default=1)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory-mib', type=int, default=8192)
    parser.add_argument('--timeout', type=int, default=600, help='Guest boot timeout per VM in seconds')
    parser.add_argument('--rounds', type=int, default=3, help='Number of timed screenshot and control round trips')
    parser.add_argument('--skip-snapshot', action='store_true')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/ais-vm-checks')
    parser.add_argument('--sft-template', type=Path, help='Prepare a private full-state SFT template in this directory')
    parser.add_argument('--test-gui', action='store_true', help='Compare guest input backends after saving the clean state')
    args = parser.parse_args()
    if args.sft_template:
        build_sft_template(args.image, args.sft_template, args.output_root, args.test_gui)
        return
    if args.suite == 'auto':
        args.suite = 'scienceboard' if 'scienceboard' in args.image.name.lower() else 'osworld'
    if min(args.vms, args.cpus, args.memory_mib, args.timeout, args.rounds) < 1:
        parser.error('VM count, CPU count, memory, timeout and rounds must be positive')
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    output = args.output_root.resolve() / run
    output.mkdir(parents=True)
    report = {'status': 'running', 'host': socket.gethostname(), 'host_only': args.host_only,
              'suite': None if args.host_only else args.suite,
              'requested_vms': args.vms, 'accelerator_requested': args.accelerator,
              'requested_vcpus_per_vm': args.cpus, 'requested_mib_per_vm': args.memory_mib,
              'snapshot_requested': not (args.skip_snapshot or args.host_only), 'vms': []}
    print('Report:', output / 'report.json', flush=True)
    save(output / 'report.json', report)
    vms = []

    def interrupted(sig, frame):
        for vm in vms:
            if vm.process is not None and vm.process.poll() is None:
                vm.process.terminate()
        raise KeyboardInterrupt(f'signal {sig}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        try:
            require_kvm()
            kvm = True
        except RuntimeError as exc:
            kvm = False
            report['kvm_error'] = str(exc)
        accel = ('kvm' if kvm else 'tcg') if args.accelerator == 'auto' else args.accelerator
        report.update(kvm_usable=kvm, accelerator=accel, cpu_affinity=len(os.sched_getaffinity(0)),
                      scratch_free_bytes=shutil.disk_usage('/tmp').free, shm_bytes=shutil.disk_usage('/dev/shm').total)
        for key in ('cpu.max', 'memory.max'):
            path = Path('/sys/fs/cgroup') / key
            report[key] = path.read_text().strip() if path.is_file() else None
        if accel == 'kvm' and not kvm:
            raise RuntimeError(report['kvm_error'])
        if shutil.which('nvidia-smi'):
            gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=15)
            report['nvidia_devices'] = gpu.stdout.strip().splitlines()
        tools = Qemu(args.qemu_prefix.resolve() if args.qemu_prefix else None)
        report['qemu'] = tools.run('qemu-system-x86_64', '--version').splitlines()[0]
        cpus, ram = (1, 256) if args.host_only else (args.cpus, args.memory_mib)
        report.update(vcpus_per_vm=cpus, memory_mib_per_vm=ram)
        limit = report['memory.max']
        if limit and limit.isdigit() and args.vms * ram * 2**20 >= int(limit):
            raise ValueError('Guest RAM allocation reaches the container memory limit')
        with tempfile.TemporaryDirectory(prefix='ais-vm-', dir='/tmp') as tmp:
            scratch = Path(tmp)
            base = args.image.resolve()
            if args.host_only:
                base = scratch / 'firmware-check.qcow2'
                tools.run('qemu-img', 'create', '-f', 'qcow2', base, '8M')
            elif not base.is_file():
                raise FileNotFoundError(base)
            else:
                from PIL import Image  # check the host screenshot decoder before booting
            info = json.loads(tools.run('qemu-img', 'info', '--output=json', base))
            if info['format'] != 'qcow2':
                raise ValueError('Expected a qcow2 base image')
            report.update(image=None if args.host_only else str(base), image_bytes=base.stat().st_size)
            save(output / 'report.json', report)
            print(f'QEMU {accel}, {args.vms} VM(s), {cpus} vCPU and {ram} MiB per VM', flush=True)
            with ThreadingHTTPServer(('127.0.0.1', 0), Fixture) as fixture:
                fixture.tokens = set()
                threading.Thread(target=fixture.serve_forever, daemon=True).start()
                try:
                    for i in range(args.vms):
                        vm = VM(tools, scratch, output, i)
                        vms.append(vm)
                        vm.start(base, accel, cpus, ram)
                    fixture.tokens = {vm.token for vm in vms}
                    mark = '/tmp/ais-vm-marker-' + run
                    with ThreadPoolExecutor(max_workers=len(vms)) as pool:
                        list(pool.map(functools.partial(exercise, args=args, port=fixture.server_port,
                                                       marker=mark), vms))
                    for vm in vms:
                        if 'write_marker' in vm.steps:
                            def check(v=vm):
                                value = v.guest('from pathlib import Path; import sys; '
                                                'print(Path(sys.argv[1]).read_text())', mark)
                                if value != v.token:
                                    raise RuntimeError('Guest marker belongs to another VM or is stale')
                                return {'unique_marker': True}
                            vm.step('overlay_marker', check)
                finally:
                    for vm in vms:
                        vm.step('cleanup', vm.stop)
                    fixture.shutdown()
        report['status'] = 'passed' if all(s['status'] == 'passed' for v in vms for s in v.steps.values()) else 'failed'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            report['error'] += ' ' + str(exc.output)[-1200:]
    finally:
        report['vms'] = [v.steps for v in vms]
        save(output / 'report.json', report)
    print(f'{report["status"]}: {output / "report.json"} {report.get("error", "")}', flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
