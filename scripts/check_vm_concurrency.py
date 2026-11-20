#!/usr/bin/env python3
"""Validate concurrent desktop boot and control with an explicit QEMU accelerator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import time
import urllib.request

from PIL import Image, ImageStat
from qemu_runtime import LocalVM, ROOT, require_kvm


def exercise(vm, rounds):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    latencies = []
    for _ in range(rounds):
        started = time.monotonic()
        with opener.open(f'http://{vm.address}:5000/screenshot', timeout=30) as response:
            picture = Image.open(io.BytesIO(response.read())).convert('RGB')
        if max(ImageStat.Stat(picture.resize((64, 36))).stddev) < 5:
            raise RuntimeError(f'Blank screenshot: {vm.id}')
        payload = json.dumps({'command': ['python', '-c',
                              'import pyautogui; print(pyautogui.size())']}).encode()
        req = urllib.request.Request(f'http://{vm.address}:5000/execute', data=payload,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=30) as response:
            result = json.load(response)
        if result.get('returncode') != 0 or 'width=' not in result.get('output', ''):
            raise RuntimeError(f'Guest control failed: {result}')
        latencies.append(time.monotonic() - started)
    return {'id': vm.id, 'rounds': rounds, 'screen_size': list(picture.size),
            'screenshot_and_control_seconds': latencies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accelerator', choices=['kvm', 'tcg'], default='kvm')
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory-mb', type=int, default=8192)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--base-image', type=Path)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/paths.json').read_text())
    host = socket.gethostname()
    if host not in cfg['workers']:
        parser.error('Only worker-0/1 may run VMs')
    if args.stages != sorted(set(args.stages)) or not all(1 <= n <= 16 for n in args.stages):
        parser.error('Use distinct increasing stages from 1 through 16')
    if not 1 <= args.rounds <= 100 or not 30 <= args.timeout <= 600:
        parser.error('rounds must be 1..100; timeout must be 30..600 seconds')
    report_path = ROOT / 'reports' / f'{args.accelerator}-concurrency-{host.removeprefix("research-a-")}.json'
    report = {'hostname': host, 'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'running', 'accelerator': args.accelerator, 'stages': [],
              'requested_concurrencies': args.stages, 'vcpus_per_vm': args.cpus,
              'memory_mb_per_vm': args.memory_mb,
              'scope': 'VM startup and guest API acceptance, not sustained rollout throughput or benchmark score'}
    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    lock_path = ROOT / 'runs' / f'vm-concurrency-{host}.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.accelerator == 'kvm':
                require_kvm()
            for count in args.stages:
                stage = {'concurrency': count, 'passed': False}
                report['stages'].append(stage)
                save()
                vms = []
                started = time.monotonic()
                try:
                    for index in range(count):
                        address = f'127.0.2.{20 + index}'
                        for port in [5000, 9222, 8080, 8000, 5900]:
                            with socket.socket() as check:
                                check.bind((address, port))
                        vms.append(LocalVM(address=address, cpus=args.cpus, memory_mb=args.memory_mb,
                                           accelerator=args.accelerator, base_image=args.base_image))
                    def boot(vm):
                        state = vm.start()
                        state['boot_seconds'] = vm.wait_ready(args.timeout)
                        return state
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        stage['vms'] = list(pool.map(boot, vms))
                        stage['all_ready_seconds'] = time.monotonic() - started
                        stage['guest_checks'] = list(pool.map(lambda vm: exercise(vm, args.rounds), vms))
                    times = [t for item in stage['guest_checks'] for t in item['screenshot_and_control_seconds']]
                    stage.update(passed=True, api_median_seconds=statistics.median(times),
                                 api_max_seconds=max(times))
                finally:
                    for vm in vms:
                        vm.stop()
                    stage['cleanup_passed'] = all(vm.process is None or vm.process.poll() is not None for vm in vms)
                    stage['elapsed_seconds'] = time.monotonic() - started
                    save()
                print(json.dumps({'stage': count, 'passed': stage['passed'],
                                  'all_ready_seconds': stage['all_ready_seconds']}), flush=True)
            report['status'] = 'passed'
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status='blocked' if not report['stages'] else 'failed', error=str(exc))
        finally:
            save()
    print(json.dumps({'status': report['status'], 'report': str(report_path), 'error': report.get('error')}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
