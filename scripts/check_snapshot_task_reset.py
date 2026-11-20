"""CPU-only snapshot reset checks using original task setup and verifiers.

Scripted diagnostic solutions are never model rollouts or benchmark results.
Each task runs once from a cold guest and three times after full-state restore.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

from benchmark_environment_runtime import ProbeVM, guest
from qemu_runtime import ROOT
from run_benchmark import now, write_json


def monitor(vm, command):
    output = vm.monitor(command)
    if any(word in output.lower() for word in ['error', 'failed', 'not supported']):
        raise RuntimeError(output[-1000:])


def run_task(benchmark, path, action, report, save, output):
    import eval_qemu_provider as adapter
    import desktop_env.providers
    import desktop_env.desktop_env

    base = ROOT / ('data/osworld-vm/Ubuntu.qcow2' if benchmark == 'osworld'
                   else 'data/scienceboard-vm/Ubuntu2_scienceboard.qcow2')
    screen = (1920, 1080) if benchmark == 'osworld' else (1280, 800)
    row = {'benchmark': benchmark, 'task_path': str(path), 'task_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'action': action, 'status': 'running', 'iterations': [], 'started_at_utc': now()}
    report['tasks'].append(row)
    save()
    Environment = adapter.install({'accelerator': 'tcg', 'cpus': 4, 'memory_mb': 8192,
        'boot_timeout': 600, 'base_image': str(base), 'recording_mode': 'steps-only',
        'proxy_policy': 'direct', 'benchmark': benchmark,
        'audit_directory': str(output / benchmark / 'task-compatibility')})

    class SnapshotProvider(adapter.QemuProvider):
        def start_emulator(self, path_to_vm, headless, os_type):
            if self.vm is not None:
                return
            if report.get('production_runtime'):
                from snapshot_vm import SnapshotVM
                self.vm = SnapshotVM('127.0.4.91', base_image=path_to_vm)
            else:
                self.vm = ProbeVM(Path(path_to_vm), Path(f'/tmp/cua-rl-task-reset-{os.getuid()}'), address='127.0.4.91')
            row['vm'] = self.vm.start()
            save()
            self.vm.wait_ready(600)

        def revert_to_snapshot(self, path_to_vm, snapshot_name):
            started = time.monotonic()
            if report.get('production_runtime'):
                self.vm.restore_clean()
            else:
                monitor(self.vm, 'loadvm cua_task_probe_clean')
                monitor(self.vm, 'cont')
                self.vm.wait_ready(120)
            self.last_restore_seconds = time.monotonic() - started
            return path_to_vm

    def factory(provider_name, region, use_proxy=False):
        return adapter.QemuManager(), SnapshotProvider()

    desktop_env.providers.create_vm_manager_and_provider = factory
    desktop_env.desktop_env.create_vm_manager_and_provider = factory
    environment = None
    try:
        started = time.monotonic()
        environment = Environment(provider_name='docker', path_to_vm=str(base),
            cache_dir=str(output / benchmark / 'cache'), screen_size=screen,
            headless=True, require_a11y_tree=False, action_space='pyautogui', enable_proxy=False)
        row['cold_environment_seconds'] = time.monotonic() - started
        if report.get('production_runtime'):
            environment.provider.vm.save_clean()
        else:
            monitor(environment.provider.vm, 'savevm cua_task_probe_clean')
        task = json.loads(path.read_text())
        for iteration in range(4):
            item = {'iteration': iteration, 'arm': 'cold' if iteration == 0 else 'snapshot', 'status': 'running'}
            row['iterations'].append(item)
            save()
            started = time.monotonic()
            observation = environment.reset(task_config=task)
            item['reset_and_setup_seconds'] = time.monotonic() - started
            item['restore_seconds'] = getattr(environment.provider, 'last_restore_seconds', None)
            if benchmark == 'scienceboard':
                # Retain the original ScienceBoard runner's post-setup settling
                # period before probing app APIs; socket readiness alone is insufficient.
                time.sleep(60)
                item['additional_settle_seconds'] = 60
                observation = environment._get_obs()
            folder = output / benchmark / str(iteration)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'initial.png').write_bytes(observation['screenshot'])
            item['dirty_marker_absent'] = guest(environment.provider.vm,
                "from pathlib import Path; print(not Path('/tmp/cua-task-reset-dirty').exists())") == 'True'
            item['initial_score'] = float(environment.evaluate())
            executed = environment.controller.execute_python_command
            def checked_execute(command):
                result = executed(command)
                item['execution'] = result
                if not result or result.get('returncode') != 0:
                    raise RuntimeError('Scripted diagnostic action failed: ' + str(result))
                return result
            environment.controller.execute_python_command = checked_execute
            started = time.monotonic()
            try:
                observation, _, _, _ = environment.step(action, pause=3)
            finally:
                environment.controller.execute_python_command = executed
            item['action_and_observation_seconds'] = time.monotonic() - started
            (folder / 'acted.png').write_bytes(observation['screenshot'])
            item['scripted_score'] = float(environment.evaluate())
            guest(environment.provider.vm, "from pathlib import Path; Path('/tmp/cua-task-reset-dirty').write_text('changed')")
            item['status'] = 'passed' if (item['dirty_marker_absent'] and item['initial_score'] == 0
                                         and item['scripted_score'] == 1) else 'failed'
            save()
            if iteration == 0 and item['status'] != 'passed':
                row['error'] = 'Cold diagnostic solution did not pass; cannot establish reset parity'
                break
        row['status'] = 'passed' if all(x['status'] == 'passed' for x in row['iterations']) else 'failed'
    except Exception as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if environment is not None and environment.provider.vm is not None:
            try:
                row['guest_diagnostics'] = guest(environment.provider.vm,
                    "import subprocess; print(subprocess.run(['pgrep','-a','kalgebra'],capture_output=True,text=True).stdout)")
            except Exception as diagnostic_error:
                row['diagnostic_error'] = str(diagnostic_error)
    finally:
        if environment is not None:
            vm = environment.provider.vm
            environment.close()
            row['cleanup_passed'] = vm is None or vm.process is None or vm.process.poll() is not None
        row['finished_at_utc'] = now()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-runtime', action='store_true')
    parser.add_argument('--benchmark', choices=['osworld', 'scienceboard'], required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'research-a-worker-0':
        parser.error('Only worker-0 CPU resources are used by this probe')
    if args.report.exists():
        parser.error('Choose a new report path; prior evidence is retained')
    lease = (ROOT / 'runs/snapshot-task-reset.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for port in [5000, 9222, 8080, 8000, 5900]:
        with socket.socket() as check:
            check.bind(('127.0.4.91', port))
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(signum)))
    source = ROOT / 'SCALE-CUA/osworld_eval'
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join([os.environ.get('NO_PROXY', ''), '127.0.4.91'])
    output = ROOT / 'runs/environment-task-reset' / args.report.stem
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'started_at_utc': now(), 'hostname': socket.gethostname(),
              'gpu_allocation': 0, 'tasks': [], 'production_runtime': args.production_runtime, 'scope': 'Selected-task engineering parity checks, not model scores or full benchmark equivalence',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'output': str(output)}
    def save():
        write_json(args.report, report)
    save()
    try:
        if args.benchmark == 'osworld':
            path = source / 'evaluation_examples/examples/os/3ce045a0-877b-42aa-8d2c-b4a863336ab8.json'
            action = "pyautogui.hotkey('ctrl','alt','t'); time.sleep(12); pyautogui.click(450,200); pyautogui.write('gsettings set org.gnome.desktop.interface text-scaling-factor 1.25', interval=0.05); pyautogui.press('enter'); time.sleep(3)"
        else:
            path = source / 'evaluation_examples/scienceboard/KAlgebra/KAlgebra_A-01.json'
            action = "pyautogui.click(544,594); pyautogui.write('p := conjugate(1+i)', interval=0.1); pyautogui.press('enter'); time.sleep(3)"
        run_task(args.benchmark, path, action, report, save, output)
        report['status'] = 'passed' if all(t['status'] == 'passed' for t in report['tasks']) else 'failed'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['finished_at_utc'] = now()
        save()
    print(json.dumps({'status': report['status'], 'report': str(args.report)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
