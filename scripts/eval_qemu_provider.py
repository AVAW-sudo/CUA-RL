"""Use isolated local QEMU disks behind the upstream DesktopEnv provider API."""
import fcntl
import io
import json
import logging
import os
from pathlib import Path
import signal
import shutil
import socket
import time
import tempfile
import uuid

from PIL import Image
from qemu_runtime import ROOT, LocalVM

OPTIONS = {}


class QemuProvider:
    def __init__(self):
        self.vm = None
        self.lease = None

    def start_emulator(self, path_to_vm, headless, os_type):
        if self.vm is not None:
            return
        lock_dir = ROOT / 'runs/vm-address-locks' / socket.gethostname()
        lock_dir.mkdir(parents=True, exist_ok=True)
        for index in range(20, 240):
            lease = (lock_dir / f'eval-{index}.lock').open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                address = f'127.0.3.{index}'
                for port in [5000, 9222, 8080, 8000, 5900]:
                    with socket.socket() as check:
                        check.bind((address, port))
                self.lease = lease
                break
            except OSError:
                lease.close()
        else:
            raise RuntimeError('No free local evaluation address')
        try:
            self.vm = LocalVM(address=address, cpus=OPTIONS['cpus'], memory_mb=OPTIONS['memory_mb'],
                              accelerator=OPTIONS['accelerator'], base_image=path_to_vm)
            state = self.vm.start()
            state['boot_seconds'] = self.vm.wait_ready(OPTIONS['boot_timeout'])
            (self.vm.directory / 'boot.json').write_text(json.dumps(state, indent=2) + '\n')
        except BaseException:
            self.stop_emulator(path_to_vm)
            raise

    def get_ip_address(self, path_to_vm):
        return f'{self.vm.address}:5000:9222:5900:8080:8000'

    def stop_emulator(self, path_to_vm):
        try:
            if self.vm is not None:
                self.vm.stop()
        finally:
            self.vm = None
            if self.lease:
                self.lease.close()
                self.lease = None

    def revert_to_snapshot(self, path_to_vm, snapshot_name):
        self.stop_emulator(path_to_vm)
        return path_to_vm  # start_emulator creates a new overlay, leaving the base immutable.

    def save_state(self, path_to_vm, snapshot_name):
        raise NotImplementedError('Evaluation uses fresh base-image overlays')


class AisEvalProvider(QemuProvider):
    """Use ephemeral guest ports and a node-local overlay on AIS."""
    def start_emulator(self, path_to_vm, headless, os_type):
        if self.vm is not None:
            return
        from ais_vm_check import Qemu, VM, check_sft_restore, save
        out = Path(OPTIONS['audit_directory']).parent / 'vms' / uuid.uuid4().hex
        out.mkdir(parents=True)
        scratch = tempfile.TemporaryDirectory(prefix='sft-vm-')
        self.lease = scratch
        try:
            prefix = os.environ.get('CUA_QEMU_PREFIX')
            path = os.environ.get('CUA_SFT_TEMPLATE')
            template = Path(path) if path else None
            self.vm = VM(Qemu(Path(prefix) if prefix else None), Path(scratch.name), out, 0)
            started = time.monotonic()
            self.vm.start(path_to_vm, OPTIONS['accelerator'], OPTIONS['cpus'], OPTIONS['memory_mb'],
                          worker_address='127.0.0.1', template=template)
            owner = {'id': self.vm.token, 'pid': self.vm.process.pid, 'hostname': socket.gethostname(),
                     'drive': f'file={self.vm.work}/disk.qcow2,if=virtio,format=qcow2'}
            tmp = out / 'owner.tmp'
            tmp.write_text(json.dumps(owner))
            tmp.replace(out / 'owner.json')
            self.vm.wait_desktop(OPTIONS['boot_timeout'], 'ready.png')
            if template:
                save(out / 'restore.json', check_sft_restore(self.vm, template))
            save(out / 'boot.json', {'mode': 'snapshot' if template else 'cold',
                                     'seconds_to_ready': time.monotonic() - started})
        except BaseException:
            self.stop_emulator(path_to_vm)
            raise

    def get_ip_address(self, path_to_vm):
        ports = self.vm.ports
        return '127.0.0.1:' + ':'.join(str(ports[p]) for p in (5000, 9222, 8006, 8080, 8000))

    def stop_emulator(self, path_to_vm):
        try:
            if self.vm is not None:
                self.vm.stop()
        finally:
            self.vm = None
            if self.lease:
                self.lease.cleanup()
                self.lease = None


class QemuManager:
    def get_vm_path(self, **kwargs):
        return OPTIONS['base_image']


def factory(provider_name, region, use_proxy=False):
    if provider_name != 'docker':
        raise ValueError('This adapter only replaces the explicitly selected docker provider API')
    return QemuManager(), AisEvalProvider() if OPTIONS.get('ais') else QemuProvider()


def install(options):
    """Preserve upstream agent/actions/verifier, checking setup errors separately."""
    OPTIONS.clear()
    OPTIONS.update(options)
    import desktop_env.providers
    import desktop_env.desktop_env
    desktop_env.providers.create_vm_manager_and_provider = factory
    desktop_env.desktop_env.create_vm_manager_and_provider = factory
    Base = desktop_env.desktop_env.DesktopEnv

    class CheckedDesktopEnv(Base):
        def __init__(self, *args, **kwargs):
            if OPTIONS.get('audit_directory'):
                kwargs['cache_dir'] = str(Path(OPTIONS['audit_directory']).parent / 'verifier-cache')
            if OPTIONS.get('proxy_policy') == 'direct':
                kwargs['enable_proxy'] = False
            super().__init__(*args, **kwargs)

        def _start_emulator(self):
            super()._start_emulator()
            self._guest_input = None
            if OPTIONS.get('ais') and os.getenv('CUA_SFT_RESIDENT_GUI') == '1':
                from sft_gui_input import GuestInput
                self._guest_input = GuestInput(self.provider.vm, self.controller)
            try:
                expected_size = (self.screen_width, self.screen_height)
                self.setup_controller._screen_size_setup(*expected_size)
                picture = Image.open(io.BytesIO(self.controller.get_screenshot()))
                if picture.size != expected_size:
                    raise RuntimeError(f'Desktop size {picture.size} differs from protocol {expected_size}')
                original_setup = self.setup_controller.setup
                original_launch = self.setup_controller._launch_setup
                if OPTIONS.get('benchmark') == 'scienceboard':
                    from scienceboard_compat import install_setup
                    install_setup(self.setup_controller)
                self._last_setup_ok = None
                self._checking_app_ready = False
                def launch(command, shell=False):
                    text = command if isinstance(command, str) else ' '.join(command)
                    app = next((name for name in ['kalgebra', 'celestia-qt6', 'chimerax', 'FLASK_PORT=8000'] if name in text), None)
                    if app == 'FLASK_PORT=8000' and OPTIONS.get('benchmark') == 'scienceboard':
                        from scienceboard_compat import prepare_grass
                        audit = prepare_grass(self.setup_controller)
                        (Path(OPTIONS['audit_directory']) / (self.provider.vm.token + '-grass.json') if OPTIONS.get('ais')
                         else self.provider.vm.directory / 'scienceboard-guest-patch.json').write_text(json.dumps(audit, indent=2) + '\n')
                    value = original_launch(command, shell=shell)
                    if app:
                        # Inspect the guest socket table instead of calling app
                        # data APIs while their GUI state is still initializing.
                        # ChimeraX also binds REST to guest loopback, so checking
                        # the externally forwarded port alone is insufficient.
                        import requests
                        deadline = time.monotonic() + 180
                        self._checking_app_ready = True
                        probe = ('from pathlib import Path; '
                                 'rows=[line.split() for name in ("tcp", "tcp6") '
                                 'for line in Path("/proc/net/"+name).read_text().splitlines()[1:]]; '
                                 'print("ready" if any(r[3]=="0A" and int(r[1].split(":")[-1],16)==8000 '
                                 'for r in rows) else "waiting")')
                        try:
                            while time.monotonic() < deadline:
                                try:
                                    response = requests.post(f'http://{self.vm_ip}:{self.server_port}/execute',
                                                             json={'command': ['python3', '-c', probe]}, timeout=10)
                                    body = response.json() if response.status_code == 200 else {}
                                    if body.get('returncode') == 0 and body.get('output', '').strip() == 'ready':
                                        if app == 'FLASK_PORT=8000' and OPTIONS.get('benchmark') == 'scienceboard':
                                            # GRASS binds before its GUI frame exists. Its dump
                                            # safely returns HTTP 500 until initialization ends.
                                            ready = requests.get(f'http://{self.vm_ip}:{self.app_port}/dump', timeout=10)
                                            if ready.status_code != 200:
                                                time.sleep(2)
                                                continue
                                        time.sleep(10)  # Let GUI state finish initializing before setup commands.
                                        logging.getLogger(__name__).info('Application listener ready: %s', app)
                                        break
                                except (requests.RequestException, ValueError):
                                    pass
                                time.sleep(2)
                            else:
                                raise RuntimeError(f'Application API did not start within 180 seconds: {app}')
                        finally:
                            self._checking_app_ready = False
                    return value
                self.setup_controller._launch_setup = launch
                def setup(*args, **kwargs):
                    import requests
                    original_request = requests.sessions.Session.request
                    failures = []
                    def tracked(session, method, url, *a, **kw):
                        try:
                            response = original_request(session, method, url, *a, **kw)
                        except requests.RequestException as exc:
                            if not self._checking_app_ready:
                                failures.append(type(exc).__name__ + ': ' + str(url))
                            raise
                        if not self._checking_app_ready and response.status_code >= 400:
                            failures.append(f'HTTP {response.status_code}: {url}')
                        if not self._checking_app_ready and str(url).endswith('/chimerax/run') and response.status_code == 200:
                            try:
                                body = response.json()
                                if body.get('error') or body.get('log messages', {}).get('error'):
                                    failures.append('ChimeraX setup command returned an application error')
                            except (ValueError, AttributeError):
                                pass
                        return response
                    requests.sessions.Session.request = tracked
                    self._last_setup_ok = False
                    try:
                        value = original_setup(*args, **kwargs)
                    finally:
                        requests.sessions.Session.request = original_request
                    if failures:
                        raise RuntimeError('Setup HTTP failure (including swallowed upstream errors): ' + repr(failures))
                    self._last_setup_ok = value
                    return value
                self.setup_controller.setup = setup
                if OPTIONS.get('recording_mode') == 'steps-only':
                    # Upstream still saves every step PNG and trajectory JSONL.
                    # Continuous guest-side H.264 encoding is optional diagnostic work.
                    self.controller.start_recording = lambda: None
                    self.controller.end_recording = lambda *args, **kwargs: None
            except BaseException:
                self.provider.stop_emulator(self.path_to_vm)
                raise

        def reset(self, task_config=None, seed=None, options=None):
            if task_config and OPTIONS.get('ais') and OPTIONS.get('benchmark') == 'osworld' and os.getenv('CUA_SFT_SETUP_INPUTS'):
                # Copy only published, digest-verified initial inputs. Preserve this episode's own verifier outputs separately.
                inputs = Path(os.environ['CUA_SFT_SETUP_INPUTS']) / task_config['id']
                if inputs.is_dir():
                    shutil.copytree(inputs, Path(self.cache_dir_base) / task_config['id'], dirs_exist_ok=True)
            if task_config is not None and OPTIONS.get('benchmark') == 'scienceboard':
                from scienceboard_compat import normalize_task
                original = task_config
                task_config, changes = normalize_task(task_config, OPTIONS.get('scienceboard_setup', 'released'))
                audit = {'task_id': original['id'], 'changes': changes,
                         'original_task': original, 'effective_task': task_config}
                directory = Path(OPTIONS['audit_directory'])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / (original['id'] + '.json')).write_text(json.dumps(audit, indent=2) + '\n')
            obs = super().reset(task_config=task_config, seed=seed, options=options)
            if task_config is not None and self._last_setup_ok is not True:
                raise RuntimeError('Task setup failed after upstream retries; not a model failure')
            return obs

        def step(self, action, *args, **kwargs):
            if self._guest_input is None:
                return super().step(action, *args, **kwargs)
            self.controller.execute_python_command = self._guest_input.execute
            try:
                return super().step(action, *args, **kwargs)
            finally:
                self.controller.execute_python_command = self._guest_input.original

        def evaluate(self):
            # Some upstream verifiers swallow HTTP failures and return zero. Keep
            # these episodes unscored for diagnosis; a failed GUI task still gets
            # the unmodified numeric verifier result when the verifier is usable.
            import requests
            original = requests.sessions.Session.request
            failures = []
            def tracked(session, method, url, *args, **kwargs):
                try:
                    response = original(session, method, url, *args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as exc:
                    failures.append(type(exc).__name__ + ': ' + str(url))
                    raise
                if response.status_code >= 500:
                    failures.append(f'HTTP {response.status_code}: {url}')
                return response
            requests.sessions.Session.request = tracked
            self._last_setup_ok = True
            try:
                score = super().evaluate()
            finally:
                requests.sessions.Session.request = original
            if self._last_setup_ok is not True:
                raise RuntimeError('Verifier postconfig failed; episode needs review')
            if failures and float(score) < 1:
                raise RuntimeError('Verifier transport failure; episode needs review: ' + repr(failures))
            return score

    return CheckedDesktopEnv


def cleanup_run(run_id, output=None):
    """Clean only this run's exact QEMU identities after worker exit/interruption."""
    owned = []
    def alive(state):
        proc = Path(f'/proc/{state["pid"]}/cmdline')
        try:
            identity = state.get('drive') or 'cua-rl-' + state['id']
            return identity.encode() in proc.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            return False
    if output is not None:
        for path in (Path(output) / 'vms').glob('*/owner.json'):
            state = json.loads(path.read_text())
            if state['hostname'] == socket.gethostname() and alive(state):
                os.kill(state['pid'], signal.SIGTERM)
                owned.append(state)
    legacy_states = () if OPTIONS.get('ais') else (ROOT / 'runs/vms').glob('*/state.json')
    for path in legacy_states:
        state = json.loads(path.read_text())
        if state.get('run_id') != run_id or state['hostname'] != socket.gethostname():
            continue
        if alive(state):
            os.kill(state['pid'], signal.SIGTERM)
            owned.append(state)
    deadline = time.monotonic() + 15
    while any(alive(s) for s in owned) and time.monotonic() < deadline:
        time.sleep(.2)
    for state in owned:
        if alive(state):
            os.kill(state['pid'], signal.SIGKILL)
    return {'terminated_orphans': [s['id'] for s in owned]}
