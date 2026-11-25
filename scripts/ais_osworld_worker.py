"""OSWorld task worker using disposable AIS guests and the upstream task and verifier."""
import asyncio
import atexit
import io
import os
from pathlib import Path
import tempfile
import threading
import uuid

import desktop_env.desktop_env
import desktop_env.providers
from desktop_env.controllers.python import PythonController
from PIL import Image
from agentrl.worker.environment import EnvironmentController
from agentrl.worker.trace_store import TraceStore
from src.environment import FrameworkVMManager, FrameworkVMProvider
import src.task
from src.session import OSWorldSession
from src.task import OSWorldTask
from ais_vm_check import Qemu, VM, check_sft_restore
from qemu_runtime import require_kvm
from ais_verifier import install_scoring_observer, prepare_guest_verifier, run_bash_once, scoring_requests


class AisTraceStore(TraceStore):
    def new_trace(self, *args, **kwargs):
        writer = super().new_trace(*args, **kwargs)
        archive = writer.finalize_callback
        fallback = writer._finalizer
        lock = threading.Lock()
        result = None

        def archive_once(path):
            nonlocal result
            with lock:
                if fallback.alive:
                    result = archive(path)
                    # Upstream close also invokes the GC callback after an explicit save.
                    # Keep the fallback armed until the archive and its indices succeed.
                    fallback.detach()
                return result

        writer.finalize_callback = archive_once
        return writer


class AisScreenshotController(PythonController):
    def run_bash_script(self, script, timeout=30, working_dir=None):
        return run_bash_once(self.http_server, script, timeout, working_dir)

    @staticmethod
    def _is_valid_image_response(content_type, data):
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ('PNG', 'JPEG'):
                    return False
                image.load()
                return True
        except (OSError, TypeError, ValueError):
            return False

    def get_screenshot(self):
        # The upstream controller retries this read; a failed read never replays the GUI action.
        data = super().get_screenshot()
        if not data:
            raise RuntimeError('OSWorld returned no decodable screenshot after its read retries')
        return data


class AisEnvironmentController(EnvironmentController):
    def __init__(self, delegation, image, audit, cpus, ram):
        super().__init__(delegation)
        self.image = Path(image).resolve(strict=True)
        self.audit = Path(audit)
        self.audit.mkdir(parents=True, exist_ok=True)
        self.cpus, self.ram = cpus, ram
        prefix = os.getenv('CUA_QEMU_PREFIX')
        self.tools = Qemu(Path(prefix) if prefix else None)
        try:
            require_kvm()
            self.accel = 'kvm'
        except RuntimeError:
            self.accel = 'tcg'
        self.leases = {}
        atexit.register(self.close)

    def close(self):
        for lease in list(self.leases):
            self.release(lease)

    def release(self, lease):
        if lease in self.leases:
            vm, scratch = self.leases.pop(lease)
            try:
                vm.stop()
            finally:
                scratch.cleanup()

    async def start_session(self, subtypes, **kwargs):
        subtype = subtypes if isinstance(subtypes, str) else subtypes[0]
        if subtype not in ('Ubuntu', 'Ubuntu-modified'):
            raise ValueError(f'Unsupported OSWorld image type: {subtype}')
        lease = uuid.uuid4().hex
        audit = self.audit / lease
        audit.mkdir()
        scratch = tempfile.TemporaryDirectory(prefix='cua-vm-')
        vm = VM(self.tools, Path(scratch.name), audit, 0)
        self.leases[lease] = vm, scratch
        address = '127.0.0.1'

        def boot():
            path = os.getenv('CUA_RL_VM_TEMPLATE')
            template = Path(path) if path else None
            vm.start(self.image, self.accel, self.cpus, self.ram * 1024, worker_address=address, template=template)
            vm.wait_desktop(600, 'ready.png')
            if template:
                check_sft_restore(vm, template)
            prepare_guest_verifier(vm)

        task = asyncio.create_task(asyncio.to_thread(boot))
        try:
            await asyncio.shield(task)
        except BaseException:
            try:
                await task
            finally:
                await asyncio.to_thread(self.release, lease)
            raise
        return lease, {subtype: lease}, {subtype: address}

    async def renew_session(self, session_id):
        pass

    async def end_session(self, session_id):
        await asyncio.to_thread(self.release, session_id)


class AisVMProvider(FrameworkVMProvider):
    def get_ip_address(self, *args, **kwargs):
        vm, _ = self.controller.leases[self.session_id]
        self.server_port, self.chromium_port, self.vnc_port, self.vlc_port, self.app_port = (
            vm.ports[p] for p in (5000, 9222, 8006, 8080, 8000))
        return super().get_ip_address(*args, **kwargs)


class AisOSWorldSession(OSWorldSession):
    def initialize_env(self, version, config):
        env = super().initialize_env(version, config)
        evaluate = env.evaluate

        def checked_evaluate():
            with scoring_requests(env.controller.http_server):
                return evaluate()

        env.evaluate = checked_evaluate
        return env


class AisOSWorldTask(OSWorldTask):
    def __init__(self, ais_image, ais_audit, **kwargs):
        super().__init__(env_driver='manual', env_options={'urls': {}}, **kwargs)
        if self.trace_store is not None:
            self.trace_store = AisTraceStore(self.trace_store.root)
        self.env_controller = AisEnvironmentController(self.env_delegation, ais_image, ais_audit,
                                                      kwargs['vm_cpu'], kwargs['vm_ram'])
        def factory(*args, **kwargs):
            return FrameworkVMManager(), AisVMProvider(self.env_controller)
        desktop_env.providers.create_vm_manager_and_provider = factory
        desktop_env.desktop_env.create_vm_manager_and_provider = factory
        desktop_env.desktop_env.PythonController = AisScreenshotController
        src.task.OSWorldSession = AisOSWorldSession
        install_scoring_observer()
