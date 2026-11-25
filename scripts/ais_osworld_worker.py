"""OSWorld task worker using disposable AIS guests and the upstream task and verifier."""
import asyncio
import atexit
from contextlib import suppress
from contextvars import ContextVar
import io
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import desktop_env.desktop_env
import desktop_env.providers
from desktop_env.controllers.python import PythonController
from PIL import Image
import requests
from agentrl.worker.environment import EnvironmentController
from agentrl.worker.typings import AgentOutput, AgentOutputStatus, SampleStatus, TaskSampleExecutionResult
from agentrl.worker.trace_store import TraceStore
from src.environment import FrameworkVMManager, FrameworkVMProvider
import src.task
from src.session import OSWorldSession
from src.task import OSWorldTask
from ais_vm_check import Qemu, VM, check_sft_restore
from qemu_runtime import require_kvm
from ais_verifier import install_scoring_observer, prepare_guest_verifier, run_bash_once, scoring_requests


_desktop_session = ContextVar('ais_desktop_session', default=None)
log = logging.getLogger(__name__)


def startup_budget():
    owner = _desktop_session.get()
    if owner is not None:
        owner.check()
        if owner.deadline is not None:
            return max(0.001, owner.deadline - time.monotonic())
    return None


class DesktopSession:
    """Own every guest allocated by one rollout, including before OSWorld reset returns."""
    def __init__(self, session, timeout):
        self.session = session
        self.deadline = time.monotonic() + timeout
        self.guests = {}
        self.error = None

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline and not self.error:
            self.error = 'OSWorld first observation deadline exceeded'
        if self.error:
            raise RuntimeError(self.error)

    def cancelled(self):
        message = self.session.controller.env_input
        return message is not None and message.status == AgentOutputStatus.CANCELLED

    async def watch(self):
        while True:
            previous = self.error
            if self.cancelled():
                self.error = 'OSWorld rollout cancelled'
            elif self.deadline is not None and time.monotonic() >= self.deadline:
                self.error = 'OSWorld first observation deadline exceeded'
            if self.error:
                if not previous:
                    log.warning('Session %s: %s', self.session.id, self.error)
                # Boot runs on a thread. A copy in progress may publish its QEMU
                # process on a later tick. The owning thread performs final cleanup.
                for vm in list(self.guests.values()):
                    process = vm.process
                    if process is not None and process.poll() is None:
                        process.kill()
            await asyncio.sleep(0.5)


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = _desktop_session.get()

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
        # Some TCG desktops take longer than 10s to capture and encode a frame.
        # Give each read more time within the previous ~45s budget.
        started = time.monotonic()
        for attempt in (1, 2):
            if self.owner:
                self.owner.check()
            try:
                response = requests.get(self.http_server + '/screenshot', timeout=(2, 20))
                response.raise_for_status()
                if not self._is_valid_image_response(response.headers.get('Content-Type'), response.content):
                    raise ValueError('invalid image payload')
                log.info('Guest screenshot %s attempt=%d elapsed=%.2fs', self.http_server, attempt, time.monotonic() - started)
                return response.content
            except (requests.RequestException, ValueError) as exc:
                log.warning('Guest screenshot %s attempt=%d elapsed=%.2fs: %s', self.http_server, attempt,
                            time.monotonic() - started, exc)
                if attempt == 2:
                    try:
                        probe = requests.get(self.http_server + '/platform', timeout=2).status_code
                    except requests.RequestException as probe_error:
                        probe = type(probe_error).__name__
                    log.warning('Guest %s /platform after failed screenshot: %s', self.http_server, probe)
                    raise RuntimeError('OSWorld screenshot reads exhausted') from exc
                time.sleep(1)


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
        entry = self.leases.pop(lease, None)
        if entry is not None:
            vm, scratch = entry
            try:
                vm.stop()
            finally:
                scratch.cleanup()

    async def start_session(self, subtypes, **kwargs):
        owner = _desktop_session.get()
        if owner is not None:
            owner.check()
        subtype = subtypes if isinstance(subtypes, str) else subtypes[0]
        if subtype not in ('Ubuntu', 'Ubuntu-modified'):
            raise ValueError(f'Unsupported OSWorld image type: {subtype}')
        lease = uuid.uuid4().hex
        audit = self.audit / lease
        audit.mkdir()
        scratch = tempfile.TemporaryDirectory(prefix='cua-vm-')
        vm = VM(self.tools, Path(scratch.name), audit, 0)
        self.leases[lease] = vm, scratch
        if owner is not None:
            owner.guests[lease] = vm
            log.info('OSWorld session %s allocated guest %s', owner.session.id, lease)
        address = '127.0.0.1'

        def boot():
            path = os.getenv('CUA_RL_VM_TEMPLATE')
            template = Path(path) if path else None
            vm.start(self.image, self.accel, self.cpus, self.ram * 1024, worker_address=address, template=template)
            if owner is not None:
                owner.check()
            vm.wait_desktop(600, 'ready.png', read_timeout=20)
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
        # Upstream constructs DesktopEnv on a dedicated executor thread.
        # Bind its provider allocations to the outer task even if reset raises.
        token = _desktop_session.set(self.session.ais_desktop)
        try:
            env = super().initialize_env(version, config)
        finally:
            _desktop_session.reset(token)
        evaluate = env.evaluate

        def checked_evaluate():
            with scoring_requests(env.controller.http_server):
                return evaluate()

        env.evaluate = checked_evaluate
        return env


class AisOSWorldTask(OSWorldTask):
    async def start_sample_custom(self, task, session):
        owner = DesktopSession(session, self.first_observation_timeout)
        session.ais_desktop = owner
        action = session.action

        async def first_action(*injection):
            owner.check()
            owner.deadline = None
            return await action(*injection)

        original_override = vars(session).get('action')
        session.action = first_action
        watcher = asyncio.create_task(owner.watch())
        execution = asyncio.create_task(super().start_sample_custom(task, session))
        try:
            result = await asyncio.shield(execution)
        except asyncio.CancelledError:
            session.controller.env_input = AgentOutput(status=AgentOutputStatus.CANCELLED)
            session.controller.env_signal.release()
            await asyncio.shield(asyncio.gather(execution, return_exceptions=True))
            raise
        except Exception:
            if owner.cancelled():
                return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
            owner.check()
            raise
        else:
            if owner.cancelled():
                return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
            owner.check()
            return result
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            if original_override is None:
                del session.action
            else:
                session.action = original_override
            del session.ais_desktop
            for lease in owner.guests:
                await self.env_controller.end_session(lease)

    def __init__(self, ais_image, ais_audit, first_observation_timeout=240, **kwargs):
        super().__init__(env_driver='manual', env_options={'urls': {}}, **kwargs)
        self.first_observation_timeout = first_observation_timeout
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
        install_scoring_observer(startup_budget)
