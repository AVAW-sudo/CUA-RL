"""OSWorld worker adapter for isolated QEMU TCG smoke episodes.

The upstream task, agent and verifier remain in use. Only VM allocation changes.
This is a single-worker fallback and is not a KVM throughput reproduction.
"""
import asyncio
import atexit
import time
import json
import socket
import uuid
from functools import partial
from pathlib import Path

from agentrl.worker.environment import EnvironmentController
import desktop_env.desktop_env
import desktop_env.providers
from src.task import OSWorldTask
from src.environment import create_manager_and_provider
from qemu_runtime import LocalVM, ROOT


class QemuController(EnvironmentController):
    def __init__(self, delegation, concurrency=2, cpus=2, memory_mb=4096, warm_reset_report=None):
        super().__init__(delegation)
        if not 1 <= concurrency <= 32:
            raise ValueError('Local rollout concurrency must be in 1..32')
        self.cpus, self.memory_mb = cpus, memory_mb
        self.addresses = asyncio.Queue()
        for index in range(concurrency):
            self.addresses.put_nowait(f"127.0.0.{20 + index}")
        self.instances = {}
        self.idle = {}
        self.warm = bool(warm_reset_report)
        if self.warm:
            from snapshot_vm import implementation_hash
            report = json.loads(Path(warm_reset_report).read_text())
            if (report.get('status') != 'passed' or report.get('hostname') != socket.gethostname()
                    or report.get('count', 0) < concurrency or not report.get('cleanup_passed')
                    or report.get('implementation_sha256') != implementation_hash()):
                raise RuntimeError('Warm reset requires matching concurrent acceptance on this host')
        self.launch_lock = asyncio.Lock()
        self.last_launch = 0.0
        atexit.register(self.stop_all)

    def stop_all(self):
        for vm in list(self.instances.values()) + list(self.idle.values()):
            vm.stop()

    async def start_session(self, subtypes, **kwargs):
        subtype = subtypes if isinstance(subtypes, str) else subtypes[0]
        if subtype not in {"Ubuntu", "Ubuntu-modified"}:
            raise ValueError(f"Local QEMU smoke only supports Ubuntu, got {subtype}")
        address = await self.addresses.get()
        vm = self.idle.pop(address, None)
        reused = vm is not None
        if vm is None:
            from snapshot_vm import SnapshotVM
            cls = SnapshotVM if self.warm else LocalVM
            vm = cls(address=address, cpus=self.cpus, memory_mb=self.memory_mb)
        lease_id = uuid.uuid4().hex
        self.instances[lease_id] = vm
        try:
            # Match the paced concurrency acceptance: avoid a simultaneous
            # boot storm while allowing all already-started VMs to keep running.
            if not reused:
                async with self.launch_lock:
                    await asyncio.sleep(max(0, 1.0 - (time.monotonic() - self.last_launch)))
                    await asyncio.to_thread(vm.start)
                    self.last_launch = time.monotonic()
                await asyncio.to_thread(vm.wait_ready)
                if self.warm:
                    await asyncio.to_thread(vm.save_clean)
            elif vm.process.poll() is not None:
                raise RuntimeError('Idle warm VM exited; replace on the next initialization attempt')
        except BaseException:
            self.instances.pop(lease_id, None)
            await asyncio.to_thread(vm.stop)
            self.addresses.put_nowait(address)
            raise
        return lease_id, {subtype: lease_id}, {subtype: address}

    async def renew_session(self, session_id):
        pass

    async def end_session(self, session_id):
        vm = self.instances.pop(session_id, None)
        if vm:
            try:
                if self.warm:
                    await asyncio.to_thread(vm.restore_clean)
                    self.idle[vm.address] = vm
                else:
                    await asyncio.to_thread(vm.stop)
            except Exception as exc:
                # The completed episode keeps its result. A bad reusable VM is
                # discarded and the next lease gets a fresh cold-booted guest.
                (vm.audit_directory / 'reset-error.json').write_text(json.dumps({'error':repr(exc)}))
                await asyncio.to_thread(vm.stop)
            finally:
                self.addresses.put_nowait(vm.address)


class LocalOSWorldTask(OSWorldTask):
    def __init__(self, **kwargs):
        kwargs["env_driver"] = "manual"
        kwargs["env_options"] = {"urls": {}}
        warm_reset_report = kwargs.pop('warm_reset_report', None)
        super().__init__(**kwargs)
        self.env_controller = QemuController(self.env_delegation, kwargs.get("concurrency", 1),
                                            kwargs.get('vm_cpu', 2), kwargs.get('vm_ram', 4) * 1024,
                                            warm_reset_report)
        factory = partial(create_manager_and_provider, self.env_controller)
        desktop_env.providers.create_vm_manager_and_provider = factory
        desktop_env.desktop_env.create_vm_manager_and_provider = factory
