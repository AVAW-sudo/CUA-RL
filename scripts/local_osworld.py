"""OSWorld worker adapter for isolated QEMU TCG smoke episodes.

The upstream task, agent and verifier remain in use. Only VM allocation changes.
This is a single-worker fallback and is not a KVM throughput reproduction.
"""
import asyncio
import atexit
from functools import partial
from pathlib import Path

from agentrl.worker.environment import EnvironmentController
import desktop_env.desktop_env
import desktop_env.providers
from src.task import OSWorldTask
from src.environment import create_manager_and_provider
from qemu_runtime import LocalVM, ROOT


class QemuController(EnvironmentController):
    def __init__(self, delegation, concurrency=2):
        super().__init__(delegation)
        self.addresses = asyncio.Queue()
        for index in range(concurrency):
            self.addresses.put_nowait(f"127.0.0.{20 + index}")
        self.instances = {}
        atexit.register(self.stop_all)

    def stop_all(self):
        for vm in self.instances.values():
            vm.stop()

    async def start_session(self, subtypes, **kwargs):
        subtype = subtypes if isinstance(subtypes, str) else subtypes[0]
        if subtype not in {"Ubuntu", "Ubuntu-modified"}:
            raise ValueError(f"Local QEMU smoke only supports Ubuntu, got {subtype}")
        address = await self.addresses.get()
        vm = LocalVM(address=address)
        self.instances[vm.id] = vm
        try:
            await asyncio.to_thread(vm.start)
            await asyncio.to_thread(vm.wait_ready)
        except BaseException:
            await self.end_session(vm.id)
            raise
        return vm.id, {subtype: vm.id}, {subtype: address}

    async def renew_session(self, session_id):
        pass

    async def end_session(self, session_id):
        vm = self.instances.pop(session_id, None)
        if vm:
            await asyncio.to_thread(vm.stop)
            self.addresses.put_nowait(vm.address)


class LocalOSWorldTask(OSWorldTask):
    def __init__(self, **kwargs):
        kwargs["env_driver"] = "manual"
        kwargs["env_options"] = {"urls": {}}
        super().__init__(**kwargs)
        self.env_controller = QemuController(self.env_delegation, min(2, kwargs.get("concurrency", 1)))
        factory = partial(create_manager_and_provider, self.env_controller)
        desktop_env.providers.create_vm_manager_and_provider = factory
        desktop_env.desktop_env.create_vm_manager_and_provider = factory
