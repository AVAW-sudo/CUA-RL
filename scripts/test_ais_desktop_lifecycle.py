"""Run with the OSWorld worker interpreter and its staged PYTHONPATH."""
import asyncio
import io
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import ais_osworld_worker as worker
except ModuleNotFoundError as exc:
    if exc.name not in ('desktop_env', 'agentrl', 'src'):
        raise
    worker = None


@unittest.skipUnless(worker, 'requires the staged OSWorld worker')
class DesktopLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from agentrl.worker.task import Session
        self.session = Session(101)
        self.session.loop = asyncio.get_running_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.controller = worker.AisEnvironmentController(SimpleNamespace(), root, root / 'audit', 4, 8)
        self.controller.loop = self.session.loop
        self.task = object.__new__(worker.AisOSWorldTask)
        self.task.env_controller = self.controller
        self.task.first_observation_timeout = 30
        self.vm = None
        self.started = asyncio.Event()

    async def simulate(self, injected_reset):
        def launch(vm, *args, **kwargs):
            self.vm = vm
            vm.process = subprocess.Popen(['sleep', '60'])
            self.session.loop.call_soon_threadsafe(self.started.set)

        def reset(os_session, version, config):
            self.controller.sync_start_session('Ubuntu')
            return injected_reset()

        async def start(task, index, session):
            os_session = object.__new__(worker.AisOSWorldSession)
            os_session.session = session
            await asyncio.to_thread(os_session.initialize_env, None, {})
            return await session.action()

        with patch.object(worker.VM, 'start', launch), patch.object(worker.VM, 'wait_desktop'), \
                patch.object(worker, 'prepare_guest_verifier'), patch.object(worker, 'check_sft_restore'), \
                patch.object(worker.OSWorldSession, 'initialize_env', reset), \
                patch.object(worker.OSWorldTask, 'start_sample_custom', start):
            return await self.task.start_sample_custom({}, self.session)

    def assert_released(self):
        self.assertIsNotNone(self.vm.process.poll())
        self.assertFalse(self.vm.work.parent.exists())
        self.assertEqual(self.controller.leases, {})

    async def test_reset_raises_before_env_is_returned(self):
        def reset():
            raise RuntimeError('injected screenshot failure')
        with self.assertRaisesRegex(RuntimeError, 'injected screenshot failure'):
            await self.simulate(reset)
        self.assert_released()

    async def test_start_deadline_reaps_owned_guest(self):
        self.task.first_observation_timeout = 0.15
        def reset():
            self.vm.process.wait(timeout=4)
            return SimpleNamespace(evaluate=lambda: 0)
        with self.assertRaisesRegex(RuntimeError, 'first observation deadline'):
            await asyncio.wait_for(self.simulate(reset), 5)
        self.assert_released()

    async def test_cancel_during_reset_reaps_guest(self):
        def reset():
            self.vm.process.wait(timeout=4)
            return SimpleNamespace(evaluate=lambda: 0)
        task = asyncio.create_task(self.simulate(reset))
        await asyncio.wait_for(self.started.wait(), 3)
        self.session.controller.env_input = worker.AgentOutput(status=worker.AgentOutputStatus.CANCELLED)
        result = await asyncio.wait_for(task, 5)
        self.assertEqual(result.status, worker.SampleStatus.CANCELLED)
        self.assert_released()

    async def test_delivered_observation_lifts_start_deadline(self):
        self.task.first_observation_timeout = 0.2
        async def action():
            await asyncio.sleep(0.6)
            self.assertIsNone(self.vm.process.poll())
            return 'next action'
        self.session.action = action
        result = await self.simulate(lambda: SimpleNamespace(evaluate=lambda: 0))
        self.assertEqual(result, 'next action')
        self.assertFalse(hasattr(self.session, 'ais_desktop'))
        self.assert_released()

    def test_screenshot_retries_only_the_read(self):
        png = io.BytesIO()
        worker.Image.new('RGB', (8, 8)).save(png, format='PNG')
        client = worker.AisScreenshotController('Ubuntu', '127.0.0.1', 49100)
        with patch.object(worker.requests, 'get', side_effect=[
                worker.requests.ReadTimeout(), Mock(content=png.getvalue(), headers={})]) as get, \
                patch.object(worker.time, 'sleep'):
            self.assertEqual(client.get_screenshot(), png.getvalue())
        self.assertEqual([c.args[0] for c in get.call_args_list], [client.http_server + '/screenshot'] * 2)

    def test_reset_requests_respect_remaining_budget(self):
        with patch.object(worker.requests.Session, 'send', autospec=True) as send:
            # Avoid installing over an annotated mock.
            send.ais_scoring_observer = False
            worker.install_scoring_observer(lambda: 9)
            with worker.requests.Session() as http:
                http.get('http://localhost/unused')
                self.assertEqual(send.call_args.kwargs['timeout'], 9)
                http.get('http://localhost/unused', timeout=(1, 12))
                self.assertEqual(send.call_args.kwargs['timeout'], (1, 9))


if __name__ == '__main__':
    unittest.main()
