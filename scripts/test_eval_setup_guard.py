"""Reject swallowed setup errors while allowing temporary app startup failures."""
import sys
import types
import unittest
from unittest.mock import patch

import requests
import eval_qemu_provider as adapter


class SetupGuardTests(unittest.TestCase):
    def environment(self, setup):
        class Base:
            def __init__(self):
                self.vm_ip = '127.0.0.1'
                self.server_port = 5000
                self.screen_width, self.screen_height = 1280, 800
                self.setup_controller = types.SimpleNamespace(
                    _screen_size_setup=lambda *a: None,
                    _launch_setup=lambda *a, **kw: None, setup=setup)
                self.controller = types.SimpleNamespace(get_screenshot=lambda: b'png')
            def _start_emulator(self):
                pass
        package = types.ModuleType('desktop_env')
        package.__path__ = []
        package.providers = types.ModuleType('desktop_env.providers')
        package.desktop_env = types.ModuleType('desktop_env.desktop_env')
        package.desktop_env.DesktopEnv = Base
        with patch.dict(sys.modules, {'desktop_env': package, 'desktop_env.providers': package.providers,
                                      'desktop_env.desktop_env': package.desktop_env}):
            cls = adapter.install({'proxy_policy': 'upstream', 'recording_mode': 'steps-only'})
            env = cls()
            with patch.object(adapter.Image, 'open', return_value=types.SimpleNamespace(size=(1280, 800))):
                env._start_emulator()
            return env

    def test_swallowed_http_500_is_not_success(self):
        def upstream_setup():
            requests.post('http://127.0.0.1:5000/chimerax/run')
            return True  # This is the upstream behavior being guarded.
        env = self.environment(upstream_setup)
        with patch.object(requests.sessions.Session, 'request', return_value=types.SimpleNamespace(status_code=500)) as original:
            with self.assertRaisesRegex(RuntimeError, 'Setup HTTP failure'):
                env.setup_controller.setup()
            self.assertIs(requests.sessions.Session.request, original)
            self.assertFalse(env._last_setup_ok)

    def test_startup_retry_does_not_invalidate_successful_setup(self):
        def upstream_setup():
            env.setup_controller._launch_setup(['chimerax', '--cmd', 'remotecontrol rest start port 8000'])
            requests.post('http://127.0.0.1:5000/chimerax/run')
            return True
        env = self.environment(upstream_setup)
        ready = types.SimpleNamespace(status_code=200, json=lambda: {'returncode': 0, 'output': 'ready\n'})
        responses = [requests.ConnectionError('not listening yet'), ready,
                     types.SimpleNamespace(status_code=200, json=lambda: {'error': None})]
        with patch.object(requests.sessions.Session, 'request', side_effect=responses) as api, patch.object(adapter.time, 'sleep'):
            self.assertTrue(env.setup_controller.setup())
            self.assertTrue(env._last_setup_ok)
            self.assertEqual(api.call_args_list[0].args[2], 'http://127.0.0.1:5000/execute')
            command = api.call_args_list[0].kwargs['json']['command']
            self.assertEqual(command[:2], ['python3', '-c'])
            self.assertIn('/proc/net/', command[2])

    def test_chimerax_application_error_inside_http_200_is_rejected(self):
        def setup():
            requests.post('http://127.0.0.1:5000/chimerax/run', json={'command': 'open invalid'})
            return True
        env = self.environment(setup)
        response = types.SimpleNamespace(status_code=200, json=lambda: {'error': 'Download failed'})
        with patch.object(requests.sessions.Session, 'request', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'application error'):
                env.setup_controller.setup()

    def test_readiness_does_not_call_app_data_endpoints(self):
        env = self.environment(lambda: True)
        cases = ['kalgebra 8000', 'celestia-qt6 8000', 'FLASK_PORT=8000 grass']
        response = types.SimpleNamespace(status_code=200, json=lambda: {'returncode': 0, 'output': 'ready\n'})
        for command in cases:
            with self.subTest(app=command), patch.object(requests.sessions.Session, 'request', return_value=response) as api, patch.object(adapter.time, 'sleep'):
                env.setup_controller._launch_setup(command, shell=True)
                self.assertEqual(api.call_args.kwargs['method'].upper(), 'POST')
                self.assertEqual(api.call_args.kwargs['url'], 'http://127.0.0.1:5000/execute')
                self.assertIn('/proc/net/', api.call_args.kwargs['json']['command'][2])


if __name__ == '__main__':
    unittest.main()
