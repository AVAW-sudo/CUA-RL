"""Exercise the same collector flags and HTTP boundaries that the AIS entry uses."""
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FrontierCollectorTest(unittest.TestCase):
    def test_ais_probe_and_shard_use_the_submitted_service(self):
        calls = Counter()
        counter_lock = Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                payload = json.loads(body) if body else None
                with counter_lock:
                    calls[self.path] += 1
                    request_number = calls[self.path]
                if self.path == '/start_sample':
                    self.server.test.assertEqual(payload['custom_task']['index'], 'test-task')
                    data = {'finish': False, 'messages': [{'role': 'user', 'content': 'test'}]}
                elif self.path == '/chat/completions':
                    self.server.test.assertEqual(payload['model'], 'test-model')
                    data = {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]}
                elif self.path == '/interact':
                    data = {'finish': True, 'status': 'completed', 'reward': 1}
                    if request_number == 6:
                        data = {'finish': True, 'status': 'task error'}
                else:
                    self.server.test.assertEqual(self.path, '/cancel')
                    data = {}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('session_id', '1')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, *_):
                pass

        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            server.test = self
            service = f'http://127.0.0.1:{server.server_port}'
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = Path(tmp)
            meta = base / 'service.json'
            meta.write_text(json.dumps({'checkpoint': 'osworld-sft', 'base_url': service, 'hostname': 'fixture',
                                        'model_revision': 'fixture', 'served_model_name': 'test-model',
                                        'command': ['test-service', '--port', str(server.server_port)]}))
            pool, tasks = base / 'pool.json', base / 'tasks.json'
            pool.write_text(json.dumps({'checkpoint': 'osworld-sft', 'service_states': [str(meta)]}))
            tasks.write_text(json.dumps(['test-task']))
            cmd = [sys.executable, str(ROOT / 'scripts/collect_frontier_rates.py'), '--indices', str(tasks),
                   '--primary-service', str(meta), '--service-pool', str(pool), '--controller', service,
                   '--task-name', 'osworld-pilot', '--concurrency', '1', '--max-attempts', '1']
            try:
                for phase, extra in [('probe', ['--probe-one', 'test-task']), ('shard', [])]:
                    folder = base / phase
                    rates = base / (phase + '.json')
                    subprocess.run(cmd + ['--output', str(folder), '--rates-output', str(rates), *extra], check=True,
                                   capture_output=True, text=True, timeout=40)
                    if phase == 'probe':
                        self.assertEqual(calls['/chat/completions'], 1)
                        self.assertEqual(json.loads((folder / 'probe.json').read_text())['result']['result'], 1)
                        self.assertFalse(rates.exists())
                    else:
                        self.assertEqual(calls['/chat/completions'], 5)
                        self.assertEqual(json.loads(rates.read_text()), {'osworld-pilot': {'test-task': 1.0}})
                self.assertEqual(calls, {route: 5 for route in ('/start_sample', '/chat/completions', '/interact', '/cancel')})
                folder = base / 'screen'
                rates = base / 'screen-rates.json'
                subprocess.run(cmd + ['--output', str(folder), '--rates-output', str(rates), '--screening',
                                      '--concurrency', '2'], check=True, capture_output=True, text=True, timeout=40)
                self.assertEqual(len(list(folder.glob('*/*/result.json'))), 3)
                self.assertFalse(rates.exists())
                attempts = [json.loads(p.read_text()) for p in folder.glob('*/*/attempt-01.json')]
                self.assertEqual(sum(a['status'] == 'infrastructure_error' for a in attempts), 1)
            finally:
                server.shutdown()
                worker.join()


if __name__ == '__main__':
    unittest.main()
