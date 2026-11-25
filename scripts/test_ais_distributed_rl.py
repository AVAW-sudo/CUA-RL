"""Regression checks for the AIS multi-entry pilot and its pinned upstream changes."""
import ast
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ais_distributed_rl as job

TRAINER = job.ROOT / 'SCALE-CUA/scalecua_rl/trainer/src/agentrl/trainer'


def upstream_method(file, cls, name, namespace):
    tree = ast.parse((TRAINER / file).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    exec(compile(ast.Module([method], []), str(file), 'exec'), namespace)
    return namespace[name]


class DistributedJobTests(unittest.TestCase):
    def test_heartbeat_uses_local_elapsed_time_and_ignores_replayed_or_unreadable_records(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(job.time, 'monotonic') as clock, \
                patch.object(job.time, 'sleep', side_effect=AssertionError('poll blocked')):
            path = Path(directory) / 'rank-0.json'
            clock.return_value = 0
            peers = job.PeerHeartbeats([path])
            peers.poll()  # Head has not published yet.
            for seq, timestamp in [(1, 10), (2, 160), (3, 310)]:
                clock.return_value = timestamp
                job.save(path, {'stage': 'ready', 'phase': 'GRPO driver', 'heartbeat': seq})
                os.utime(path, (1, 1))
                peers.poll()
            self.assertTrue(peers.all_ready())
            clock.return_value = 485
            job.save(path, {'stage': 'startup', 'phase': 'Ray head', 'heartbeat': 2})
            peers.poll()
            self.assertTrue(peers.all_ready())
            path.write_text('{"stage":')
            peers.poll()
            with patch.object(Path, 'read_text', side_effect=OSError(errno.ESTALE, 'stale')):
                peers.poll()
            clock.return_value = 490
            with self.assertRaisesRegex(RuntimeError, 'rank-0 stopped reporting for 180s.*last heartbeat 3 during GRPO driver'):
                peers.poll()

    def test_startup_timeout_and_explicit_peer_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(job.time, 'monotonic') as clock:
            path = Path(directory) / 'rank-7.json'
            clock.return_value = 0
            peers = job.PeerHeartbeats([path])
            clock.return_value = job.HEARTBEAT_TIMEOUT
            with self.assertRaisesRegex(RuntimeError, 'rank-7.*no heartbeat observed'):
                peers.poll()
            job.save(path, {'stage': 'failed', 'error': 'desktop exited'})
            with self.assertRaisesRegex(RuntimeError, 'rank-7: desktop exited'):
                peers.poll()

    def test_worker_detects_frozen_head_during_desktop_registration_and_reaps_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / 'local'
            local.mkdir()
            head = root / 'rank-0.json'
            job.save(head, {'stage': 'ready', 'phase': 'all desktop workers', 'heartbeat': 12})
            os.utime(head, (4_000_000_000, 4_000_000_000))
            job.save(root / 'head.json', {})
            job.save(root / 'placement.json', {})
            job.save(root / 'controller.json', {'url': 'http://127.0.0.1:18520/api'})
            states = [{'rank': i, 'runtime': {'ip': '127.0.0.1'}} for i in range(8)]
            info = dict(run=directory, address='127.0.0.1:12345', participants=states)
            popen = subprocess.Popen
            children = []

            def spawn(command, **kwargs):
                child = popen(['sleep', '100'], **kwargs)
                children.append(child)
                return child

            with patch.dict(os.environ, CUA_CONTROLLER_URL=''), patch.object(job, 'HEARTBEAT_INTERVAL', 0.1), \
                    patch.object(job, 'HEARTBEAT_TIMEOUT', 1), patch.object(job.subprocess, 'Popen', side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError, 'rank-0 stopped reporting.*last heartbeat 12'):
                    job.supervise(info, 1, local, [])
            failure = job.read_rank_status(root / 'rank-1.json')
            self.assertEqual((failure['stage'], failure['phase']), ('failed', 'desktop worker registration'))
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.poll() is not None for child in children))

    def test_dispatch_preserves_single_entry(self):
        env = dict(os.environ, CUA_TRAIN_PYTHON='echo', CUA_CONTROLLER_URL='http://example.invalid/api')
        script = str(job.ROOT / 'scripts/ais_entry.sh')
        for size, target in [('8', 'ais_distributed_rl.py'), ('1', 'launch_local_rl.py')]:
            output = subprocess.check_output([script, 'train'], env=dict(env, WORLD_SIZE=size), text=True)
            self.assertIn(target, output)

    def test_weight_buckets_preserve_every_tensor(self):
        ns = {'dist': Mock(), 'send_buffer': Mock(), 'broadcast_object_list': Mock()}
        send = upstream_method('components/nccl_tensor_comm.py', 'NCCLTensorSender', 'send', ns)
        for sizes in ([5, 5], [4, 7, 2, 1, 14], [11], [1, 2]):
            with self.subTest(sizes=sizes):
                tensors = [(str(i), Mock(numel=Mock(return_value=n), element_size=Mock(return_value=1)))
                           for i, n in enumerate(sizes)]
                ns['send_buffer'].reset_mock()
                send(SimpleNamespace(worker_rank=0, pg='group',
                                     worker=SimpleNamespace(param_generator=lambda: iter(tensors))), 10)
                batches = [call.args[0] for call in ns['send_buffer'].call_args_list]
                self.assertTrue(all(batches))
                self.assertEqual([item for batch in batches for item in batch], tensors)

    def test_first_weight_update_waits_for_engine_socket(self):
        update = upstream_method('workers/vllm/async_vllm_worker.py', 'AsyncvLLMBaseWorker', 'update_params',
                                 {'asyncio': asyncio, 'defaultdict': defaultdict, 'torch': Mock()})

        async def check():
            connected = False

            async def connect():
                nonlocal connected
                await asyncio.sleep(0)
                connected = True

            async def send(dtype, tensors):
                self.assertTrue(connected)
                self.assertEqual(dtype, 'bfloat16')

            worker = SimpleNamespace(receiver_ready=asyncio.run_coroutine_threadsafe(connect(), asyncio.get_running_loop()),
                                     _update_params_group=send)
            await update(worker, [('weight', SimpleNamespace(dtype='torch.bfloat16'))])

        asyncio.run(check())

    def test_task_threads_preserve_capacity(self):
        worker = Mock()
        start = upstream_method('components/task_manager.py', 'DistributedTaskManager', 'start',
                                {'distributed_worker': worker})
        for count, threads, sizes in [(1, 2, [1]), (5, 2, [2, 2, 1]), (32, 2, [2] * 16)]:
            worker.reset_mock()
            start(SimpleNamespace(_started=False, num_workers=count, num_threads=threads,
                                  queue=None, buffer=None, task_fn=None, workers=[]))
            self.assertEqual([c.args[-1] for c in worker.remote.call_args_list], sizes)

    def test_eight_supervisors_share_one_driver_and_propagate_exit(self):
        # Exercise the lifecycle and shared files using small child processes in place of Ray and GRPO.
        popen = subprocess.Popen
        for driver_code, args in [(0, []), (7, []), (0, ['--check-only'])]:
            with self.subTest(driver_code=driver_code, args=args), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                states = [{'rank': i, 'runtime': {'ip': '127.0.0.1'}} for i in range(8)]
                info = dict(run=directory, address='127.0.0.1:12345', participants=states)
                launched = []
                lock = threading.Lock()

                def spawn(command, **kwargs):
                    if 'ray.scripts.scripts' in command:
                        name = command[command.index('--node-name') + 1]
                        self.assertNotIn('WORLD_SIZE', kwargs['env'])
                        substitute = ['sleep', '100']
                    elif '--verify' in command:
                        name = 'verify'
                        job.save(root / 'head.json', {})
                        job.save(root / 'placement.json', {})
                        substitute = [sys.executable, '-c', 'pass']
                    elif 'rl-desktop' in command:
                        name = 'desktop'
                        self.assertEqual(command[command.index('--concurrency') + 1], '2')
                        target = Path(command[command.index('--ready-file') + 1])
                        if target.parent.name != 'local-0':
                            self.assertEqual(command[command.index('--controller-url') + 1], controller)
                        job.save(target, {'controller': controller})
                        substitute = ['sleep', '100']
                    else:
                        self.assertIn('--ais-pilot', command)
                        name = 'driver'
                        self.assertEqual(kwargs['env']['CUA_CONTROLLER_URL'], controller)
                        for path in root.glob('rank-*.json'):
                            self.assertEqual(job.read_rank_status(path)['stage'], 'ready')
                        substitute = [sys.executable, '-c', f'raise SystemExit({driver_code})']
                    process = popen(substitute, **kwargs)
                    with lock:
                        launched.append((name, process))
                    return process

                controller = 'http://127.0.0.1:18520/api'
                with patch.dict(os.environ, CUA_CONTROLLER_URL=controller if driver_code else ''), \
                        patch.object(job, 'HEARTBEAT_INTERVAL', 0.25), \
                        patch.object(job.subprocess, 'Popen', side_effect=spawn), ThreadPoolExecutor(8) as pool:
                    futures = []
                    for rank in range(8):
                        local = root / f'local-{rank}'
                        local.mkdir()
                        futures.append(pool.submit(job.supervise, info, rank, local, args))
                    errors = [f.exception(timeout=25) for f in futures]
                self.assertEqual([bool(e) for e in errors], [bool(driver_code)] * 8)
                names = [name for name, _ in launched]
                self.assertEqual(names.count('driver'), int(not args))
                self.assertEqual(names.count('desktop'), 8 if not args and not driver_code else 0)
                self.assertEqual(names.count('verify'), 1)
                self.assertTrue(all(c.poll() is not None for _, c in launched))
                self.assertEqual(json.loads((root / 'result.json').read_text())['passed'], not driver_code)


if __name__ == '__main__':
    unittest.main()
