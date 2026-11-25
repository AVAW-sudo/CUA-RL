"""Regression checks for the AIS multi-entry pilot and its pinned upstream changes."""
import ast
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
                    else:
                        self.assertIn('--ais-pilot', command)
                        name = 'driver'
                        substitute = [sys.executable, '-c', f'raise SystemExit({driver_code})']
                    process = popen(substitute, **kwargs)
                    with lock:
                        launched.append((name, process))
                    return process

                with patch.dict(os.environ, CUA_CONTROLLER_URL='http://127.0.0.1:18520/api'), \
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
                self.assertEqual(names.count('verify'), 1)
                self.assertTrue(all(c.poll() is not None for _, c in launched))
                self.assertEqual(json.loads((root / 'result.json').read_text())['passed'], not driver_code)


if __name__ == '__main__':
    unittest.main()
