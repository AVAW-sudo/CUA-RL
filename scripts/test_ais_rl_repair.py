"""Regression checks for Frontier recovery, trainer sample integrity and checkpoint reporting."""
import ast
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ais_frontier_shards import require_equivalent_protocol
from ais_frontier_reuse import score_windows
from collect_frontier_rates import release_failed_start
from ais_pilot import checkpoint_capacity, final_checkpoint
from ais_rl_job import run_job
from runtime_env import launch_ais

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'SCALE-CUA/scalecua_rl/trainer/src/agentrl/trainer'


class RepairTests(unittest.TestCase):
    def test_rl_entry_propagates_verified_assembler_or_stops_before_ray(self):
        image_tool = '/usr/local/cuda/bin/ptxas'
        training_tool = '/training/triton/backends/nvidia/bin/ptxas'
        with patch.dict(os.environ, TRITON_PTXAS_PATH=image_tool), \
                patch('runtime_env.probe_interpreter', return_value={'selected': 0, 'attempts': [{'passed': True}]}), \
                patch('importlib.util.find_spec', return_value=True), patch('ais_frontier_server.check_cuda') as rotary, \
                patch('os.execve') as launch:
            rotary.return_value = {'ptxas_path': training_tool, 'vision_rotary_cold_compile': 'passed'}
            launch_ais(['scripts/ais_distributed_rl.py'])
            env = launch.call_args.args[-1]
            self.assertEqual(env['TRITON_PTXAS_PATH'], training_tool)
            receipt = json.loads(env['CUA_AIS_CUDA_PROBE'])['vision_preflight']
            self.assertEqual(receipt['inherited_ptxas_path'], image_tool)
            self.assertEqual(receipt['vision_rotary_cold_compile'], 'passed')

            launch.reset_mock()
            rotary.side_effect = RuntimeError('kernel load failed')
            with self.assertRaises(SystemExit):
                launch_ais(['scripts/ais_distributed_rl.py'])
            launch.assert_not_called()

    def test_preflight_failure_skips_desktop(self):
        with patch('ais_rl_job.start_desktop') as start:
            self.assertEqual(run_job([sys.executable, '-c', 'raise SystemExit(2)']), 2)
        start.assert_not_called()

    def test_colocated_service_lifetime_and_failure(self):
        trainer = [sys.executable, '-c',
                   'from pathlib import Path\n'
                   'import os\n'
                   'from ais_rl_job import local_controller\n'
                   'assert local_controller(Path(os.environ["CUA_LOCAL_DESKTOP_DIR"])) == "http://127.0.0.1/test"']
        with patch.dict(os.environ, PYTHONPATH=str(ROOT / 'scripts')):
            for healthy in (True, False):
                with self.subTest(healthy=healthy):
                    children = []
                    def start(directory):
                        command = ['sleep', '60'] if healthy else [sys.executable, '-c', 'raise SystemExit(3)']
                        child = subprocess.Popen(command, start_new_session=True)
                        children.append(child)
                        if healthy:
                            (directory / 'ready.json').write_text(json.dumps({'controller': 'http://127.0.0.1/test'}))
                        return child
                    with patch('ais_rl_job.start_desktop', side_effect=start):
                        if healthy:
                            self.assertEqual(run_job(trainer), 0)
                        else:
                            with self.assertRaisesRegex(RuntimeError, 'OSWorld service exited.*code 3'):
                                run_job(trainer)
                    self.assertIsNotNone(children[0].poll())

    def test_reuse_rejects_live_sampling_changes_before_collection(self):
        service = dict(model_revision='r', served_model_name='m', runtime_compatibility=None,
                       command=['python','-m','vllm','--model','path','--host','localhost','--port','1','--dtype','bfloat16'])
        previous = {'services':[service], 'temperature':0.8, 'indices_sha256':'tasks'}
        live = dict(previous, services=[dict(service, command=service['command'].copy())])
        live['services'][0]['command'][8] = '2'
        require_equivalent_protocol(previous, live)
        with self.assertRaises(ValueError):
            require_equivalent_protocol(previous,dict(live, temperature=1))
        live['services'][0]['command'][-1] = 'float16'
        with self.assertRaises(ValueError):
            require_equivalent_protocol(previous, live)

    def test_legacy_scoring_errors_are_bound_to_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'worker.log'
            path.write_text('Evaluating final score for session_id=4...\nTraceback\n'
                'Evaluating final score for session_id=5...\n[ERROR] [python.py:12]: Failed to get file. Status code: 404\n'
                '[ERROR] [python.py:12]: Failed to get file.\nFinal score: 0.0, session_id=5\n'
                'Evaluating final score for session_id=6...\n[ERROR] [python.py:12]: Failed to get accessibility tree. Status code: 500\n'
                'Final score: 0.0, session_id=6\n')
            windows = score_windows(path)
            self.assertEqual(set(windows), {'5', '6'})
            self.assertEqual(windows['5']['errors'], [])
            self.assertEqual(windows['5']['file_404s'], 1)
            self.assertIn('500', windows['6']['errors'][0])

    def test_private_worker_cleanup_targets_only_assigned_session(self):
        client = Mock()
        response = Mock(status=200)
        request = AsyncMock()
        request.__aenter__.return_value = response
        client.post.return_value = request
        node = 'http://127.0.0.1:18121/api'
        self.assertIsNone(asyncio.run(release_failed_start(client, node, SimpleNamespace(headers={}))))
        client.post.assert_not_called()
        result = asyncio.run(release_failed_start(client, node, SimpleNamespace(headers={'session_id': '7'})))
        self.assertEqual(result['status'], 'released')
        client.post.assert_called_once_with(node + '/cancel', json={'session_id': 7}, timeout=240)
        response.status = 400
        response.text = AsyncMock(return_value='{"detail":"No such session"}')
        result = asyncio.run(release_failed_start(client, node, SimpleNamespace(headers={'session_id': '7'})))
        self.assertEqual(result['status'], 'already_released')

    @patch.dict(os.environ, CUA_STRICT_OSWORLD_REWARDS='1')
    def test_completed_status_guard(self):
        source = ast.parse((TRAINER/'agentic/loops.py').read_text())
        custom = next(f for f in source.body if getattr(f,'name','') == 'openai_chat_agent_loop_custom')
        loop = next(f for f in custom.body if isinstance(f, ast.For))
        statement = next(f for f in loop.body if isinstance(f, ast.If) and "CUA_STRICT_OSWORLD_REWARDS" in ast.unparse(f.test))
        mock = AsyncMock()
        context = {'end_fn':mock, 'os':os}
        test_fn = ast.parse('async def check(obs, sid=1, task_index=42):\n    return True').body[0]
        test_fn.body.insert(0,statement)
        exec(compile(ast.Module([test_fn], []), '<actual terminal guard>', 'exec'), context)
        def usable(status, reward):
            return asyncio.run(context['check'](dict(finish=True,status=status,reward=reward)))
        for score in (0, 0.5, 1):
            self.assertTrue(usable('completed',score))
        self.assertTrue(usable('agent invalid action',0))
        for status, score in [('completed',float('nan')), ('completed',None), ('task error',0), ('cancelled',0)]:
            self.assertIsNone(usable(status,score))
        self.assertEqual(mock.await_count,4)

    @patch.dict(os.environ, CUA_STRICT_OSWORLD_REWARDS='1')
    def test_incomplete_group_preserves_accounting(self):
        spec = importlib.util.spec_from_file_location('buffer', TRAINER/'components/buffer.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        async def feed():
            buffer = mod.Buffer(50,8)
            for i in range(8):
                await buffer.add({'group_id':1,'results':[i]})
                await buffer.add({'group_id':2,'results':None if i==3 else [i]})
            self.assertEqual(await buffer.get(16), [[i] for i in range(8)] + [None]*8)
        asyncio.run(feed())

    def test_checkpoint_requires_payload_and_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            step = run/'global_step_25'
            (run/'latest_checkpointed_iteration.txt').write_text('25')
            with self.assertRaises(ValueError):
                final_checkpoint(run,25,2)
            for name in ['huggingface/config.json','huggingface/weights.safetensors','model/mp_rank_00/model.pt',
                         'model/mp_rank_01/model.pt','optim/first.pt','rng_states/one.pt']:
                p = step/name
                p.parent.mkdir(exist_ok=True,parents=True)
                p.write_bytes(b'x')
            (step/'huggingface/model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'weights.safetensors'}}))
            self.assertEqual(final_checkpoint(run,25,2),step)
            (step/'model/mp_rank_00/model.pt').write_bytes(b'')
            with self.assertRaises(ValueError):
                final_checkpoint(run,25,2)
            with patch('ais_pilot.shutil.disk_usage',return_value=Mock(free=900)):
                with self.assertRaisesRegex(ValueError,'model copies'):
                    checkpoint_capacity({'metadata':{'total_size':500}},['model','hf_model'],run/'new')
