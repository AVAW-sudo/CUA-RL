import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml
from build_initial_rates import collect
from checkpoint_specs import checkpoint_spec
import pipeline_supervisor as supervisor
import original_rl_pipeline as stages
from run_benchmark import ROOT
from write_evaluation_report import generate, summarize_campaign
from trainer_save_boundary import add_final_save


class PipelineTests(unittest.TestCase):
    def state(self):
        return {'hostname': 'research-a-worker-1', 'pid': 10, 'status': 'running', 'workers': 4,
                'protocol': {'checkpoint': 'osworld-sft', 'benchmark': 'osworld', 'model_revision': 'pinned',
                             'cpus': 4, 'memory_mb': 8192, 'proxy_policy': 'direct', 'task_timeout_seconds': 3600},
                'launch_history': [{}], 'tasks': {
                    'a/1': {'domain': 'a', 'status': 'completed', 'score': 0, 'attempts': []},
                    'a/2': {'domain': 'a', 'status': 'pending', 'score': None, 'attempts': []}}}

    def test_report_keeps_missing_denominator_and_refuses_early_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); state = self.state()
            (directory / 'campaign.json').write_text(json.dumps(state))
            result = summarize_campaign(directory)
            self.assertIsNone(result['overall']['success_rate'])
            self.assertEqual(result['full_denominator_success_bounds'], [0, .5])
            with self.assertRaises(ValueError):
                generate([directory], directory / 'report', final=True)

    def test_rolling_upgrade_drains_only_one_coordinator(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / str(i) for i in range(2)]
            for p in paths:
                p.mkdir(); (p / 'campaign.json').write_text(json.dumps(self.state()))
            state = {}
            with patch.object(supervisor, 'campaign_live', return_value=True), patch.object(supervisor, 'remote') as remote:
                self.assertTrue(supervisor.maintain_campaigns(paths, state, upgrade=True))
                self.assertEqual(remote.call_count, 1)
                self.assertIn('SIGUSR1', remote.call_args.args[1][-1])

    def test_resume_preserves_protocol_and_never_requests_blanket_error_retries(self):
        state = self.state()
        with patch.object(supervisor, 'remote', return_value=types.SimpleNamespace(stdout='{"pid":20}')) as remote:
            supervisor.resume(Path('/test/campaign'), state)
            command = remote.call_args.args[1]
            self.assertNotIn('--retry-errors', command)
            self.assertEqual(command[command.index('--cpus') + 1], '4')
            self.assertEqual(command[command.index('--proxy-policy') + 1], 'direct')

    def test_frontier_rejects_missing_and_duplicate_runs(self):
        rows = [{'model': 'sft', 'task_index': 'a', 'run_number': i, 'result': int(i == 1)} for i in range(1, 5)]
        self.assertEqual(collect(rows, ['a'], 'sft'), {'a': .25})
        for invalid in [rows[:3], rows + rows[:1]]:
            with self.assertRaises(ValueError):
                collect(invalid, ['a'], 'sft')

    def test_original_recipe_is_preserved_and_variants_are_disabled(self):
        local = yaml.safe_load((ROOT / 'configs/qwen3vl-osworld-original-8gpu.yaml').read_text())
        source = yaml.safe_load((ROOT / 'SCALE-CUA/scalecua_rl/examples/training/configs/qw3vl_train_std.yaml').read_text())
        for key in ['advantage', 'loss', 'max_steps', 'save_interval', 'task_adv_norm']:
            self.assertEqual(local[key], source[key], key)
        self.assertEqual(local['train']['batch_size'], source['train']['batch_size'])
        self.assertEqual(local['train']['n'], source['train']['n'])
        self.assertTrue(local['dynamic_sampling']['enabled'])
        for key in ['images_to_keep', 'min_removal_threshold', 'max_total_len', 'max_turns']:
            self.assertEqual(local['task'][key], source['task'][key])
        self.assertEqual(local['actor']['optim'], source['actor']['optim'])
        plan = json.loads((ROOT / 'configs/original-rl-pipeline.json').read_text())
        self.assertFalse(plan['algorithm_changes_enabled'])
        self.assertEqual(plan['frontier_tasks'] * plan['frontier_runs_per_task'], 8196)

    def test_released_rl_has_a_distinct_pinned_identity(self):
        sft = checkpoint_spec(ROOT, 'osworld-sft'); rl = checkpoint_spec(ROOT, 'osworld-rl-reference')
        for field in ['model_revision', 'model_dir', 'served_model_name', 'state_file']:
            self.assertNotEqual(sft[field], rl[field])

    def test_selected_budget_changes_only_the_stopping_boundary(self):
        plan = json.loads((ROOT / 'configs/original-rl-pipeline.json').read_text())
        full = yaml.safe_load((ROOT / plan['full_training_config']).read_text())
        first = yaml.safe_load((ROOT / plan['training_config']).read_text())
        self.assertEqual(plan['automatic_update_budget'], 25)
        self.assertEqual(first.pop('max_steps'), 26)
        self.assertEqual(full.pop('max_steps'), 1000)
        self.assertEqual(first, full)

    def test_partial_stage_resume_does_not_stop_an_existing_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'runs/benchmarks/osworld-rl-reference__osworld-original-pipeline-v1'
            campaign.mkdir(parents=True)
            (campaign / 'campaign.json').write_text('{}')
            pool = root / 'runs/inference-pools'
            pool.mkdir()
            (pool / 'osworld-rl-reference-pipeline.json').write_text(json.dumps({'service_states': [str(i) for i in range(8)]}))
            events = []
            with patch.object(stages, 'ROOT', root), patch.object(stages, 'stop_inference', side_effect=lambda: events.append('stop')), \
                 patch.object(stages, 'call', side_effect=lambda *a: events.append('launch_missing')), \
                 patch.object(stages, 'maintain_campaigns', return_value=False), patch.object(stages, 'generate'), \
                 patch.object(stages, 'comparison_report'), patch.object(stages, 'start_pool') as start_pool, \
                 patch.object(stages, 'vm_acceptance') as acceptance:
                stages.evaluate('osworld-rl-reference')
            self.assertEqual(events, ['launch_missing', 'stop'])
            start_pool.assert_not_called()
            acceptance.assert_not_called()

    def test_secondary_environment_worker_requires_passing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / 'pipeline.json'
            cfg.write_text(json.dumps({'secondary_environment_worker': {
                'enabled': True, 'hostname': 'research-a-worker-0', 'rpc_smoke_report': 'absent.json'}}))
            with patch.object(stages, 'ROOT', root), patch.object(stages, 'CONFIG', cfg), \
                 patch.object(stages, 'DIRECTORY', root), patch.object(stages, 'remote') as remote:
                self.assertEqual(stages.secondary_worker_start(), 0)
            remote.assert_not_called()
            selection = json.loads((root / 'secondary-worker-selection.json').read_text())
            self.assertFalse(selection['enabled'])
            self.assertEqual(selection['gpu_allocation'], 0)

    def test_final_empty_batch_preserves_preceding_updates(self):
        source = '''def run(batches, actor, ray):
    global_step = 1
    max_steps = len(batches) + 1
    save_interval = 25
    while global_step < max_steps:
        data = batches[global_step - 1]
        if not data:
            global_step += 1
            continue
        actor.update(data)
        if global_step % save_interval == 0:
            actor.save_checkpoint(global_step)
        ray.get(actor.step_finished())
        global_step += 1
'''
        namespace = {}
        exec(add_final_save(source), namespace)
        updates, saves = [], []
        actor = types.SimpleNamespace(update=updates.append, save_checkpoint=saves.append, step_finished=lambda: None)
        namespace['run']([1, 2, None], actor, types.SimpleNamespace(get=lambda x: x))
        self.assertEqual(updates, [1, 2])
        self.assertEqual(saves, [3])
        actual = (ROOT / 'SCALE-CUA/scalecua_rl/examples/training/async_trainer.py').read_text()
        compile(add_final_save(actual), '<adapted trainer>', 'exec')


if __name__ == '__main__':
    unittest.main()
