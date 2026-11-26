import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ais_dataset_screen import DEFAULT_BATCHES, ROOT, checked_score, coverage, expected_protocol, merge, plan, restore, summarize
from ais_vm_check import save
from build_overnight_screen import MANIFEST
from overnight_prompts import export


class DatasetScreenTest(unittest.TestCase):
    def test_targeted_queue_separates_new_tasks_from_extra_trials(self):
        result = subprocess.run(['bash', str(ROOT / 'AIS_ENTRY.txt'), 'screen-expand', '--plan'],
                                capture_output=True, text=True, check=True)
        batch = json.loads(result.stdout)
        self.assertTrue(batch['check_initial_state'])
        self.assertEqual((batch['ranks'], batch['desktops_per_rank'], batch['trials']), (8, 2, 32))
        evidence = MANIFEST.parent / 'targeted-20261126-review.jsonl'
        rows = [json.loads(line) for line in evidence.read_text().splitlines()]
        self.assertEqual([[r['task_index']] for r in rows], batch['shards'])
        previous = set()
        for path in (*DEFAULT_BATCHES, MANIFEST, MANIFEST.parent / 'expansion-20261126.json',
                     MANIFEST.parent / 'wide-20261126.json', ROOT / 'configs/original-pilot/train-indices.json'):
            previous.update(json.loads(path.read_text()))
        self.assertEqual(sum(r['purpose'] == 'first_four' for r in rows), 5)
        for row in rows:
            if row['purpose'] == 'first_four':
                self.assertNotIn(row['task_index'], previous)
            else:
                self.assertEqual(row['purpose'], 'additional_four')
                self.assertIn(row['task_index'], previous)
                self.assertEqual(len(row['prior_raw_rewards']), 4)
                self.assertTrue(all(0 <= reward < 1 for reward in row['prior_raw_rewards']))
            controls = row['offline_controls']
            self.assertEqual((controls['untouched'], controls['correct']), (0, 1))
            wrong = [v for k, v in controls.items() if k not in ('untouched', 'correct')]
            self.assertTrue(wrong and all(0 <= v < 1 for v in wrong))

    def test_wide_entry_screens_fresh_training_variants(self):
        result = subprocess.run(['bash', str(ROOT / 'AIS_ENTRY.txt'), 'screen-wide', '--plan'],
                                capture_output=True, text=True, check=True)
        batch = json.loads(result.stdout)
        self.assertTrue(batch['check_initial_state'])
        self.assertEqual((batch['ranks'], batch['desktops_per_rank'], batch['trials']), (8, 2, 64))
        self.assertEqual([len(shard) for shard in batch['shards']], [2] * 8)
        tasks = {task for shard in batch['shards'] for task in shard}
        self.assertEqual(len(tasks), 16)
        for path in (*DEFAULT_BATCHES, MANIFEST, MANIFEST.parent / 'expansion-20261126.json',
                     ROOT / 'configs/original-pilot/train-indices.json'):
            self.assertTrue(tasks.isdisjoint(json.loads(path.read_text())), path)
        families = {task.rsplit('_traj_verify_', 1)[0].rsplit('_', 1)[-1] for task in tasks}
        self.assertEqual(len(families), 11)
        splits = json.loads((ROOT / 'configs/research-splits/manifest.json').read_text())['splits']
        for split in ('dev', 'test'):
            self.assertTrue(families.isdisjoint(splits[split]['base_ids']))

    def test_expansion_entry_checks_untouched_state_on_eight_new_tasks(self):
        result = subprocess.run(['bash', str(ROOT / 'AIS_ENTRY.txt'), 'screen-next', '--plan'],
                                capture_output=True, text=True, check=True)
        batch = json.loads(result.stdout)
        self.assertTrue(batch['check_initial_state'])
        self.assertEqual([len(shard) for shard in batch['shards']], [1] * 8)
        self.assertEqual(batch['trials'], 32)
        old = set(json.loads(MANIFEST.read_text()))
        old.update(json.loads((ROOT / 'configs/original-pilot/train-indices.json').read_text()))
        for previous in DEFAULT_BATCHES:
            old.update(json.loads(previous.read_text()))
        self.assertTrue(all(shard[0] not in old for shard in batch['shards']))

    def test_overnight_pool_is_larger_than_target_and_rejects_nonzero_controls(self):
        candidates = json.loads(MANIFEST.read_text())
        self.assertGreater(len(candidates), 256)
        self.assertEqual(plan(MANIFEST, 2)['trials'], 4 * len(candidates))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / '23599999'
            def records(filename):
                return [json.loads(line) for line in (directory / filename).read_text().splitlines()]
            for state, reward in [('passed', 0), ('nonzero_initial_reward', 1)]:
                rewards = [1, 0, 0, 1] if reward == 0 else [None] * 4
                results = {i: {'initial_state': {'task_index': i, 'status': state, 'reward': reward, 'session_id': '1'},
                               'raw_rewards': rewards} for i in candidates}
                export(directory, directory, final=results)
                scored = records('prompts-scored.jsonl')
                status = json.loads((directory / 'prompt-status.json').read_text())
                self.assertEqual(len(scored), len({r['prompt'].strip().casefold() for r in scored}))
                self.assertTrue(all(r['valid_trials'] == 4 for r in scored))
                self.assertEqual(status['scored_shortfall'], 256 - len(scored))
                if state == 'passed':
                    self.assertEqual(len(scored), 256)
                    self.assertGreater(len(records('prompt-pool.jsonl')), 256)
                    self.assertLess(len(records('prompts-reviewed.jsonl')), 256)
                else:
                    self.assertGreater(status['scored_shortfall'], 0)
                    self.assertFalse(set(candidates).intersection(r['task_index'] for r in scored))

    def test_batch_is_disjoint_and_train_only(self):
        data = plan(None, 2)
        original = plan(DEFAULT_BATCHES[0], 2)
        self.assertEqual(data['tasks'], 48)
        for first, full in zip(original['shards'], data['shards']):
            self.assertEqual(first, full[:len(first)])
        tasks = [i for shard in data['shards'] for i in shard]
        self.assertEqual(len(set(tasks)), data['tasks'])
        self.assertEqual((len(data['shards']), data['trials']), (8, len(tasks) * 4))
        pilot = json.loads((ROOT / 'configs/original-pilot/train-indices.json').read_text())
        self.assertFalse(set(tasks).intersection(pilot))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'indices.json'
            for invalid in ([tasks[0], tasks[0]], ['holdout'], []):
                save(p, invalid)
                with self.assertRaises(ValueError):
                    plan(p, 2)

    def test_reuse_and_partial_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            old = out / 'original'
            old.mkdir()
            indices = old / 'indices.json'
            save(indices, ['fixture'])
            protocol = expected_protocol(indices)
            slot = old / 'episodes/fixture/1'
            slot.mkdir(parents=True)
            save(old / 'episodes/protocol.json', protocol)
            save(old / 'cohort.json', {'runtime': 'test', 'tasks': ['fixture'], 'batch_sha256': 'old'})
            row = {'task_index': 'fixture', 'result': 0, 'raw_reward': 0.4, 'run_number': 1,
                   'model': protocol['services'][0]['served_model_name'], 'model_revision': protocol['model_revision'],
                   'attempt': 1, 'terminal_status': 'completed'}
            save(slot / 'result.json', row)
            save(slot / 'attempt-01.json', dict(row, result=row, status='completed', model_calls=1, session_id='0',
                                                terminal={'finish': True, 'status': 'completed', 'reward': 0.4}))
            (slot / 'trace-01.jsonl').write_text('{"turn":0,"message":{"role":"assistant"}}\n')
            new = out / 'resumed'
            tasks = ['fixture', 'next']
            extended = out / 'extended.json'
            save(extended, tasks)
            current = expected_protocol(extended)
            cohort = {'runtime': 'test', 'tasks': tasks, 'batch_sha256': 'extended'}
            with self.assertRaises(ValueError):
                restore(old, new, tasks, current, dict(cohort, runtime='other'))
            with self.assertRaises(ValueError):
                restore(old, new, tasks, dict(current, temperature=99), cohort)
            self.assertEqual(restore(old, new, tasks, current, cohort), 1)
            result = summarize(coverage(new, ['fixture'], protocol))
            self.assertEqual(result['scored'], 1)
            self.assertEqual(result['unscored'], 3)
            self.assertEqual(result['results']['fixture']['raw_rewards'], [0.4, None, None, None])
            self.assertIsNone(result['results']['fixture']['full_credit_rate'])
            with self.assertRaises(ValueError):
                checked_score(new / 'episodes/fixture/1', 'other', 1, protocol)

    def test_merge_eight_shards_and_detect_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            batch = plan(None, 2)
            for rank, tasks in enumerate(batch['shards']):
                lane = directory / f'rank-{rank}'
                lane.mkdir()
                save(lane / 'cohort.json', {'batch_sha256': batch['batch_sha256'], 'rank': rank, 'tasks': tasks})
                save(lane / 'coverage.json', summarize(coverage(lane, tasks, {})))
            result = merge(directory, batch, {})
            self.assertEqual(result['unscored'], batch['trials'])
            self.assertEqual(result['tasks_complete'], 0)
            save(directory / 'rank-7/coverage.json', {'results': {}})
            with self.assertRaises(ValueError):
                merge(directory, batch, {})


if __name__ == '__main__':
    unittest.main()
