import json
from pathlib import Path
import tempfile
import unittest

from ais_dataset_screen import DEFAULT_BATCHES, ROOT, checked_score, coverage, expected_protocol, merge, plan, restore, summarize
from ais_vm_check import save


class DatasetScreenTest(unittest.TestCase):
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
