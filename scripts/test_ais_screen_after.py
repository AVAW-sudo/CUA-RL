from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ais_full_screen_pool import extend, load_inventory
from ais_screen_after import full_pool_source, ready, sampling_history, skip_filled_retries
from ais_screen_queue import QUOTAS, ROOT, ScreenQueue
from ais_frontier_shards import read
from ais_vm_check import save
from test_ais_screen_queue import baseline, candidate, outcome


class UpstreamDependencyTest(unittest.TestCase):
    def test_original_entry_tracks_its_own_latest_usable_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job, parent, status in [('100', None, 'candidates_exhausted'), ('101', '100', 'candidates_exhausted'),
                                        ('102', '101', 'failed'), ('103', '101', 'running'), ('104', '1000', 'running')]:
                folder = root / job
                (folder / 'rank-0').mkdir(parents=True)
                save(folder / 'campaign.json', {'upstream_job': parent})
                save(folder / 'rank-0/status.json', {'status': status})
                if status == 'candidates_exhausted':
                    save(folder / 'summary.json', {'status': status})
            with patch('ais_screen_after.CONFIG', root):
                save(root / 'full-pool.json', {'source_job': '100'})
                self.assertEqual(full_pool_source(root / '100'), (root / '101', True))
                self.assertEqual(full_pool_source(root / '100', '103'), (root / '101', True))
                self.assertEqual(full_pool_source(root / '104'), (root / '104', False))
                self.assertEqual(sampling_history(root / '101', '103'), [root / '102', root / '101'])

    def test_wait_requires_final_export_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            rank = job / 'rank-0'
            rank.mkdir()
            save(rank / 'status.json', {'status': 'shard_complete'})
            self.assertFalse(ready(job, wait=False))
            save(rank / 'status.json', {'status': 'failed'})
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                ready(job, wait=True)
            save(rank / 'status.json', {'status': 'shard_complete'})
            def publish(_):
                save(job / 'summary.json', {'status': 'candidates_exhausted'})
            with patch('ais_screen_after.time.sleep', side_effect=publish) as sleep:
                self.assertTrue(ready(job, wait=True))
                sleep.assert_called_once()


class FullPoolTest(unittest.TestCase):
    def test_main_followup_skips_only_filled_roles_without_a_weak_seed(self):
        rows = [dict(task_index=i, role=role, seed_score=seed) for i, role, seed in
                [('a', 'breadth', 1), ('b', 'breadth', None), ('c', 'breadth', 0),
                 ('d', 'main', 0), ('e', 'component', 1)]]
        decisions = [dict(task_index=r['task_index'], decision='retry_missing', missing_slots=2) for r in rows]
        queue, saved = skip_filled_retries(rows, decisions, dict(breadth=64, main=49, component=63))
        self.assertEqual([r['task_index'] for r in queue], ['c', 'd', 'e'])
        self.assertEqual(saved, 4)
        self.assertEqual([d['decision'] for d in decisions].count('quota_met_after_cpu'), 2)

    def test_pinned_inventory_partitions_official_training_split(self):
        train = read(ROOT / 'configs/research-splits/train-indices.json')
        data = load_inventory(train, read(ROOT / 'configs/sources.lock.json')['training_data']['revision'])
        self.assertGreater(len(data['candidates']), 400)
        self.assertTrue(any(r['shared_setup_inputs'] and r['role'] == 'main' for r in data['candidates'].values()))
        with self.assertRaisesRegex(ValueError, 'account for each'):
            load_inventory(train[:-1], data['revision'])

    def test_expand_checks_hash_holdout_and_existing_ids(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'data/ScaleCUA/osworld/rl_tasks/os'
            folder.mkdir(parents=True)
            save(folder / '1.json', {'id': '1', 'instruction': 'Adjust brightness'})
            evidence = dict(role='main', seed_score=0, shared_setup_inputs=True,
                            sha256=hashlib.sha256((folder / '1.json').read_bytes()).hexdigest())
            inv = {'excluded': {'previously_screened': ['os_0']}, 'candidates': {'os_1': evidence, 'os_0': evidence}}
            rows = []
            args = (inv, rows, [], [], {'os_0'}, {'os_1': 'os', 'os_0': 'os'})
            with self.assertRaisesRegex(ValueError, 'held-out'):
                extend(*args, {'adjust brightness'}, root, {})
            report = extend(*args, set(), root, {})
            self.assertEqual(report['first_time_by_role'], {'main': 1})
            self.assertIsNone(rows[0]['reviewed_skill'])
            self.assertTrue(rows[0]['shared_setup_inputs'])
            save(folder / '1.json', {'id': '1', 'instruction': 'different'})
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                extend(*args, set(), root, {})

    def test_full_class_keeps_sampling_until_global_quota_met(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [dict(candidate('more-breadth', 'breadth'), full_pool_candidate=True), candidate('last-main')]
            old = [baseline(role, j) for role, cap in QUOTAS.items() for j in range(cap - int(role == 'main'))]
            q = ScreenQueue(Path(tmp), {r['task_index'] for r in rows + old}, fixed_inputs=(rows, old))
            first = q.claim(0)
            self.assertEqual(first, rows[0])
            save(q.tasks / first['task_index'] / 'outcome.json', outcome(first))
            final_main = q.claim(1)
            save(q.tasks / final_main['task_index'] / 'outcome.json', outcome(final_main))
            self.assertIsNone(q.next_task(2))
            self.assertEqual(q.publish(final=True)['strict_total'], 256)
