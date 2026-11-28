import json
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from ais_screen_queue import (ARCHIVE_REVIEW, QUOTAS, REVIEWED, ROOT, ROSTER, ScreenQueue, check_manifest, complete_task,
                              earlier_jobs, eligible, previous_batches, records)
from ais_vm_check import save
from overnight_prompts import write_lines


def candidate(i, role='main'):
    return {'task_index': i, 'prompt': f'Exercise {i}', 'role': role, 'purpose': 'first_four',
            'reviewed_skill': 'Use the requested file', 'offline_controls': {'untouched': 0, 'correct': 1, 'wrong': 0}}


def outcome(row, finished=1):
    return {'task_index': row['task_index'], 'candidate': row, 'finished': finished,
            'initial_state': {'status': 'passed'}, 'raw_rewards': [0, 1, 1, 0]}


def baseline(role, n):
    i = f'old-{role}-{n}'
    return dict(task_index=i, prompt=f'Archived {i}', selection_role=role, valid_trials=4,
                skill_evidence='reviewed', judge_controls_reviewed=True)


def worker(root, training, rank, rendezvous, completed):
    queue = ScreenQueue(root, training, root / 'roster', root / 'reviewed')
    row = queue.claim(rank)
    rendezvous.wait(timeout=30)  # All eight ranks own a task before any terminal scores arrive.
    while row:
        i = row['task_index']
        save(queue.tasks / i / 'outcome.json', outcome(row))
        completed.append((rank, i))
        row = queue.claim(rank)


class SharedQueueTest(unittest.TestCase):
    def test_eight_concurrent_ranks_claim_each_task_once_and_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [candidate(f'task-{i}') for i in range(32)]
            training = {r['task_index'] for r in rows}
            write_lines(root / 'roster', rows)
            write_lines(root / 'reviewed', [])
            ctx = multiprocessing.get_context('spawn')
            with ctx.Manager() as manager:
                completed = manager.list()
                rendezvous = ctx.Barrier(8)
                processes = [ctx.Process(target=worker, args=(root, training, rank, rendezvous, completed))
                             for rank in range(8)]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=45)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(len(completed), 32)
                self.assertEqual({rank for rank, _ in completed}, set(range(8)))
                self.assertEqual({i for _, i in completed}, training)
            queue = ScreenQueue(root, training, root / 'roster', root / 'reviewed')
            summary = queue.publish(final=True)
            self.assertEqual((summary['status'], summary['strict_total']), ('candidates_exhausted', 32))
            self.assertEqual(json.loads((root / 'progress.json').read_text())['status'], summary['status'])

    def test_live_refill_and_role_stop_with_inflight_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = [baseline(role, n) for role, cap in QUOTAS.items()
                   for n in range(cap - int(role in ('main', 'component')))]
            training = {r['task_index'] for r in old} | {'one', 'two', 'three', 'new'}
            rows = [candidate(i) for i in ('one', 'two', 'three')]
            write_lines(root / 'roster', rows)
            write_lines(root / 'reviewed', old)
            queue = ScreenQueue(root, training, root / 'roster', root / 'reviewed')
            first, second = queue.claim(1), queue.claim(2)
            for n, r in enumerate([first, second]):
                save(queue.tasks / r['task_index'] / 'outcome.json', outcome(r, n))
            self.assertIsNone(queue.claim(3))  # One slot left, two finished, no more main tasks.
            rows.append(candidate('new', 'component'))
            write_lines(root / 'roster', rows)
            self.assertIsNone(queue.publish(final=True))
            r = queue.claim(1)
            self.assertEqual(r['task_index'], 'new')
            save(queue.tasks / 'new/outcome.json', outcome(r))
            final = queue.publish(final=True)
            self.assertEqual((final['status'], final['strict_total']), ('quotas_met', 256))
            self.assertFalse((queue.tasks / 'three').exists())
            self.assertEqual(len(records(root / 'prompts-reviewed.jsonl')), 256)

    def test_unreviewed_task_requires_controls_and_can_be_reviewed_during_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = candidate('one')
            unreviewed = dict(row, reviewed_skill=None)
            write_lines(root / 'roster', [unreviewed])
            write_lines(root / 'reviewed', [])
            q = ScreenQueue(root, {'one'}, root / 'roster', root / 'reviewed')
            selected = q.claim(0)
            save(q.tasks / 'one/outcome.json', outcome(selected))
            self.assertEqual(q.publish()['strict_total'], 0)
            write_lines(root / 'roster', [row])
            self.assertEqual(q.publish()['strict_total'], 1)
            self.assertFalse(eligible(row, dict(outcome(row), raw_rewards=[1, 1, 1, None])))
            self.assertFalse(eligible(row, dict(outcome(row), raw_rewards=[0, 0, 0, 0])))
            self.assertFalse(eligible(row, dict(outcome(row), initial_state={'status': 'unavailable'})))

    def test_idle_rank_takes_live_refill_while_another_rank_is_busy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first, second = candidate('slow'), candidate('refill')
            roster = root / 'roster'
            write_lines(roster, [first])
            write_lines(root / 'reviewed', [])
            q = ScreenQueue(root, {'slow', 'refill'}, roster, root / 'reviewed')
            self.assertEqual(q.next_task(0), first)
            with patch('ais_screen_queue.time.sleep', side_effect=lambda _: write_lines(roster, [first, second])):
                self.assertEqual(q.next_task(1), second)
            rank = root / 'rank-0'
            rank.mkdir()
            save(rank / 'status.json', {'status': 'failed', 'error': 'lost service'})
            with self.assertRaisesRegex(RuntimeError, 'lost service'):
                q.next_task(2)

    def test_entry_and_frozen_sampling_history(self):
        plan = json.loads(subprocess.check_output(['bash', str(ROOT / 'AIS_ENTRY.txt'), 'screen-expand', '--plan']))
        self.assertEqual((plan['ranks'], plan['desktops_per_rank'], plan['scheduling']), (8, 2, 'shared_queue'))
        self.assertTrue(plan['check_initial_state'])
        self.assertGreaterEqual(plan['tasks'], 64)
        self.assertEqual(plan['trials'], plan['tasks'] * 4)
        archive = records(ARCHIVE_REVIEW)
        gpu_ids = {r['task_index'] for r in records(ROSTER)}
        baseline_ids = {r['task_index'] for r in records(REVIEWED)}
        archived_ids = {r['task_index'] for r in archive}
        self.assertEqual(len(archived_ids), len(archive))
        self.assertEqual(sum(plan['archived_candidates_by_next_step'].values()),
                         len(archived_ids - gpu_ids - baseline_ids))
        self.assertEqual(sum(plan['shortfall_at_start'].values()), sum(QUOTAS.values()) - len(baseline_ids))
        for row in archive:
            scores = row['raw_rewards']
            self.assertEqual(len(scores), 4)
            self.assertTrue(all(type(v) in (int, float) and 0 <= v <= 1 for v in scores))
            expected_step = ('review_breadth' if row['role'] == 'breadth' else
                             'check_solvability_before_resampling' if scores.count(1) == 0 else 'review_mixed')
            self.assertEqual(row['next_step'], expected_step)
            if expected_step == 'review_mixed':
                self.assertLess(scores.count(1), len(scores))
        previous = previous_batches()
        for row in records(ROSTER):
            self.assertEqual(row['purpose'] == 'additional_four', row['task_index'] in previous)
        with self.assertRaises(ValueError):
            check_manifest([candidate('old')], {'old'}, {'old'})

    def test_supplemental_trials_resume_only_the_same_second_round(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first = candidate('one')
            prior_scores = [0, 0, 0.5, 0]
            row = dict(first, purpose='additional_four', prior_followup=True, prior_job='100',
                       prior_raw_rewards=prior_scores)
            for job, purpose, prior in [('100', 'first_four', None), ('101', 'additional_four', '100'),
                                        ('102', 'additional_four', '99')]:
                source = root / job / 'tasks/one'
                (source / 'episodes').mkdir(parents=True)
                save(source / 'episodes/protocol.json', {})
                save(source / 'cohort.json', dict(purpose=purpose, prior_job=prior))
            original = root / '100/tasks/one/outcome.json'
            save(original, dict(outcome(first), raw_rewards=prior_scores))
            folder = root / '103/tasks/one'
            (folder / 'episodes').mkdir(parents=True)
            history = [root / job for job in ['102', '101', '100']]
            collect = Mock(side_effect=lambda *a, **kw: save(folder / 'episodes/protocol.json', {}))
            with (patch('ais_screen_queue.expected_protocol', return_value={}),
                  patch('ais_screen_queue.require_equivalent_protocol'),
                  patch('ais_screen_queue.coverage', return_value={'one': outcome(row)}),
                  patch('ais_screen_queue.restore', return_value=2) as restore):
                complete_task(folder, row, history, {}, collect)
                self.assertEqual([c.args[0] for c in restore.call_args_list], [root / '101/tasks/one'])
                collect.assert_called_once()
                self.assertEqual(json.loads((folder / 'outcome.json').read_text())['reused_slots'], 2)
                save(original, dict(outcome(first), raw_rewards=[0, 0, 0, 0]))
                with self.assertRaisesRegex(ValueError, 'original four'):
                    complete_task(folder, row, history, {}, collect)

    def test_complete_history_skips_gui_and_partial_reuse_collects(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('120', '130', '140', 'notebook-ok', 'notebook-open'):
                p = root / name
                p.mkdir()
                save(p / 'campaign.json', {'runtime': {}})
            save(root / 'notebook-ok/summary.json', {'status': 'notebook_complete'})
            self.assertEqual([p.name for p in earlier_jobs(root, '140', '120')], ['120', 'notebook-ok'])
            prior = root / '120/tasks/one/episodes'
            prior.mkdir(parents=True)
            save(prior / 'protocol.json', {})
            row = candidate('one')
            def restore_slots(source, dest, ids, protocol, cohort, *, merge_existing):
                self.assertTrue(merge_existing)
                (dest / 'episodes/one').mkdir(parents=True)
                save(dest / 'episodes/one/initial-state.json', {'status': 'passed'})
                return slots
            for slots in (4, 2):
                folder = root / f'new-{slots}'
                folder.mkdir()
                collect = Mock(side_effect=lambda *a, **kw: save(folder / 'episodes/protocol.json', {}))
                with (patch('ais_screen_queue.expected_protocol', return_value={}),
                      patch('ais_screen_queue.require_equivalent_protocol'),
                      patch('ais_screen_queue.coverage', return_value={'one': outcome(row)}),
                      patch('ais_screen_queue.restore', side_effect=restore_slots)):
                    complete_task(folder, row, [root / '120'], {}, collect)
                self.assertEqual(collect.call_count, int(slots < 4))
                self.assertEqual(json.loads((folder / 'outcome.json').read_text())['reused_slots'], slots)


if __name__ == '__main__':
    unittest.main()
