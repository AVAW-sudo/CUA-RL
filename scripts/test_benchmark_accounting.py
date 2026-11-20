"""Protect benchmark denominators and the no-success-selective-rerun policy."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evaluate_sft import summarize
from run_benchmark import aggregate, needs_run, official_tasks, select_queue, fingerprint


class AccountingTests(unittest.TestCase):
    def test_missing_and_errors_prevent_full_metric(self):
        rows = [{'status': 'completed', 'score': 1}, {'status': 'completed', 'score': 0},
                {'status': 'infrastructure_error', 'score': None}, {'status': 'pending', 'score': None}]
        result = aggregate(rows)
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['completed'], 2)
        self.assertEqual(result['successes'], 1)
        self.assertEqual(result['infrastructure_or_verifier_errors'], 1)
        self.assertIsNone(result['success_rate'])
        self.assertIsNone(result['mean_score'])

    def test_partial_credit_and_success_rate_differ(self):
        result = aggregate([{'status': 'completed', 'score': v} for v in [1., .5, 0.]])
        self.assertEqual(result['success_rate'], 1 / 3)
        self.assertEqual(result['mean_score'], .5)

    def test_resume_never_retries_valid_failure(self):
        for retry in [False, True]:
            self.assertFalse(needs_run({'status': 'completed', 'score': 0}, retry))
        self.assertFalse(needs_run({'status': 'infrastructure_error'}, False))
        self.assertTrue(needs_run({'status': 'infrastructure_error'}, True))

    def test_scoped_recovery_keeps_completed_failures_and_unselected_errors(self):
        tasks = ['valid_zero', 'boot_error', 'verifier_error']
        rows = {'valid_zero': {'status': 'completed', 'score': 0},
                'boot_error': {'status': 'infrastructure_error', 'score': None},
                'verifier_error': {'status': 'infrastructure_error', 'score': None}}
        self.assertEqual(select_queue(tasks, rows, True, ['valid_zero', 'boot_error']), ['boot_error'])
        self.assertEqual(aggregate(list(rows.values()))['total'], 3)
        with self.assertRaises(ValueError):
            select_queue(tasks, rows, True, ['unknown'])

    def test_verifier_nan_empty_and_error_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(action_space='a', observation_type='o', model='m')
            directory = root / 'results/a/o/m/domain/id'
            directory.mkdir(parents=True)
            result = directory / 'result.txt'
            for value in ['nan', 'inf', '-1', '2']:
                result.write_text(value)
                row = summarize(root, args, [('domain', 'id')])['tasks'][0]
                self.assertEqual(row['status'], 'infrastructure_error')
                self.assertIsNone(row['score'])
            result.write_text('')
            self.assertEqual(summarize(root, args, [('domain', 'id')])['missing_count'], 1)
            result.write_text('0')
            self.assertEqual(summarize(root, args, [('domain', 'id')])['completed_count'], 1)
            result.write_text('1')
            (directory / 'traj.jsonl').write_text(json.dumps({'Error': 'Verifier failed'}) + '\n')
            self.assertEqual(summarize(root, args, [('domain', 'id')])['infrastructure_error_count'], 1)

    def test_official_denominators(self):
        self.assertEqual(len(official_tasks('osworld')[0]), 361)
        self.assertEqual(len(official_tasks('scienceboard')[0]), 169)

    def test_resume_verifies_original_runtime_file_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'old.py').write_text('original')
            pinned = fingerprint(root, ['old.py'])
            (root / 'new.py').write_text('new adapter')
            self.assertEqual(fingerprint(root, pinned), pinned)
            (root / 'old.py').write_text('changed')
            self.assertNotEqual(fingerprint(root, pinned), pinned)
            (root / 'old.py').unlink()
            with self.assertRaises(FileNotFoundError):
                fingerprint(root, pinned)


if __name__ == '__main__':
    unittest.main()
