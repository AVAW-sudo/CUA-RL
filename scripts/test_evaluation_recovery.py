import json
from pathlib import Path
import tempfile
import unittest

from evaluation_recovery import classify_attempt, retry_eligible


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / '001'
        self.path.mkdir()
        self.log = self.path.parent / '001.log'
        self.attempt = {'attempt_dir': str(self.path), 'log': str(self.log)}

    def test_zero_action_boot_failure_can_retry_but_is_bounded(self):
        self.log.write_text('[Init Fail] task: VM not ready after 600s')
        row = {'status': 'infrastructure_error', 'attempts': [self.attempt]}
        self.assertTrue(retry_eligible(row))
        row['attempts'] *= 3
        self.assertFalse(retry_eligible(row))

    def test_acted_verifier_failure_is_never_resampled(self):
        self.log.write_text('Setup step failed during verifier postconfig')
        (self.path / 'summary.json').write_text(json.dumps({'tasks': [{'action_records': 50}]}))
        self.assertFalse(classify_attempt(self.attempt)['retryable'])

    def test_missing_summary_with_a_trajectory_is_not_zero_action(self):
        self.log.write_text('[Init Fail] earlier task diagnostic')
        (self.path / 'traj.jsonl').write_text('{"action":"DONE"}\n')
        self.assertEqual(classify_attempt(self.attempt)['action_records'], 1)
        self.assertFalse(classify_attempt(self.attempt)['retryable'])

    def test_unknown_missing_result_is_not_evidence_of_no_sampling(self):
        self.assertFalse(classify_attempt(self.attempt)['retryable'])

    def test_upstream_error_only_row_is_not_a_model_action(self):
        self.log.write_text('Setup HTTP failure')
        (self.path / 'traj.jsonl').write_text('{"Error":"Setup HTTP failure"}\n')
        self.assertTrue(classify_attempt(self.attempt)['retryable'])

    def test_a_valid_zero_score_is_not_retried(self):
        self.log.write_text('[Init Fail] historical log text')
        self.assertFalse(retry_eligible({'status': 'completed', 'score': 0, 'attempts': [self.attempt]}))


if __name__ == '__main__':
    unittest.main()
