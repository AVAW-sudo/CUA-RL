import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from ais_vm_check import save
from review_screen_job import published_selection, upstream_state, verify_trials


class NightReviewTest(unittest.TestCase):
    def test_shared_mount_can_briefly_show_an_empty_final_list(self):
        rows = [{'task_index': 'task'}]
        with patch('review_screen_job.records', side_effect=[[], rows]), patch('review_screen_job.time.sleep'):
            self.assertEqual(published_selection(Path('run'), 1), rows)

    def test_wait_for_final_summary_even_after_last_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            save(directory / 'progress.json', {'available': 0, 'in_flight': 1})
            self.assertIsNone(upstream_state(directory))
            (directory / 'summary.json').write_text('')
            self.assertIsNone(upstream_state(directory))
            for terminal in ('candidates_exhausted', 'quotas_met'):
                save(directory / 'summary.json', {'status': terminal})
                self.assertEqual(upstream_state(directory), terminal)

    def test_failed_rank_triggers_partial_review(self):
        with tempfile.TemporaryDirectory() as temp:
            lane = Path(temp) / 'rank-1'
            lane.mkdir()
            save(lane / 'status.json', {'status': 'failed'})
            self.assertEqual(upstream_state(lane.parent), 'upstream_failed')

    def test_missing_initial_receipt_cannot_be_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'episodes/case'
            folder.mkdir(parents=True)
            save(folder.parent / 'protocol.json', {})
            with patch('review_screen_job.expected_protocol'), patch('review_screen_job.require_equivalent_protocol'):
                with self.assertRaises(FileNotFoundError):
                    verify_trials('case', folder, [1, 0, 0, 1], require_initial=True)
                save(folder / 'initial-state.json', dict(task_index='case', session_id='1',
                                                         status='nonzero_initial_reward', reward=0.5))
                self.assertEqual(verify_trials('case', folder, [None]*4), 'initial_state_nonzero_initial_reward')
