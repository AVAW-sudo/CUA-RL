import unittest

from profile_evaluation_logs import parse_log


class ProfileTests(unittest.TestCase):
    def test_does_not_double_count_runner_and_environment_action_logs(self):
        text = '''[2026-11-20 00:00:00,000 INFO desktop_env/1-worker] Initializing...
[2026-11-20 00:01:00,000 INFO desktop_env/1-worker] Environment setup complete.
[2026-11-20 00:02:00,000 INFO lib_run_single/1-worker] Step 1: click
[2026-11-20 00:02:00,000 INFO desktop_env/1-worker] Step 1 in trajectory 0 with action: click
[2026-11-20 00:02:04,000 INFO lib_run_single/1-worker] Reward: 0.00
[2026-11-20 00:02:07,000 INFO desktop_env/1-worker] Step 2 in trajectory 0 with action: click
[2026-11-20 00:02:12,000 INFO lib_run_single/1-worker] Reward: 0.00'''
        result = parse_log(text)
        self.assertEqual(result['environment_actions'], [4, 5])
        self.assertEqual(result['between_action_gaps'], [3])
        self.assertEqual(result['boot_and_setup_seconds'], 60)
        self.assertEqual(result['setup_to_first_action_seconds'], 60)
        self.assertEqual(result['unmatched_actions'], 0)

    def test_incomplete_action_is_reported_and_not_used_as_model_gap(self):
        text = '''[2026-11-20 00:00:00,000 INFO desktop_env/1-worker] Step 1 in trajectory 0 with action: click
[2026-11-20 00:00:04,000 INFO desktop_env/1-worker] Step 2 in trajectory 0 with action: click
[2026-11-20 00:00:09,000 INFO lib_run_single/1-worker] Reward: 0.00'''
        result = parse_log(text)
        self.assertEqual(result['environment_actions'], [5])
        self.assertEqual(result['between_action_gaps'], [])
        self.assertEqual(result['unmatched_actions'], 1)


if __name__ == '__main__':
    unittest.main()
