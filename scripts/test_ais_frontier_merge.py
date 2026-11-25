"""Exercise eight-shard publication using disposable records."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ais_frontier import verify_and_publish, wait_for_shards
from ais_frontier_shards import merge_campaign, ROOT


class FrontierMergeTest(unittest.TestCase):
    def test_master_waits_for_late_shard_and_reports_failed_peer(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for rank in range(8):
                path = directory / f'rank-{rank}/status.json'
                path.parent.mkdir()
                if rank != 6:
                    path.write_text('{"status":"shard_complete"}')
            peer = directory / 'rank-6/status.json'
            updates = iter(['collecting', 'shard_complete'])
            with patch('ais_frontier.time.sleep', side_effect=lambda _: peer.write_text(json.dumps({'status': next(updates)}))) as sleep:
                wait_for_shards(directory)
            self.assertEqual(sleep.call_count, 2)
            peer.write_text('{"status":"failed","error":"collector exited"}')
            with self.assertRaisesRegex(RuntimeError, 'rank 6 failed: collector exited'):
                wait_for_shards(directory)

    def test_coverage_protocol_and_matching_attempt_required(self):
        plan = ROOT / 'runs/ais-reproduction/frontier-8pod-plan'
        selection = ROOT / 'configs/original-pilot/train-indices.json'
        protocol_file = ROOT / 'runs/frontier-pilot-initialization/protocol.json'
        protocol = json.loads(protocol_file.read_text())
        primary = protocol['services'][0]
        task_name = protocol['task_name']

        def put(path, value):
            path.write_text(json.dumps(value))

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for rank in range(8):
                lane = directory / f'rank-{rank}'
                episodes = lane / 'episodes'
                episodes.mkdir(parents=True)
                shard = plan / f'rank-{rank}.json'
                tasks = json.loads(shard.read_text())
                shard_protocol = dict(protocol, tasks=8, services=[primary],
                                      indices_sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
                put(episodes / 'protocol.json', shard_protocol)
                put(lane / 'status.json', {'status': 'shard_complete'})
                put(lane / 'rates.json', {task_name: {task: 0.5 for task in tasks}})
                rows = []
                for task in tasks:
                    for run_number in range(1, 5):
                        run = episodes / task / str(run_number)
                        run.mkdir(parents=True)
                        reward = run_number % 2
                        row = {'task_index': task, 'run_number': run_number, 'attempt': 1, 'raw_reward': reward,
                               'result': reward, 'terminal_status': 'completed', 'model': primary['served_model_name'],
                               'model_revision': primary['model_revision']}
                        attempt = dict(row, result=row, model_calls=2, status='completed')
                        put(run / 'attempt-01.json', attempt)
                        put(run / 'result.json', row)
                        rows.append(json.dumps(row))
                (episodes / 'results.jsonl').write_text('\n'.join(rows) + '\n')
            def merge():
                return merge_campaign(directory, plan, selection, protocol_file)
            put(run / 'attempt-01.json', dict(attempt, result=None))
            with self.assertRaisesRegex(ValueError, 'Unverified terminal'):
                merge()
            self.assertFalse((directory / 'verified').exists())
            put(run / 'attempt-01.json', attempt)
            put(episodes / 'protocol.json', dict(shard_protocol, temperature=0))
            with self.assertRaisesRegex(ValueError, 'Unexpected sampling'):
                merge()
            put(episodes / 'protocol.json', shard_protocol)
            (episodes / 'results.jsonl').write_text('\n'.join(rows[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'Missing 1 of 32'):
                merge()
            (episodes / 'results.jsonl').write_text('\n'.join(rows) + '\n')
            dest = merge()
            self.assertEqual(json.loads((dest / 'audit.json').read_text())['episodes'], 256)
            self.assertEqual(set(json.loads((dest / 'initial_pass_rates.json').read_text())[task_name].values()), {0.5})
            self.assertEqual(merge(), dest)
            rates_link = directory / 'repair-rates.json'
            with patch('ais_frontier.plan_rank', return_value={'counts': {'keep': 26, 'no_terminal': 6}}):
                with self.assertRaisesRegex(ValueError, 'needs verified trials'):
                    verify_and_publish(directory, plan, selection, rates_link)
            self.assertFalse(rates_link.is_symlink())
            with patch('ais_frontier.plan_rank', return_value={'counts': {'keep': 32}}) as checked:
                verify_and_publish(directory, plan, selection, rates_link)
            self.assertEqual(checked.call_count, 8)
            self.assertEqual(rates_link.resolve(), dest / 'initial_pass_rates.json')


if __name__ == '__main__':
    unittest.main()
