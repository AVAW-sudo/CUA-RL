import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ais_screen_followup as screen
from ais_screen_queue import ScreenQueue, eligible
from ais_vm_check import save
from overnight_prompts import write_lines


class FollowupTest(unittest.TestCase):
    def test_preferences_judge_checks_the_shared_matcher_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = Path(temp)
            prefs = worker / 'thunderbird.py'
            prefs.write_text(
                '_pref_pattern: str = re.compile(r"enabled=(.+)")\n'
                'def check_thunderbird_prefs(path, rules):\n'
                '    with open(path) as file:\n'
                '        match = _pref_pattern.fullmatch(file.read())\n'
                '    return float(bool(match) and _match_pref(match.group(1), rules))\n')
            helper = worker / 'utils.py'
            helper.write_text('def _match_value_to_rule(value, rule):\n'
                              '    return operator.eq(value, rule["ref"])\n')
            case = dict(module=prefs.name, module_sha256=hashlib.sha256(prefs.read_bytes()).hexdigest(),
                        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(), input_format='text_file',
                        inputs={'untouched': '', 'correct': 'enabled=yes', 'disabled': 'enabled=no'})
            evaluator = dict(func='check_thunderbird_prefs', expected={'rules': {'ref': 'yes'}})
            self.assertEqual(screen.score_cases(worker, case, evaluator),
                             {'untouched': 0, 'correct': 1, 'disabled': 0})
            case['helper_sha256'] = 'stale'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                screen.score_cases(worker, case, evaluator)

    def test_multi_file_judge_controls_use_each_requested_path(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = Path(temp)
            module = worker / 'judge.py'
            module.write_text(
                'def get_pair(env, config):\n'
                '    return [env.controller.get_file(p) for p in config["paths"]]\n'
                'def match(state, rules):\n'
                '    return float(state == [b"one", b"two"])\n')
            checksum = hashlib.sha256(module.read_bytes()).hexdigest()
            case = dict(module=module.name, module_sha256=checksum,
                        getter_module=module.name, getter_sha256=checksum, input_format='text_files',
                        inputs={'untouched': {}, 'correct': {'/a': 'one', '/b': 'two'},
                                'wrong_path': {'/a': 'one', '/other': 'two'}})
            evaluator = dict(func='match', result={'type': 'pair', 'paths': ['/a', '/b']},
                             expected={'rules': {}})
            self.assertEqual(screen.score_cases(worker, case, evaluator),
                             {'untouched': 0, 'correct': 1, 'wrong_path': 0})

    def test_cpu_builds_queue_and_checks_frozen_history_without_old_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = root / 'configs'
            bundle = cfg / 'dataset-screen/followup'
            split = cfg / 'research-splits'
            split.mkdir(parents=True)
            bundle.mkdir(parents=True)
            worker = root / 'worker'
            tasks = worker / 'data/ScaleCUA/osworld/rl_tasks/app'
            tasks.mkdir(parents=True)
            out = root / 'job'
            out.mkdir()
            rows = []
            for name in ('new', 'scored', 'partial', 'unreviewed', 'reserve', 'duplicate', 'rejected',
                         'shared', 'historical'):
                kind = name if name in ('scored', 'partial') else 'new'
                if name == 'historical':
                    kind = 'scored'
                prompt = 'unreviewed' if name == 'duplicate' else name
                task = {'instruction': prompt, 'id': name, 'evaluator': {'func': 'metric', 'expected': {'rules': {}}}}
                path = tasks / (name + '.json')
                save(path, task)
                i = 'app_' + name
                row = dict(kind=kind, task_index=i, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           role='main', seed_score=0, reviewed_skill='Change the setting',
                           offline_controls={'untouched': 0, 'correct': 1, 'wrong_setting': 0})
                if kind != 'new':
                    row.update(source='ais-screen-expand/123/tasks/' + i,
                               raw_rewards=[0, 0.5, 0, int(kind == 'scored')],
                               receipt={'gate': dict(status='passed', reward=0, task_index=i, session_id='2'),
                                        'results': [str(n) for n in range(4)]})
                if name in ('unreviewed', 'reserve', 'duplicate'):
                    row.pop('reviewed_skill')
                    row.pop('offline_controls')
                    row.update(role='breadth', awaiting_review=True, reserve=name == 'reserve')
                if name == 'rejected':
                    row['hold'] = 'prior_review_rejected'
                if name in ('historical', 'shared'):
                    del row['reviewed_skill'], row['offline_controls']
                    if name == 'historical':
                        row['hold'] = 'needs_skill_or_judge_controls'
                    else:
                        row.update(awaiting_review=True, shared_setup_inputs=['common.pdf'])
                rows.append(row)
            save(split / 'train-examples.json', {'app': [r['task_index'] for r in rows]})
            for name in ('dev', 'test'):
                save(split / (name + '-indices.json'), [])
                save(split / (name + '-examples.json'), {})
            save(split / 'test-indices.json', ['app_heldout'])
            save(split / 'test-examples.json', {'app': ['app_heldout']})
            save(tasks / 'heldout.json', {'instruction': 'different goal'})
            for name in ('expand-review', 'expand-queue'):
                write_lines(bundle.parent / (name + '.jsonl'), [])
            write_lines(bundle / 'candidates.jsonl', rows)
            write_lines(bundle / 'baseline.jsonl', [])
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
            save(bundle / 'manifest.json', dict(revision='v1', source_job='123', files=hashes))
            save(cfg / 'sources.lock.json', {'training_data': {'revision': 'v1'}})
            with (patch.object(screen, 'ROOT', root), patch.object(screen, 'CONFIG', bundle.parent),
                  patch.object(screen, 'BUNDLE', bundle)):
                summary, queue, baseline = screen.prepare(out, worker)
                self.assertEqual((summary['strict_after_cpu'], summary['max_new_trials']), (1, 20))
                self.assertEqual([r['task_index'] for r in queue],
                                 ['app_partial', 'app_new', 'app_shared', 'app_unreviewed', 'app_reserve'])
                self.assertEqual(summary['new_main_with_shared_inputs'], 1)
                self.assertEqual(summary['historical_main_for_cpu_review'], 1)
                self.assertFalse(eligible(queue[2], {'raw_rewards': [1, 0, 0, 0], 'initial_state': {'status': 'passed'}}))
                screen.finish(ScreenQueue(out, set(), fixed_inputs=([], [])))
                self.assertEqual([r['task_index'] for r in screen.records(out / 'scored-for-review.jsonl')],
                                 ['app_historical'])
                self.assertEqual(summary['gpu_with_controls'], 2)
                self.assertFalse(eligible(queue[-1], {'raw_rewards': [0, 1, 1, 0], 'initial_state': {'status': 'passed'}}))
                self.assertEqual([r['task_index'] for r in baseline], ['app_scored'])
                save(tasks / 'heldout.json', {'instruction': ' SHARED '})
                with self.assertRaisesRegex(ValueError, 'held-out instruction'):
                    screen.prepare(out, worker)
                save(tasks / 'heldout.json', {'instruction': 'different goal'})
                save(tasks / 'new.json', {'instruction': 'different'})
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    screen.prepare(out, worker)

    def test_final_export_uses_the_same_fixed_snapshot_as_the_ranks(self):
        with tempfile.TemporaryDirectory() as temp:
            queue = ScreenQueue(Path(temp), set(), fixed_inputs=([], []))
            result = screen.finish(queue)
            self.assertEqual(result['strict_total'], 0)
            self.assertEqual(screen.read(Path(temp) / 'train-indices.json'), [])


if __name__ == '__main__':
    unittest.main()
