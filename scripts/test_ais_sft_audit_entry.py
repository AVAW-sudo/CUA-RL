import json
import errno
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import ais_sft_audit as audit
from ais_sft_runtime import FILES, main as runtime_main, read_rank_status, run_campaign, snapshot
from ais_sft_eval import checkpointed_profile, run_profiles
from ais_sft_reuse import collect_scores, handoff, seed_rank
from prepare_sft_eval import prepare


class AuditEntryTests(unittest.TestCase):
    def test_master_recovers_from_peer_status_permission_error_without_resampling(self):
        groups = {'osworld': [f'd/{i}' for i in range(4)], 'scienceboard': [f's/{i}' for i in range(4)]}
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            for rank in range(8):
                shard = audit.assignment(groups, rank)
                domain, task = shard['tasks'][0].split('/')
                rows = [{'domain': domain, 'id': task, 'score': 0, 'status': 'completed'}]
                path = parent / f'rank-{rank}/status.json'
                path.parent.mkdir()
                audit.save(path, {'assignment': shard, 'plan_sha256': 'p', 'status': 'finished',
                                  'results': {profile: rows for profile in shard['profiles']}})
            peer = parent / 'rank-1/status.json'
            original_read = Path.read_text
            failures = [errno.ESTALE, '', errno.EACCES]

            def read(path, *args, **kwargs):
                if path == peer and failures:
                    code = failures.pop()
                    if code == '':
                        return ''
                    raise OSError(code, os.strerror(code), str(peer))
                return original_read(path, *args, **kwargs)

            with patch.object(Path, 'read_text', read), patch('ais_sft_runtime.time.sleep') as sleep, \
                 patch.object(audit, 'run_shard', side_effect=AssertionError('resampled')):
                run_campaign(parent, 0, None, groups, 'p', full=True)
            self.assertEqual([c.args for c in sleep.call_args_list], [(1,), (2,), (4,)])
            report = json.loads((parent / 'summary.json').read_text())
            self.assertEqual([group['scored'] for group in report['groups'].values()], [4, 4, 4])

    def test_rank_status_retry_is_bounded_and_invalid_json_fails_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            error = PermissionError(errno.EACCES, 'denied', str(path))
            with patch.object(Path, 'read_text', side_effect=error), patch('ais_sft_runtime.time.sleep') as sleep:
                with self.assertRaises(PermissionError):
                    read_rank_status(path)
                self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 120)
            path.write_text('{')
            with patch('ais_sft_runtime.time.sleep') as sleep:
                with self.assertRaises(json.JSONDecodeError):
                    read_rank_status(path)
                sleep.assert_not_called()
            path.write_text(' ')
            with patch('ais_sft_runtime.time.sleep') as sleep:
                with self.assertRaisesRegex(json.JSONDecodeError, str(path)):
                    read_rank_status(path)
                self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 120)

    def test_resume_worker_ignores_empty_peer_and_recovers_own_status(self):
        groups = {'scienceboard': ['s/0', 's/1', 's/2', 's/3'], 'osworld': []}
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            campaign = project / 'runs/ais-sft-evaluations/123'
            (campaign / 'runtime').mkdir(parents=True)
            (campaign / 'runtime/sha256.json').touch()
            peer = campaign / 'full/rank-4/status.json'
            peer.parent.mkdir(parents=True)
            peer.touch()
            own = campaign / 'full/rank-5/status.json'
            own.parent.mkdir()
            own.touch()
            terminal = {'status': 'finished', 'rank': 5, 'plan_sha256': 'p', 'assignment': audit.assignment(groups, 5)}
            def restore(delay):
                self.assertEqual(delay, 1)
                audit.save(own, terminal)
            with patch('ais_sft_runtime.PROJECT', project), patch('ais_sft_runtime.snapshot', return_value=project), \
                 patch.object(audit, 'prepare'), patch.object(audit, 'load_plan', return_value=(groups, 'p')), \
                 patch.object(audit, 'run_shard', side_effect=AssertionError('resampled')), \
                 patch.dict(os.environ, {'JOB_ID': '456', 'WORLD_SIZE': '8', 'RANK': '5'}), \
                 patch.object(sys, 'path', sys.path.copy()), patch.object(sys, 'argv', ['sft', '--full', '--resume-job', '123']), \
                 patch('ais_sft_runtime.time.sleep', side_effect=restore) as sleep:
                runtime_main()
            sleep.assert_called_once_with(1)

    def test_import_scores_uses_identity_and_profile_and_keeps_zero(self):
        from ais_vm_check import ROOT
        from checkpoint_specs import checkpoint_spec
        from sft_vm_image import DIGESTS
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / '12', Path(tmp) / '13'
            dst = new / 'full/rank-0'
            dst.parent.mkdir(parents=True)
            place = old / 'full/rank-0'
            place.mkdir(parents=True)
            (place / 'rank.lock').touch()
            snapshot(ROOT, old)
            shard = {'benchmark': 'scienceboard', 'checkpoint': 'scienceboard-sft',
                     'tasks': ['A/zero', 'A/partial', 'A/missing'], 'profiles': ['released', 'restored']}
            audit.save(place / 'status.json', {'rank': 0, 'assignment': shard, 'plan_sha256': 'old'})
            source = prepare()
            model = checkpoint_spec(ROOT, shard['checkpoint'])
            frozen = json.loads((old / 'runtime/sha256.json').read_text())
            adapters = {k: frozen['scripts/' + k] for k in ('evaluate_sft.py', 'eval_qemu_provider.py', 'sft_episode_audit.py')}
            for task, value in zip(shard['tasks'], [0.0, 0.5, None]):
                folder = place / 'released/tasks' / task
                episode = folder / 'evaluation'
                episode.mkdir(parents=True)
                row = {'domain': 'A', 'id': task.split('/')[1], 'score': value, 'model_requested': True, 'errors': [],
                       'status': 'completed' if value is not None else 'infrastructure_error'}
                # Model the interval after the terminal manifest and before the rank receipt.
                if task != 'A/partial':
                    audit.save(folder / 'outcome.json', row)
                summary = {'tasks': [row]}
                audit.save(episode / 'summary.json', summary)
                audit.save(episode / 'manifest.json', {'status': 'finished', 'task_selection': [task.split('/')],
                           'checkpoint': {'repo_id': model['model_id'], 'revision': model['model_revision']},
                           'prepared_source': json.loads((source / 'sft-source.json').read_text()),
                           'vm_sha256': DIGESTS['scienceboard'], 'scienceboard_setup': 'released',
                           'vm_backend': 'ais', 'proxy_policy': 'direct', 'upstream_args': {'max_steps': 50},
                           'generation_overrides': json.loads((ROOT / 'runs' / model['state_file']).read_text())['generation_overrides'],
                           'summary': summary, 'adapter_sha256': adapters})
            args = old, new, ROOT, source, [shard], 'new'
            with (place / 'rank.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with self.assertRaisesRegex(RuntimeError, 'still evaluating'):
                    handoff(*args)
            report = handoff(*args)
            self.assertEqual(report['groups']['scienceboard-released']['to_evaluate'], 1)
            self.assertEqual(report['groups']['scienceboard-restored']['to_evaluate'], 3)
            with patch('ais_sft_reuse.collect_scores', side_effect=AssertionError('source re-read')):
                self.assertEqual(handoff(*args), report)
            seed_rank(dst, shard, 'new')
            seed_rank(dst, shard, 'new')
            reused = json.loads((dst / 'released/tasks/A/zero/outcome.json').read_text())
            self.assertEqual((reused['score'], reused['reused_from']['job']), (0.0, '12'))
            self.assertFalse((dst / 'restored/tasks/A/zero/outcome.json').exists())
            self.assertFalse((dst / 'released/tasks/A/missing/outcome.json').exists())
            audit.save(dst / 'released/tasks/A/zero/outcome.json', {})
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                seed_rank(dst, shard, 'new')
            f = place / 'released/tasks/A/zero/evaluation/manifest.json'
            changed = json.loads(f.read_text())
            changed['scienceboard_setup'] = 'restored'
            audit.save(f, changed)
            with self.assertRaisesRegex(ValueError, 'evidence mismatch'):
                collect_scores(old, ROOT, source, [shard], 'new')

    def test_concurrent_tasks_own_receipts_and_stop_before_next_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'released'
            barrier = threading.Barrier(2, timeout=5)
            tasks = ['d/one', 'd/two']
            def run(out, selected, log):
                barrier.wait()
                out.mkdir()
                task, = selected
                audit.save(out / 'summary.json', {'tasks': [{'domain': 'd', 'id': task.split('/')[1],
                           'status': 'completed', 'score': 1.0}]})
                return 0
            self.assertEqual(checkpointed_profile(target, tasks, run, desktops=2), 0)
            summary = json.loads((target / 'summary.json').read_text())
            self.assertEqual([r['id'] for r in summary['tasks']], ['one', 'two'])
            self.assertEqual(summary['scored'], 2)
            rerun = Mock(side_effect=AssertionError('resampled'))
            self.assertEqual(checkpointed_profile(target, tasks, rerun, desktops=2), 0)
            rerun.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'before starting'):
                checkpointed_profile(target, ['d/three', 'd/four'], lambda *args: 2, desktops=1)
            self.assertFalse((target / 'tasks/d/four').exists())

    def test_resident_input_accepts_only_whole_literal_sequences(self):
        from sft_gui_input import input_events
        self.assertEqual(input_events("pyautogui.click(123,24);pyautogui.press('a', presses=2)"),
                         [['click', [123,24], {}], ['press', ['a'], {'presses':2}]])
        for command in ['pyautogui.click(1,2);other()', 'pyautogui.press(name)', 'pyautogui.click(**opts)',
                        'import pyautogui', 'pyautogui.click();x=2', 'pyautogui.write(b"text")']:
            self.assertIsNone(input_events(command))

    def test_resident_acknowledges_unix_socket_input(self):
        from sft_gui_input import __file__ as script
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sock = root / 'input.sock'
            (root / 'pyautogui.py').write_text("def press(key):\n    assert key == 'a'\n")
            with subprocess.Popen([sys.executable, '-S', script, str(sock)], env=dict(os.environ, PYTHONPATH=tmp),
                                  stderr=subprocess.DEVNULL) as server:
                try:
                    deadline = time.monotonic() + 5
                    while not sock.exists() and time.monotonic() < deadline and server.poll() is None:
                        time.sleep(.01)
                    with socket.socket(socket.AF_UNIX) as client:
                        client.settimeout(5)
                        client.connect(str(sock))
                        body = b'[["press", ["a"], {}]]'
                        client.sendall(b'POST /input HTTP/1.0\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
                        self.assertIn(b' 200 ', client.makefile('rb').readline())
                finally:
                    server.terminate()

    def test_snapshot_survives_replacement_of_shared_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'CODE/CUA-RL'
            for name in FILES:
                source = project / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(name)
            campaign = project / 'runs/ais-sft-evaluations/123'
            (campaign / 'full').mkdir(parents=True)
            root = snapshot(project, campaign)
            (project / 'scripts/evaluate_sft.py').write_text('older evaluator')
            self.assertEqual(snapshot(project, campaign), root)
            evaluator = root / 'scripts/evaluate_sft.py'
            self.assertEqual(evaluator.read_text(), 'scripts/evaluate_sft.py')
            self.assertEqual((root / 'runs/ais-sft-evaluations/123/full').resolve(), campaign / 'full')
            evaluator.chmod(0o600)
            evaluator.write_text('damaged frozen evaluator')
            with self.assertRaisesRegex(ValueError, 'Frozen SFT runtime changed'):
                snapshot(project, campaign)

    def test_template_retries_only_before_task_setup(self):
        from ais_vm_check import VM, build_sft_template
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempts = []
            def create_disk(vm, *args, **kwargs):
                (vm.work / 'disk.qcow2').touch()
                attempts.append(vm.work.name)
            with patch.object(VM, 'start', create_disk), \
                 patch.object(VM, 'wait_desktop', side_effect=[TimeoutError('boot timed out'), None]), \
                 patch.object(VM, 'guest'), patch.object(VM, 'qmp', return_value=''), patch.object(VM, 'stop') as stop:
                build_sft_template(root / 'base.qcow2', root, root)
            self.assertEqual(attempts, ['0', '1'])
            self.assertEqual(stop.call_count, 2)
            self.assertEqual(json.loads((root / 'template/startup-attempts.json').read_text()), ['boot timed out', 'ready'])
            self.assertTrue((root / 'clean.qcow2').is_file())

    def test_full_shards_and_denominators(self):
        groups, digest = audit.load_plan(prepare(), full=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rank in range(8):
                shard = audit.assignment(groups, rank)
                directory = root / f'rank-{rank}'
                directory.mkdir()
                rows = []
                for task in shard['tasks']:
                    domain, identifier = task.split('/')
                    rows.append({'domain': domain, 'id': identifier, 'status': 'completed', 'score': 0.25})
                audit.save(directory / 'status.json', {'assignment': shard, 'plan_sha256': digest, 'status': 'finished',
                           'results': {p: rows for p in shard['profiles']}})
            self.assertTrue(audit.publish_summary(root, groups, digest))
            report = json.loads((root / 'summary.json').read_text())['groups']
            for group, expected in [('osworld-released', 361), ('scienceboard-released', 169), ('scienceboard-restored', 169)]:
                self.assertEqual((report[group]['expected'], report[group]['scored'], report[group]['mean_score']),
                                 (expected, expected, 0.25))

    def test_full_restart_preserves_scores_and_interrupted_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'released'
            tasks = ['Lean/zero', 'Lean/interrupted', 'Lean/timeout', 'Lean/next']
            def complete(out, selected, log):
                out.mkdir()
                if selected == ['Lean/timeout']:
                    return 124
                domain, identifier = selected[0].split('/')
                audit.save(out / 'summary.json', {'tasks': [
                    {'domain': domain, 'id': identifier, 'status': 'completed', 'score': 0.0}]})
                return 0
            self.assertEqual(checkpointed_profile(target, tasks[:1], complete), 0)
            (target / 'tasks/Lean/interrupted/evaluation').mkdir(parents=True)
            runner = Mock(side_effect=complete)
            self.assertEqual(checkpointed_profile(target, tasks, runner), 2)
            self.assertEqual([c.args[1] for c in runner.call_args_list], [[tasks[2]], [tasks[3]]])
            report = json.loads((target / 'summary.json').read_text())
            self.assertEqual([row['score'] for row in report['tasks']], [0.0, None, None, 0.0])
            runner.reset_mock()
            self.assertEqual(checkpointed_profile(target, tasks, runner), 2)
            runner.assert_not_called()

    def test_assignments_and_independent_scores(self):
        groups, digest = audit.load_plan(prepare())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in range(8):
                shard = audit.assignment(groups, rank)
                self.assertEqual(len(shard['tasks']), 2)
                directory = root / f'rank-{rank}'
                directory.mkdir()
                for profile in shard['profiles']:
                    target = directory / profile
                    target.mkdir()
                    rows = []
                    for task in shard['tasks']:
                        domain, identifier = task.split('/')
                        missing = rank == 3 and not rows
                        rows.append({'domain': domain, 'id': identifier,
                                     'score': None if missing else 1.0,
                                     'status': 'missing_result' if missing else 'completed'})
                    (target / 'summary.json').write_text(json.dumps({'tasks': rows}))
                audit.save(directory / 'status.json', {'status': 'finished', 'assignment': shard,
                           'plan_sha256': digest, 'results': audit.collect_rank(directory, shard)})
                audit.publish_summary(root, groups, digest)
                self.assertEqual((root / 'summary.json').exists(), rank == 7)
            summary = json.loads((root / 'summary.json').read_text())['groups']
            self.assertEqual(summary['osworld-released']['scored'], 7)
            self.assertIsNone(summary['osworld-released']['mean_score'])
            self.assertEqual(summary['scienceboard-released']['tasks'], summary['scienceboard-restored']['tasks'])
            for key in ('scienceboard-released', 'scienceboard-restored'):
                self.assertEqual(set(summary[key]['tasks']), set(groups['scienceboard']))
                self.assertEqual(summary[key]['mean_score'], 1.0)
            self.assertEqual(set(summary['osworld-released']['tasks']), set(groups['osworld']))

    def test_paired_profile_continues_after_unscored_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = Mock()
            policy.poll.return_value = None
            calls = []
            def launch(cmd, **kwargs):
                if '--state-dir' in cmd:
                    target = Path(cmd[cmd.index('--state-dir') + 1])
                    target.mkdir()
                    audit.save(target / 'service.json', {'status': 'ready'})
                    return policy
                calls.append(cmd)
                target = Path(cmd[cmd.index('--run-dir') + 1])
                target.mkdir()
                audit.save(target / 'summary.json', {})
                code = 2 if target.name == 'released' else 0
                proc = Mock(returncode=code)
                proc.poll.return_value = code
                return proc
            with patch('ais_sft_eval.stage_worker', return_value=(root, {})), patch('ais_sft_eval.stop'), \
                 patch('ais_sft_eval.signal.signal'), patch('ais_sft_eval.subprocess.Popen', side_effect=launch) as popen:
                rc = run_profiles(root, 'scienceboard', 'scienceboard-sft', ['Lean/Lean_E-02'], root,
                                  [('released', 'released'), ('restored', 'restored')])
            self.assertEqual(rc, 2)
            self.assertEqual(popen.call_count, 3)
            self.assertEqual([cmd[cmd.index('--scienceboard-setup') + 1] for cmd in calls], ['released', 'restored'])
            self.assertEqual([cmd[cmd.index('--service-state') + 1] for cmd in calls], [str(root / 'policy/service.json')] * 2)
            self.assertEqual([cmd[cmd.index('--tasks') + 1:] for cmd in calls], [['Lean/Lean_E-02']] * 2)
