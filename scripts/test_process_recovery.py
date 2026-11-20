import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import persistent_job as jobs
import pipeline_watchdog as watchdog


class RecoveryTests(unittest.TestCase):
    def test_identity_rejects_reused_pid(self):
        value = jobs.identity(os.getpid())
        self.assertTrue(jobs.alive(value))
        self.assertFalse(jobs.alive({**value, 'start_ticks': value['start_ticks'] + 1}))

    def test_resume_attaches_to_one_command_and_retains_nonzero_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            count = root / 'starts.txt'
            command = [sys.executable, '-c',
                       f'from pathlib import Path; import time; p=Path({str(count)!r}); '
                       'p.write_text(p.read_text()+"x" if p.exists() else "x"); time.sleep(.3); raise SystemExit(7)']
            first = jobs.start(root / 'job', command, root)
            second = jobs.start(root / 'job', command, root)
            self.assertEqual(first.pid, second.pid)
            deadline = time.monotonic() + 10
            while first.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(first.poll(), 7)
            self.assertEqual(count.read_text(), 'x')
            self.assertEqual(jobs.attach(root / 'job').poll(), 7)

    def test_missing_runner_never_duplicates_live_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
            try:
                jobs.write(root / 'process.json', {'runner': {'pid': -1}, 'child': jobs.identity(child.pid)})
                self.assertIsNone(jobs.Job(root).poll())
            finally:
                child.terminate(); child.wait()
            with self.assertRaises(RuntimeError):
                jobs.Job(root).poll()

    def test_watchdog_obeys_disable_complete_and_supervisor_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = root / 'config.json'
            jobs.write(cfg, {'enabled': False})
            with patch.object(watchdog.subprocess, 'Popen') as spawn:
                watchdog.tick(root, cfg, ['unused'])
                jobs.write(cfg, {'enabled': True})
                jobs.write(root / 'state.json', {'phase': 'complete'})
                watchdog.tick(root, cfg, ['unused'])
                jobs.write(root / 'state.json', {})
                with (root / 'supervisor.lock').open('a') as lease:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertEqual(watchdog.tick(root, cfg, ['unused'])['action'], 'supervisor_lock_held')
                spawn.assert_not_called()

    def test_killed_caller_does_not_kill_persistent_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, '-c', 'import time; time.sleep(.5)']
            code = (f'import sys; sys.path.insert(0, {str(Path(jobs.__file__).parent)!r}); '
                    f'from persistent_job import start; from pathlib import Path; import time; '
                    f'start(Path({str(root / "job")!r}), {command!r}, Path({tmp!r})); time.sleep(30)')
            caller = subprocess.Popen([sys.executable, '-c', code])
            try:
                deadline = time.monotonic() + 10
                while not (root / 'job/process.json').exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                caller.kill(); caller.wait()
                job = jobs.attach(root / 'job')
                self.assertIsNotNone(job)
                while job.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertEqual(job.poll(), 0)
            finally:
                if caller.poll() is None:
                    caller.kill(); caller.wait()


if __name__ == '__main__':
    unittest.main()
