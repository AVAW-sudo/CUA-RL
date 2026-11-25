import copy
import hashlib
import os
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluate_sft import summarize
from sft_local_weights import prepare_weights
from sft_vm_image import verified_image
from evaluation_recovery import classify_attempt
from prepare_sft_eval import prepare
from scienceboard_compat import normalize_task, setup_repairs
from sft_episode_audit import EpisodeAudit, evidence_path


class SFTAuditTests(unittest.TestCase):
    def test_official_task_pairing_and_opt_in(self):
        source = prepare() / 'evaluation_examples'
        domains = json.loads((source / 'test_scienceboard.json').read_text())
        affected = set()
        for domain, ids in domains.items():
            for identifier in ids:
                task = json.loads((source / 'scienceboard' / domain / (identifier + '.json')).read_text())
                original = copy.deepcopy(task)
                released, _ = normalize_task(task)
                restored, changes = normalize_task(task, 'restored')
                self.assertEqual(task, original)
                self.assertEqual(restored['instruction'], released['instruction'])
                self.assertEqual(restored['evaluator']['eval_steps'], released['evaluator']['eval_steps'])
                if restored != released:
                    affected.add(identifier)
                    self.assertTrue(any(x['kind'].startswith('restore_') for x in changes))
                    bad = copy.deepcopy(task)
                    bad['config'].append({'type': 'sleep', 'parameters': {'seconds': 1}})
                    with self.assertRaises(ValueError):
                        normalize_task(bad, 'restored')
        self.assertEqual(affected, set(setup_repairs()))
        self.assertEqual(len(affected), 48)
        self.assertEqual(sum(map(len, domains.values())), 169)

    def test_weight_cache_pins_shards_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'weights'
            source.mkdir()
            shard = source / 'model.safetensors'
            shard.write_bytes(b'fixture')
            (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': shard.name}}))
            lock = root / 'lock.json'
            expected = {shard.name: hashlib.sha256(b'fixture').hexdigest()}
            lock.write_text(json.dumps({'c': {'revision': 'rev', 'sha256': expected}}))
            with patch('sft_local_weights.tempfile.gettempdir', return_value=tmp):
                prepare_weights('c', 'rev', source, None, root / 'view', lock)
                self.assertEqual((root / 'view' / shard.name).read_bytes(), b'fixture')
                prepare_weights('c', 'rev', source, None, root / 'another', lock)
                shard.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    prepare_weights('c', 'rev', source, source, root / 'rejected', lock)

    def test_vm_alternate_image_requires_matching_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'image'
            f.write_bytes(b'1234')
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            with patch('sft_vm_image.tempfile.gettempdir', return_value=tmp), patch.dict('sft_vm_image.DIGESTS', fixture=digest):
                with patch.dict(os.environ, CUA_EVAL_VM_IMAGE=str(f)):
                    self.assertEqual(verified_image('fixture', Path('absent'), 4), (f, digest))
                    f.write_bytes(b'4321')
                    with self.assertRaises(ValueError):
                        verified_image('fixture', Path('absent'), 4)

    def test_audited_scores_and_retry(self):
        for mode in ('valid_zero', 'verifier_error', 'inference_error'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = evidence_path(root, 'd', 'id')
                args = SimpleNamespace(action_space='a', observation_type='o', model='m')
                directory = root / 'results/a/o/m/d/id'
                directory.mkdir(parents=True)
                def model_call():
                    if mode == 'inference_error':
                        raise ValueError('context exceeded')
                def worker(_):
                    try:
                        runner.call_llm()
                    except ValueError:
                        pass
                    if mode == 'verifier_error':
                        logging.getLogger('desktopenv.scienceboard.evaluator').error('ScienceBoard evaluation error for KAlgebra: missing')
                    (directory / 'result.txt').write_text('0')
                    return 'd', 'id', 0
                runner = SimpleNamespace(call_llm=model_call, _worker_run=worker)
                EpisodeAudit(path).run(runner, None)
                self.assertIs(runner.call_llm, model_call)
                row = summarize(root, args, [('d', 'id')])['tasks'][0]
                self.assertTrue(row['model_requested'])
                self.assertEqual(row['upstream_score'], 0)
                expected = {'valid_zero': 'completed', 'verifier_error': 'verifier_review', 'inference_error': 'infrastructure_error'}
                self.assertEqual(row['status'], expected[mode])
                self.assertEqual(row['score'], 0 if mode == 'valid_zero' else None)
                attempt = {'attempt_dir': str(root), 'error': '[Init Fail]'}
                self.assertFalse(classify_attempt(attempt)['retryable'])
                path.write_text('{')
                self.assertFalse(classify_attempt(attempt)['retryable'])
                path.unlink()
                (root / 'summary.json').write_text(json.dumps({'tasks': [row]}))
                self.assertFalse(classify_attempt(attempt)['retryable'])


if __name__ == '__main__':
    unittest.main()
