import copy
from pathlib import Path
import tempfile
import unittest

from inference_pool import validate_services, routing_upgrade_source, validate_routing_upgrade


class ReplicaTests(unittest.TestCase):
    def service(self, port=18080):
        return {'checkpoint': 'osworld-sft', 'model_id': 'pinned-sft', 'model_revision': 'abc',
                'model_dir': '/weights/sft', 'served_model_name': 'sft', 'hostname': 'worker-1',
                'base_url': f'http://worker-1:{port}/v1',
                'command': ['python', 'vllm', '--model', '/weights/sft', '--port', str(port),
                            '--tensor-parallel-size', '1', '--max-model-len', '65536']}

    def test_equivalent_replicas_may_only_change_the_port(self):
        primary, replica = self.service(), self.service(18082)
        self.assertEqual(len(validate_services([primary, replica], primary)), 2)
        for key, value in [('model_revision', 'wrong'), ('model_dir', '/weights/rl'),
                           ('checkpoint', 'scienceboard-sft'), ('hostname', 'worker-0'),
                           ('runtime_compatibility', {'sha256': 'different-adapter'})]:
            other = copy.deepcopy(replica)
            other[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_services([primary, other], primary)
        replica['command'][-1] = '4096'
        with self.assertRaises(ValueError):
            validate_services([primary, replica], primary)

    def test_duplicate_endpoints_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_services([self.service(), self.service()], self.service())

    def test_upgrade_rejects_changes_to_sampling_or_verifiers(self):
        source = ("    args = parser.parse_args()\n"
                  "    state = json.loads(service_path.read_text())\n"
                  "    temperature = 0.4\n"
                  "    manifest['adapter_sha256'] = {}\n")
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'old', Path(tmp) / 'new'
            old.mkdir(); new.mkdir()
            for name in ['eval_qemu_provider.py', 'qemu_runtime.py', 'scienceboard_compat.py']:
                (old / name).write_text('unchanged')
                (new / name).write_text('unchanged')
            (old / 'evaluate_sft.py').write_text(source)
            patched = routing_upgrade_source(source)
            (new / 'evaluate_sft.py').write_text(patched)
            self.assertTrue(validate_routing_upgrade(old, new)['environment_and_verifier_adapters_unchanged'])
            (new / 'evaluate_sft.py').write_text(patched.replace('temperature = 0.4', 'temperature = 0.8'))
            with self.assertRaises(ValueError):
                validate_routing_upgrade(old, new)
            (new / 'evaluate_sft.py').write_text(patched)
            (new / 'scienceboard_compat.py').write_text('changed verifier')
            with self.assertRaises(ValueError):
                validate_routing_upgrade(old, new)


if __name__ == '__main__':
    unittest.main()
