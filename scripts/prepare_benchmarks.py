"""Convert official OSWorld lists to controller IDs, preserving the exact task set."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'SCALE-CUA/osworld_eval/evaluation_examples'
target = ROOT / 'configs/benchmarks'
target.mkdir(exist_ok=True)
domains = sorted(p.name for p in (source / 'examples').iterdir() if p.is_dir())
reports = {}
for name, count in [('test_nogdrive', 361), ('test_all', 369)]:
    path = source / (name + '.json')
    official = json.loads(path.read_text())
    examples = {domain: [f'{domain}_{index}' for index in official.get(domain, [])] for domain in domains}
    assert not (set(official) - set(domains))
    indices = [index for group in examples.values() for index in group]
    assert len(indices) == len(set(indices)) == count
    for domain, raw_ids in official.items():
        for index in raw_ids:
            task = json.loads((source / 'examples' / domain / (index + '.json')).read_text())
            assert task['id'] == index
    (target / (name + '-examples.json')).write_text(json.dumps(examples, indent=2) + '\n')
    (target / (name + '-indices.json')).write_text(json.dumps(indices, indent=2) + '\n')
    reports[name] = {'task_count': count, 'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'per_domain': {k: len(v) for k, v in examples.items()}}
(ROOT / 'reports/benchmark-lists.json').write_text(json.dumps(reports, indent=2) + '\n')
protocol = {'status': 'prepared, not evaluated', 'primary': 'test_nogdrive', 'separate_secondary': 'test_all',
            'task_names': {'test_nogdrive': 'osworld-eval-361', 'test_all': 'osworld-eval-369'},
            'worker_data_path': str(source / 'examples'), 'max_environment_steps': 50,
            'screen_size': [1920, 1080], 'image_size': [1280, 720], 'relative_coordinates': True,
            'temperature': 0.8, 'top_p': 1.0, 'max_output_tokens': 4096,
            'sampling_note': 'Also explicitly set and record top_k (checkpoint default is 20), seed support, and chat/reasoning parsers when launching services; do not silently vary them across checkpoints',
            'checkpoints': ['SFT', 'published RL reference', 'local original-algorithm RL', 'research ablations'],
            'report': 'per-task outcome and aggregate success; retain infrastructure failures separately; use identical retry policy and VM image for all methods',
            'examples_path': 'Use the generated *-examples.json maps for workers; *-indices.json for clients',
            'current_limitations': ['No benchmark worker registered yet', 'TCG bring-up is not a suitable full benchmark throughput environment', 'Public training seed IDs overlap official benchmark; disclose protocol']}
(target / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
print(json.dumps({k: v['task_count'] for k, v in reports.items()}))
