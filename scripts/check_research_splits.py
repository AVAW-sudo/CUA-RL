"""Validate the actual upstream OSWorld loader against the research partition."""
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ['SCALE-CUA/osworld_eval', 'SCALE-CUA/osworld_env/worker']]
from src.task import OSWorldTask

task = object.__new__(OSWorldTask)
task.logger = logging.getLogger('split-check')
task.use_proxy = False
manifest = json.loads((ROOT / 'configs/research-splits/manifest.json').read_text())
loaded = {}
for name in ['train', 'dev', 'test']:
    data = task._load_tasks(str(ROOT / 'data/ScaleCUA/osworld/rl_tasks'),
                            str(ROOT / f'configs/research-splits/{name}-examples.json'))
    expected = manifest['splits'][name]
    assert set(data) == set(expected['task_indices']), name
    assert {t['id'] for t in data.values()} == set(expected['base_ids']), name
    loaded[name] = set(data)
assert sum(map(len, loaded.values())) == 2049
assert not any(loaded[a] & loaded[b] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')])
report = {'passed': True, 'actual_upstream_loader_counts': {k: len(v) for k, v in loaded.items()}}
benchmark_counts = {}
for name, count in [('test_nogdrive', 361), ('test_all', 369)]:
    data = task._load_tasks(str(ROOT / 'SCALE-CUA/osworld_eval/evaluation_examples/examples'),
                           str(ROOT / f'configs/benchmarks/{name}-examples.json'))
    expected = json.loads((ROOT / f'configs/benchmarks/{name}-indices.json').read_text())
    assert len(data) == count and set(data) == set(expected), name
    benchmark_counts[name] = len(data)
report['actual_upstream_benchmark_counts'] = benchmark_counts
(ROOT / 'reports/research-split-loader.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
