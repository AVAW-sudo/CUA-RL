#!/usr/bin/env python3
"""Validate four server_agent.py runs per task before writing Frontier rates.

This validates aggregate result coverage. The upstream JSONL does not record the
full rollout protocol or terminal status; audit those in the run configuration
and traces. A numeric result alone cannot establish environment validity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_AGENT_STATES = {'completed', 'agent context limit', 'agent validation failed',
                         'agent invalid action', 'task limit reached'}


def collect(rows, indices, model):
    expected = {str(index) for index in indices}
    if not expected or len(expected) != len(indices) or '-1' in expected:
        raise ValueError('Expected task indices must be nonempty, unique and exclude -1')
    results = {}
    for row in rows:
        if row.get('model') != model:
            raise ValueError('Input contains another model; use a dedicated SFT result file')
        index, run = str(row['task_index']), row['run_number']
        if index not in expected or type(run) is not int or run not in range(1, 5):
            raise ValueError(f'Unexpected task/run: {index}/{run}')
        key = (index, run)
        if key in results:
            raise ValueError(f'Duplicate result: {key}; use the resumed server_agent aggregate file')
        reward = row['result']
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError(f'Invalid/error result for {key}: {reward!r}')
        results[key] = int(reward == 1)
    missing = {(index, run) for index in expected for run in range(1, 5)} - results.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} of {len(expected) * 4} required runs; no rates written')
    return {index: sum(results[index, run] for run in range(1, 5)) / 4 for index in sorted(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Dedicated SFT server_agent.py aggregate JSONL')
    parser.add_argument('--indices', type=Path, default=ROOT / 'configs/rl-indices.json')
    parser.add_argument('--model', default='scalecua-qwen3-vl-8b-sft')
    parser.add_argument('--task-name', default='osworld-rl')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/initial_pass_rate.osworld-rl.json')
    args = parser.parse_args()
    raw_input, raw_indices = args.input.read_bytes(), args.indices.read_bytes()
    rows = [json.loads(line) for line in raw_input.decode().splitlines() if line.strip()]
    rates = collect(rows, json.loads(raw_indices), args.model)
    if args.output.exists():
        parser.error('Output already exists; choose a new --output to preserve experiment provenance')
    with args.output.open('x') as stream:
        stream.write(json.dumps({args.task_name: rates}, indent=2) + '\n')
    report = {'tasks': len(rates), 'runs_per_task': 4, 'episodes': len(rows),
              'input': str(args.input.resolve()), 'input_sha256': hashlib.sha256(raw_input).hexdigest(),
              'indices_sha256': hashlib.sha256(raw_indices).hexdigest(),
              'model': args.model, 'output': str(args.output.resolve()),
              'success_definition': 'verifier result == 1',
              'scope': 'Coverage/schema validation only; audit SFT revision, terminal statuses, horizon and sampling from run config/traces'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
