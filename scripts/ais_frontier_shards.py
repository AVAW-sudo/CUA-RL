"""Partition the 64 frozen pilot tasks across eight single-GPU Frontier workers using historical timings."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def serving_options(command):
    """Retain model behavior, allowing the AIS interpreter, address, port and model mount."""
    options = list(command[command.index('-m'):])
    for name in ('--model', '--host', '--port'):
        i = options.index(name)
        del options[i:i + 2]
    return options


def serving_identity(service):
    return (service['model_revision'], service['served_model_name'], service['runtime_compatibility'],
            serving_options(service['command']))


def require_equivalent_protocol(previous, current):
    if ({k: v for k, v in previous.items() if k != 'services'} !=
            {k: v for k, v in current.items() if k != 'services'} or
            [serving_identity(s) for s in previous['services']] !=
            [serving_identity(s) for s in current['services']]):
        raise ValueError('New Frontier sampling protocol differs from the reused episodes')


def read(path):
    return json.loads(path.read_text())


def verify_protocol(episodes, rank, plan, manifest, original):
    baseline = original['services'][0]
    allowed_differences = {'tasks', 'indices_sha256', 'services'}
    frozen = {k: v for k, v in original.items() if k not in allowed_differences}
    p = read(episodes / 'protocol.json')
    shard_file = plan / f'rank-{rank}.json'
    tasks = read(shard_file)
    digest = hashlib.sha256(shard_file.read_bytes()).hexdigest()
    if (digest != manifest['sha256'][shard_file.name] or digest != p['indices_sha256']
            or p['tasks'] != len(tasks) or len(tasks) != 8):
        raise ValueError(f'Unexpected task set for rank {rank}')
    if {k: v for k, v in p.items() if k not in allowed_differences} != frozen or len(p['services']) != 1:
        raise ValueError(f'Unexpected sampling protocol for rank {rank}')
    service = p['services'][0]
    if serving_identity(service) != serving_identity(baseline):
        raise ValueError(f'Unexpected inference service for rank {rank}')
    return tasks


def verify_attempt(run, row, original):
    from build_initial_rates import TERMINAL_AGENT_STATES
    attempt = read(run / f"attempt-{row['attempt']:02d}.json")
    reward = row['raw_reward']
    if (attempt.get('result') != row or attempt['status'] != 'completed'
            or attempt['task_index'] != row['task_index'] or attempt['run_number'] != row['run_number']
            or attempt['model_revision'] != original['model_revision'] or attempt['model_calls'] < 1
            or attempt['attempt'] != row['attempt'] or row['model_revision'] != original['model_revision']
            or row['model'] != original['services'][0]['served_model_name']
            or type(reward) not in (float, int) or not math.isfinite(reward) or not 0 <= reward <= 1
            or int(reward == 1) != row['result'] or row['terminal_status'] not in TERMINAL_AGENT_STATES):
        raise ValueError(f'Unverified terminal record at {run}')
    return attempt


def write_shard_rates(directory, target, protocol, indices, model):
    from build_initial_rates import collect
    from run_benchmark import write_json
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob('*/*/result.json'))]
    result_file = directory / 'results.jsonl'
    result_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    rates = collect(rows, indices, model)
    if target.exists() and json.loads(target.read_text()) != {protocol['task_name']: rates}:
        raise ValueError('Refusing to replace existing Frontier rates with different results')
    write_json(target, {protocol['task_name']: rates})
    write_json(target.with_suffix('.audit.json'), {'protocol': protocol, 'episodes': len(rows),
                                                  'results_sha256': hashlib.sha256(result_file.read_bytes()).hexdigest()})


def merge_campaign(directory, plan, original_indices, original_protocol):
    from ais_rank_status import read_rank_status
    from build_initial_rates import collect
    manifest = read(plan / 'manifest.json')
    original = read(original_protocol)
    baseline = original['services'][0]
    indices = read(original_indices)
    if manifest['source_indices_sha256'] != hashlib.sha256(original_indices.read_bytes()).hexdigest():
        raise ValueError('Shard plan and original selection differ')

    def verify_rank(rank):
        lane = directory / f'rank-{rank}'
        episodes = lane / 'episodes'
        if read_rank_status(lane / 'status.json')['status'] != 'shard_complete':
            raise ValueError(f'Rank {rank} has no completed shard')
        tasks = verify_protocol(episodes, rank, plan, manifest, original)
        result_file = episodes / 'results.jsonl'
        rows = [json.loads(line) for line in result_file.read_text().splitlines()]
        rates = collect(rows, tasks, baseline['served_model_name'])
        if read(lane / 'rates.json') != {original['task_name']: rates}:
            raise ValueError(f'Rank {rank} rates differ from its terminal rows')
        for row in rows:
            run = episodes / row['task_index'] / str(row['run_number'])
            if read(run / 'result.json') != row:
                raise ValueError(f'Terminal file differs from aggregate at {run}')
            verify_attempt(run, row, original)
        return rows, {'protocol_sha256': hashlib.sha256((episodes / 'protocol.json').read_bytes()).hexdigest(),
                      'results_sha256': hashlib.sha256(result_file.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(verify_rank, range(8)))
    rows = [r for lane, _ in verified for r in lane]
    combined = {original['task_name']: collect(rows, indices, baseline['served_model_name'])}
    audit = {'episodes': len(rows), 'tasks': len(indices), 'source_indices_sha256': manifest['source_indices_sha256'],
             'model_revision': original['model_revision'], 'ranks': [evidence for _, evidence in verified]}
    target = directory / 'verified'
    if target.exists():
        if read(target / 'initial_pass_rates.json') != combined or read(target / 'audit.json') != audit:
            raise ValueError('Existing published rates have different provenance')
        return target
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        staged = Path(temporary) / 'verified'
        staged.mkdir()
        for name, value in [('initial_pass_rates.json', combined), ('audit.json', audit)]:
            (staged / name).write_text(json.dumps(value, indent=2) + '\n')
        staged.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'configs/original-pilot/train-indices.json'
    indices = json.loads(source.read_text())
    assert len(indices) == len(set(indices)) == 64
    history = ROOT / 'runs/frontier-pilot-initialization'

    def elapsed(index):
        durations = []
        for run in range(1, 5):
            files = sorted((history / index / str(run)).glob('attempt-*.json'))
            for path in files:
                attempt = json.loads(path.read_text())
                if attempt['status'] == 'completed':
                    timing = attempt['timings']
                    seconds = sum(timing[k] for k in ('initialization_seconds', 'inference_seconds', 'environment_seconds'))
                    if math.isfinite(seconds) and seconds > 0:
                        durations.append(seconds)
                        break
        return median(durations) if durations else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        observed = dict(zip(indices, pool.map(elapsed, indices)))
    fallback = median(v for v in observed.values() if v is not None)
    cost = {index: 4 * (v if v is not None else fallback) for index, v in observed.items()}
    shards = [[] for _ in range(8)]
    load = [0.0] * 8
    for index in sorted(indices, key=lambda i: (-cost[i], i)):
        rank = min((r for r in range(8) if len(shards[r]) < 8), key=lambda r: (load[r], r))
        shards[rank].append(index)
        load[rank] += cost[index]
    round_robin = [sum(cost[i] for i in indices[r::8]) for r in range(8)]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        ready = Path(tmp) / 'plan'
        ready.mkdir()
        hashes = {}
        for rank, tasks in enumerate(shards):
            name = f'rank-{rank}.json'
            path = ready / name
            path.write_text(json.dumps(tasks, indent=2) + '\n')
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {'source_indices_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'history': str(history.relative_to(ROOT)), 'task_count': 64, 'ranks': 8,
                    'tasks_per_rank': 8, 'runs_per_task': 4, 'tasks_with_cost_history': sum(v is not None for v in observed.values()),
                    'basis': 'Four times median elapsed time for completed attempts; global median fallback. Outcome values are unused.',
                    'round_robin_predicted_seconds': round_robin, 'balanced_predicted_seconds': load, 'sha256': hashes}
        (ready / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            if (output / 'manifest.json').read_bytes() != (ready / 'manifest.json').read_bytes() or any(
                    (output / name).read_bytes() != (ready / name).read_bytes() for name in hashes):
                raise FileExistsError(f'Existing plan differs: {output}')
        else:
            ready.rename(output)
    print(f"256 slots; predicted longest lane {max(round_robin):.0f}s -> {max(load):.0f}s on historical timings")


if __name__ == '__main__':
    main()
