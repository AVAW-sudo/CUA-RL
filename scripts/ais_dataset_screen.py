"""Screen training variants with four frozen SFT trials on eight single-GPU AIS ranks."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys

from ais_frontier import run_collector, wait_for_shards
from ais_frontier_shards import read, require_equivalent_protocol, verify_attempt
from ais_rank_status import read_rank_status
from ais_vm_check import ROOT, save

DEFAULT_BATCH = ROOT / 'configs/dataset-screen/batch-02.json'
RUNS = ROOT / 'runs/ais-dataset-screen'


def plan(indices, lanes):
    tasks = read(indices)
    training = {i for domain in read(ROOT / 'configs/research-splits/train-examples.json').values() for i in domain}
    if (not isinstance(tasks, list) or not tasks or any(not isinstance(i, str) or i not in training for i in tasks)
            or len(tasks) != len(set(tasks))):
        raise ValueError('Select a nonempty list of unique research training variants')
    return {'batch_sha256': hashlib.sha256(indices.read_bytes()).hexdigest(), 'tasks': len(tasks),
            'trials': 4 * len(tasks), 'ranks': 8, 'desktops_per_rank': lanes,
            'shards': [tasks[rank::8] for rank in range(8)]}


def expected_protocol(indices):
    original = read(ROOT / 'runs/frontier-pilot-initialization/protocol.json')
    return dict(original, tasks=len(read(indices)), indices_sha256=hashlib.sha256(indices.read_bytes()).hexdigest(),
                services=original['services'][:1])


def checked_score(slot, task, number, protocol):
    row = read(slot / 'result.json')
    if row['task_index'] != task or row['run_number'] != number:
        raise ValueError(f'Misplaced result at {slot}')
    attempt = verify_attempt(slot, row, protocol)
    terminal = {'finish': True, 'status': row['terminal_status'], 'reward': row['raw_reward']}
    trace = slot / f"trace-{row['attempt']:02d}.jsonl"
    with trace.open() as stream:
        turns = [json.loads(line) for line in stream]
    if (not attempt.get('session_id') or attempt.get('terminal') != terminal or len(turns) != attempt['model_calls']
            or any(t['turn'] != n or 'message' not in t for n, t in enumerate(turns))):
        raise ValueError(f'Incomplete session or model trace at {slot}')
    if row['terminal_status'] == 'agent context limit' and turns[-1].get('usage', {}).get('total_tokens', 0) <= protocol['max_total_len']:
        raise ValueError(f'Unverified context limit at {slot}')
    return row


def coverage(out, tasks, protocol):
    rows = {}
    for task in tasks:
        rewards, missing = [], {}
        for trial in range(1, 5):
            slot = out / 'episodes' / task / str(trial)
            if (slot / 'result.json').is_file():
                rewards.append(checked_score(slot, task, trial, protocol)['raw_reward'])
            else:
                attempts = sorted(slot.glob('attempt-*.json'))
                last = read(attempts[-1]) if attempts else {}
                missing[str(trial)] = last.get('error') or last.get('status') or 'no terminal record'
                rewards.append(None)
        rows[task] = {'raw_rewards': rewards, 'unscored': missing,
                      'full_credit_rate': None if missing else rewards.count(1) / 4}
    return rows


def restore(source, out, tasks, protocol, cohort):
    # Only completed slots are reusable. Failed trials retain their original evidence under the source job.
    if not source.is_dir() or not (source / 'episodes/protocol.json').exists():
        if any((source / 'episodes').glob('*/*/attempt-*.json')):
            raise ValueError(f'Missing sampling protocol for existing attempts at {source}')
        return 0
    if read(source / 'cohort.json') != cohort:
        raise ValueError(f'Different task batch or runtime at {source}')
    require_equivalent_protocol(read(source / 'episodes/protocol.json'), protocol)
    kept = 0
    for task in tasks:
        for trial in range(1, 5):
            old = source / 'episodes' / task / str(trial)
            if not (old / 'result.json').is_file():
                continue
            checked_score(old, task, trial, protocol)
            new = out / 'episodes' / task / str(trial)
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old, new)
            checked_score(new, task, trial, protocol)
            kept += 1
    return kept


def summarize(rows):
    complete = [r['full_credit_rate'] for r in rows.values() if r['full_credit_rate'] is not None]
    scored = sum(sum(v is not None for v in row['raw_rewards']) for row in rows.values())
    return {'tasks': len(rows), 'trials': 4 * len(rows), 'scored': scored, 'unscored': 4 * len(rows) - scored,
            'tasks_complete': len(complete), 'full_credit_counts_out_of_four': dict(sorted(Counter(
                int(4 * rate) for rate in complete).items())), 'results': rows}


def merge(directory, batch, runtime):
    rows = {}
    for rank, tasks in enumerate(batch['shards']):
        lane = directory / f'rank-{rank}'
        record = read_rank_status(lane / 'cohort.json')
        if record != dict(runtime, batch_sha256=batch['batch_sha256'], rank=rank, tasks=tasks):
            raise ValueError(f'Inconsistent batch or runtime at rank {rank}')
        chunk = read_rank_status(lane / 'coverage.json')
        if set(chunk['results']) != set(tasks):
            raise ValueError(f'Incomplete task coverage at rank {rank}')
        rows.update(chunk['results'])
    summary = dict(summarize(rows), batch_sha256=batch['batch_sha256'], runtime=runtime)
    save(directory / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices', type=Path, default=DEFAULT_BATCH)
    parser.add_argument('--plan', action='store_true', help='Print the eight shards without starting services')
    parser.add_argument('--reuse-from', metavar='JOB_ID', help='Reuse verified scored slots from this screening job')
    parser.add_argument('--desktops-per-rank', type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    batch = plan(args.indices, args.desktops_per_rank)
    if args.plan:
        print(json.dumps(batch, indent=2))
        return
    rank = int(os.environ['RANK'])
    job = os.environ['JOB_ID']
    if int(os.environ['WORLD_SIZE']) != 8 or rank not in range(8) or not re.fullmatch(r'[0-9]+', job):
        parser.error('Screening needs one AIS job with eight single-GPU ranks and a numeric JOB_ID')
    if args.reuse_from and (not re.fullmatch(r'[0-9]+', args.reuse_from) or args.reuse_from == job
                            or not (RUNS / args.reuse_from).is_dir()):
        parser.error('--reuse-from needs a different existing numeric screening job ID')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    tasks = batch['shards'][rank]
    out = RUNS / job / f'rank-{rank}'
    out.mkdir(parents=True, exist_ok=False)
    indices = out / 'indices.json'
    save(indices, tasks)
    status = {'status': 'starting', 'rank': rank, 'tasks': len(tasks), 'reuse_from': args.reuse_from,
              'desktops_per_rank': args.desktops_per_rank, 'reused_slots': 0}
    print(f'AIS dataset screening rank {rank} | {len(tasks)} tasks | {out}', flush=True)
    save(out / 'status.json', status)
    try:
        cache = ROOT.parent.parent / 'ENV/cache'
        runtime = {name: read(path)['sha256'] for name, path in {
            'worker': Path(os.getenv('CUA_WORKER_MANIFEST', cache / 'ais-osworld-worker/manifest.json')),
            'policy': Path(os.getenv('CUA_POLICY_MANIFEST', cache / 'ais-frontier-vllm/manifest.json')),
            'fixtures': cache / 'ais-sft-osworld-inputs/manifest.json'}.items()}
        runtime['model_revision'] = read(ROOT / 'configs/sources.lock.json')['sft_model']['revision']
        cohort = dict(runtime, batch_sha256=batch['batch_sha256'], rank=rank, tasks=tasks)
        save(out / 'cohort.json', cohort)
        protocol = expected_protocol(indices)
        reused_protocol = None
        if args.reuse_from:
            source = RUNS / args.reuse_from / out.name
            status['reused_slots'] = restore(source, out, tasks, protocol, cohort)
            if status['reused_slots']:
                reused_protocol = source / 'episodes/protocol.json'
        if status['reused_slots'] < 4 * len(tasks):
            run_collector(out, indices, set(tasks), status, False, reused_protocol, screen_lanes=args.desktops_per_rank)
            require_equivalent_protocol(read(out / 'episodes/protocol.json'), protocol)
        elif tasks:
            save(out / 'episodes/protocol.json', read(reused_protocol))
        result = summarize(coverage(out, tasks, protocol))
        save(out / 'coverage.json', result)
        status.update(status='shard_complete', scored=result['scored'], unscored=result['unscored'])
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save(out / 'status.json', status)
    if rank == 0:
        wait_for_shards(out.parent)
        summary = merge(out.parent, batch, runtime)
        print(f"Screened {summary['scored']}/{summary['trials']} slots, {summary['tasks_complete']}/{summary['tasks']} tasks have four scores. {out.parent / 'summary.json'}", flush=True)
        if not summary['scored']:
            raise RuntimeError('No screening trial returned a verified score')


if __name__ == '__main__':
    main()
