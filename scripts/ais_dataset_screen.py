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
import threading

from ais_frontier import run_collector, wait_for_shards
from ais_frontier_shards import read, require_equivalent_protocol, verify_attempt
from ais_rank_status import read_rank_status
from ais_vm_check import ROOT, save

BATCH_DIR = ROOT / 'configs/dataset-screen'
DEFAULT_BATCHES = (BATCH_DIR / 'batch-02.json', BATCH_DIR / 'batch-03.json')
RUNS = ROOT / 'runs/ais-dataset-screen'


def plan(indices, lanes):
    manifests = DEFAULT_BATCHES if indices is None else (indices,)
    tasks = []
    for path in manifests:
        items = read(path)
        if not isinstance(items, list) or not items:
            raise ValueError(f'Expected a nonempty task list in {path}')
        tasks.extend(items)
    training = {i for domain in read(ROOT / 'configs/research-splits/train-examples.json').values() for i in domain}
    if (not isinstance(tasks, list) or not tasks or any(not isinstance(i, str) or i not in training for i in tasks)
            or len(tasks) != len(set(tasks))):
        raise ValueError('Select a nonempty list of unique research training variants')
    contents = manifests[0].read_bytes() if len(manifests) == 1 else json.dumps(tasks).encode()
    return {'batch_sha256': hashlib.sha256(contents).hexdigest(), 'tasks': len(tasks),
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


def validate_control(control, task):
    status = control.get('status')
    if control.get('task_index') != task or status not in ('passed', 'nonzero_initial_reward', 'unavailable'):
        raise ValueError(f'Invalid initial-state receipt for {task}')
    reward = control.get('reward')
    if status == 'unavailable':
        if reward is not None:
            raise ValueError(f'Unavailable control has a reward for {task}')
    elif (not control.get('session_id') or type(reward) not in (int, float) or not 0 <= reward <= 1
          or (reward == 0) != (status == 'passed')):
        raise ValueError(f'Invalid initial-state reward for {task}')


def coverage(out, tasks, protocol, *, overnight=False):
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
        if overnight:
            control = read(out / 'episodes' / task / 'initial-state.json')
            validate_control(control, task)
            if control['status'] != 'passed' and any(v is not None for v in rewards):
                raise ValueError(f'Unexpected scores without a zero baseline for {task}')
            rows[task]['initial_state'] = control
    return rows


def restore(source, out, tasks, protocol, cohort, *, merge_existing=False):
    # Only completed slots are reusable. Failed trials retain their original evidence under the source job.
    if not source.is_dir() or not (source / 'episodes/protocol.json').exists():
        if any((source / 'episodes').glob('*/*/attempt-*.json')):
            raise ValueError(f'Missing sampling protocol for existing attempts at {source}')
        return 0
    previous = read(source / 'cohort.json')
    source_tasks = read(source / 'indices.json')
    if (source_tasks != previous['tasks'] or tasks[:len(source_tasks)] != source_tasks or
            {k: v for k, v in previous.items() if k not in ('tasks', 'batch_sha256')} !=
            {k: v for k, v in cohort.items() if k not in ('tasks', 'batch_sha256')}):
        raise ValueError(f'Different task batch or runtime at {source}')
    previous_protocol = read(source / 'episodes/protocol.json')
    require_equivalent_protocol(previous_protocol, expected_protocol(source / 'indices.json'))
    # Only the roster grew. Check sampling and serving options independently of that roster.
    require_equivalent_protocol(dict(previous_protocol, tasks=protocol['tasks'],
                                     indices_sha256=protocol['indices_sha256']), protocol)
    kept = 0
    for task in source_tasks:
        if cohort.get('initial_state_check'):
            previous_control = source / 'episodes' / task / 'initial-state.json'
            control = read(previous_control) if previous_control.is_file() else None
            if not control or control['status'] == 'unavailable':
                if any((source / 'episodes' / task).glob('*/result.json')):
                    raise ValueError(f'Missing verified control for scored {task}')
                continue
            validate_control(control, task)
            if control['status'] != 'passed' and any((source / 'episodes' / task).glob('*/result.json')):
                raise ValueError(f'Unexpected scores under a rejected initial state for {task}')
            dest = out / 'episodes' / task
            dest.mkdir(parents=True, exist_ok=True)
            gate = dest / 'initial-state.json'
            if merge_existing and gate.is_file():
                existing = read(gate)
                validate_control(existing, task)
                if existing['status'] != control['status']:
                    raise ValueError(f'Conflicting initial-state verdicts for {task}')
            else:
                save(gate, control)
        for trial in range(1, 5):
            old = source / 'episodes' / task / str(trial)
            if not (old / 'result.json').is_file():
                continue
            checked_score(old, task, trial, previous_protocol)
            new = out / 'episodes' / task / str(trial)
            if merge_existing and (new / 'result.json').is_file():
                continue
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
    parser.add_argument('--indices', type=Path, help='Override the default consecutive screening batches')
    parser.add_argument('--overnight', action='store_true', help='Run the frozen overnight queue and export the prompt set')
    parser.add_argument('--check-initial-state', action='store_true',
                        help='Require a zero-reward untouched desktop before sampling each candidate')
    parser.add_argument('--plan', action='store_true', help='Print the eight shards without starting services')
    parser.add_argument('--reuse-from', metavar='JOB_ID', help='Reuse verified scored slots from this screening job')
    parser.add_argument('--desktops-per-rank', type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    check_initial_state = args.overnight or args.check_initial_state
    if args.overnight:
        from overnight_prompts import MANIFEST, export, publish_latest
        if args.indices:
            parser.error('The overnight prompt export uses its frozen task manifest')
        args.indices = MANIFEST
    batch = plan(args.indices, args.desktops_per_rank)
    if args.overnight:
        # Freeze the exporter to this job even if the shared checkout is updated later.
        roster = sorted((8 * n + rank, task) for rank, shard in enumerate(batch['shards'])
                        for n, task in enumerate(shard))
        candidates = [task for _, task in roster]
    if args.plan:
        batch['check_initial_state'] = check_initial_state
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
    stop_export = threading.Event()
    reporter = None
    try:
        cache = ROOT.parent.parent / 'ENV/cache'
        runtime = {name: read(path)['sha256'] for name, path in {
            'worker': Path(os.getenv('CUA_WORKER_MANIFEST', cache / 'ais-osworld-worker/manifest.json')),
            'policy': Path(os.getenv('CUA_POLICY_MANIFEST', cache / 'ais-frontier-vllm/manifest.json')),
            'fixtures': cache / 'ais-sft-osworld-inputs/manifest.json'}.items()}
        runtime['model_revision'] = read(ROOT / 'configs/sources.lock.json')['sft_model']['revision']
        if check_initial_state:
            runtime['initial_state_check'] = 'untouched-done-v1'
        cohort = dict(runtime, batch_sha256=batch['batch_sha256'], rank=rank, tasks=tasks)
        save(out / 'cohort.json', cohort)
        protocol = expected_protocol(indices)
        reused_protocol = None
        if args.reuse_from:
            source = RUNS / args.reuse_from / out.name
            status['reused_slots'] = restore(source, out, tasks, protocol, cohort)
            if status['reused_slots']:
                reused_protocol = out / 'reuse-protocol.json'
                save(reused_protocol, protocol)
        if args.overnight and rank == 0:
            def snapshots():
                while not stop_export.is_set():
                    try:
                        export(out.parent, out.parent, candidates=candidates)
                        publish_latest(out.parent)
                    except (OSError, ValueError, KeyError) as exc:
                        print(f'Prompt snapshot will retry: {exc}', flush=True)
                    stop_export.wait(300)
            reporter = threading.Thread(target=snapshots, daemon=True)
            reporter.start()
        if status['reused_slots'] < 4 * len(tasks):
            run_collector(out, indices, set(tasks), status, False, reused_protocol, screen_lanes=args.desktops_per_rank,
                          check_initial_state=check_initial_state)
            require_equivalent_protocol(read(out / 'episodes/protocol.json'), protocol)
        elif tasks:
            save(out / 'episodes/protocol.json', read(reused_protocol))
        result = summarize(coverage(out, tasks, protocol, overnight=check_initial_state))
        save(out / 'coverage.json', result)
        status.update(status='shard_complete', scored=result['scored'], unscored=result['unscored'])
    except BaseException as exc:
        stop_export.set()
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save(out / 'status.json', status)
    if rank == 0:
        try:
            wait_for_shards(out.parent)
            summary = merge(out.parent, batch, runtime)
        finally:
            stop_export.set()
            if reporter is not None:
                reporter.join()
        if args.overnight:
            export(out.parent, out.parent, final=summary['results'], candidates=candidates)
            publish_latest(out.parent)
        print(f"Screened {summary['scored']}/{summary['trials']} slots, {summary['tasks_complete']}/{summary['tasks']} tasks have four scores. {out.parent / 'summary.json'}", flush=True)
        if not summary['scored']:
            raise RuntimeError('No screening trial returned a verified score')


if __name__ == '__main__':
    main()
