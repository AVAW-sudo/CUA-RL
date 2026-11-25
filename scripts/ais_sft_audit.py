"""Run the fixed OSWorld and paired ScienceBoard SFT audit on eight AIS ranks."""
import argparse
import hashlib
import json
import os

from ais_sft_eval import checkpointed_profile, official_tasks, run_profiles
from ais_sft_reuse import seed_rank
from ais_vm_check import ROOT, save
from prepare_sft_eval import prepare


def load_plan(source, full=False):
    if full:
        groups = {}
        for benchmark, expected in [('osworld', 361), ('scienceboard', 169)]:
            tasks = official_tasks(source, benchmark)
            if len(tasks) != expected or len(set(tasks)) != expected:
                raise ValueError(f'Expected {expected} distinct official {benchmark} tasks')
            groups[benchmark] = tasks
        protocol = hashlib.sha256(json.dumps(groups, sort_keys=True).encode())
        for file in ['configs/sources.lock.json', 'configs/sft-eval-weights.json', 'configs/paths.json',
                     'patches/sft-evaluation.patch', 'scripts/ais_sft_audit.py', 'scripts/ais_sft_eval.py',
                     'scripts/evaluate_sft.py', 'scripts/eval_qemu_provider.py', 'scripts/ais_frontier_server.py',
                     'scripts/ais_vm_check.py', 'scripts/sft_gui_input.py', 'scripts/ais_sft_reuse.py',
                     'scripts/sft_episode_audit.py', 'scripts/scienceboard_compat.py', 'scripts/scienceboard_setup_repairs.json']:
            protocol.update((ROOT / file).read_bytes())
        protocol.update(os.getenv('CUA_SFT_TASK_TIMEOUT_SECONDS', '5400').encode())
        protocol.update(json.dumps([int(os.getenv('CUA_SFT_DESKTOPS', '1')),
                                    os.getenv('CUA_SFT_BOOT_MODE', 'cold')]).encode())
        return groups, protocol.hexdigest()
    content = (ROOT / 'configs/benchmarks/sft-audit-8.json').read_bytes()
    groups = json.loads(content)
    for benchmark in ('osworld', 'scienceboard'):
        selected = groups[benchmark]
        if len(selected) != 8 or len(set(selected)) != 8 or set(selected) - set(official_tasks(source, benchmark)):
            raise ValueError(f'Expected eight distinct official {benchmark} tasks')
    return groups, hashlib.sha256(content).hexdigest()


def assignment(groups, rank):
    benchmark = 'osworld' if rank < 4 else 'scienceboard'
    profiles = ['released'] if benchmark == 'osworld' else ['released', 'restored']
    return {'benchmark': benchmark, 'checkpoint': benchmark + '-sft', 'profiles': profiles,
            'tasks': groups[benchmark][rank % 4::4]}


def collect_rank(out, shard, partial=False):
    results = {}
    for profile in shard['profiles']:
        path = out / profile / 'summary.json'
        if partial and not path.exists():
            continue
        summary = json.loads(path.read_text())
        rows = summary['tasks']
        ids = [f"{r['domain']}/{r['id']}" for r in rows]
        expected = set(shard['tasks'])
        if len(ids) != len(set(ids)) or not set(ids) <= expected or (not partial and set(ids) != expected):
            raise ValueError(f'Unexpected evaluation task IDs in {out / profile}')
        results[profile] = rows
    return results


def publish_summary(parent, groups, digest):
    records = [parent / f'rank-{rank}' / 'status.json' for rank in range(8)]
    groups_out = {}
    failed_ranks = {}
    for rank, path in enumerate(records):
        try:
            status = json.loads(path.read_text())
        except FileNotFoundError:
            return False
        if status['status'] == 'running':
            return False
        shard = assignment(groups, rank)
        if status['plan_sha256'] != digest or status['assignment'] != shard:
            raise ValueError(f'Rank {rank} has a different evaluation plan')
        if status['status'] == 'failed':
            failed_ranks[rank] = status['error']
        for profile in shard['profiles']:
            key = f"{shard['benchmark']}-{profile}"
            rows = groups_out.setdefault(key, {})
            for row in status.get('results', {}).get(profile, []):
                rows[f"{row['domain']}/{row['id']}"] = row
    summary = {}
    for key, rows in groups_out.items():
        scores = [row['score'] for row in rows.values() if row['status'] == 'completed']
        expected = len(groups[key.split('-')[0]])
        summary[key] = {'expected': expected, 'scored': len(scores),
                        'mean_score': sum(scores) / expected if len(scores) == expected else None, 'tasks': rows}
    report = {'plan_sha256': digest, 'failed_ranks': failed_ranks, 'groups': summary}
    ledger = parent / 'reuse/manifest.json'
    if ledger.is_file():
        report['reuse_scored_from'] = json.loads(ledger.read_text())['source_job']
        for group in summary.values():
            reused = sum('reused_from' in r for r in group['tasks'].values())
            group.update(reused_scored=reused, newly_scored=group['scored'] - reused)
    save(parent / 'summary.json', report)
    return True


def main():
    from ais_sft_runtime import run_campaign
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Use both complete pinned official benchmark lists')
    parser.add_argument('--plan', action='store_true', help='Validate and print all eight assignments without starting services')
    args = parser.parse_args()
    source = prepare()
    groups, digest = load_plan(source, args.full)
    if args.plan:
        print(json.dumps([assignment(groups, r) for r in range(8)], indent=2))
        return
    job = os.environ.get('JOB_ID', '')
    rank = int(os.environ.get('RANK', '-1'))
    if not job.isdecimal() or os.environ.get('WORLD_SIZE') != '8' or not 0 <= rank < 8:
        parser.error('sft-audit expects an eight-rank AIS job with JOB_ID, RANK and WORLD_SIZE')
    parent = ROOT / 'runs/ais-sft-evaluations' / job / ('full' if args.full else 'audit-8')
    run_campaign(parent, rank, source, groups, digest, args.full)


def run_shard(source, out, rank, shard, digest, full):
    state = {'rank': rank, 'plan_sha256': digest, 'assignment': shard, 'status': 'running'}
    save(out / 'status.json', state)
    print(f'AIS SFT audit rank {rank}: {shard}, output: {out}', flush=True)
    try:
        if full:
            seed_rank(out, shard, digest)
        pending = []
        for profile in shard['profiles']:
            target = out / profile
            if full and all((target / 'tasks' / task / 'outcome.json').is_file() for task in shard['tasks']):
                checkpointed_profile(target, shard['tasks'], None)
            else:
                pending.append((profile, profile))
        if pending:
            run_profiles(source, shard['benchmark'], shard['checkpoint'], shard['tasks'], out,
                         pending, per_task=full)
        results = collect_rank(out, shard)
        unscored = sum(row['status'] != 'completed' for rows in results.values() for row in rows)
        state.update(status='finished_with_unscored' if unscored else 'finished', results=results)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        try:
            state['results'] = collect_rank(out, shard, partial=True)
        except (OSError, ValueError, KeyError, TypeError) as read_error:
            state['summary_error'] = repr(read_error)
        raise
    finally:
        save(out / 'status.json', state)
    print(f'AIS SFT audit rank {rank}: {state["status"]}, unscored: {unscored}', flush=True)


if __name__ == '__main__':
    main()
