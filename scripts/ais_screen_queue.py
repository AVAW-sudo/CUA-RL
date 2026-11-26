"""Eight AIS ranks share a replenishable queue of official training tasks."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from ais_dataset_screen import DEFAULT_BATCHES, RUNS as OLD_RUNS, coverage, expected_protocol, restore
from ais_frontier import collector_services, wait_for_shards
from ais_frontier_shards import read, require_equivalent_protocol
from ais_rank_status import read_rank_status
from ais_vm_check import ROOT, save
from build_overnight_screen import AUDIT, NOTEBOOK, normalized
from overnight_prompts import write_lines

CONFIG = ROOT / 'configs/dataset-screen'
ROSTER = CONFIG / 'expand-queue.jsonl'
REVIEWED = CONFIG / 'strict-expanded-20261126.jsonl'
RUNS = ROOT / 'runs/ais-screen-expand'
QUOTAS = {'main': 128, 'component': 64, 'breadth': 64}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def controls_pass(scores):
    wrong = [v for key, v in scores.items() if key not in ('correct', 'untouched')]
    return (scores.get('correct') == 1 and scores.get('untouched') == 0 and bool(wrong)
            and all(type(v) in (int, float) and 0 <= v < 1 for v in wrong))


def previous_batches():
    paths = (*DEFAULT_BATCHES, CONFIG / 'overnight-v1.json', CONFIG / 'expansion-20261126.json',
             CONFIG / 'wide-20261126.json', ROOT / 'configs/original-pilot/train-indices.json')
    previous = {i for p in paths for i in read(p)}
    if NOTEBOOK.is_file():
        previous.update(r['variant_id'] for r in read(NOTEBOOK)['tasks'])
    for summary in OLD_RUNS.glob('*/summary.json'):
        previous.update(read(summary)['results'])
    return previous


def check_manifest(rows, training, previous=(), catalog=None):
    ids, prompts = set(), set()
    sampled_prompts = {normalized(catalog[i]['instruction']) for i in previous
                       if catalog is not None and i in catalog}
    for row in rows:
        i = row['task_index']
        prompt = normalized(row['prompt'])
        if i not in training or i in ids or not prompt or prompt in prompts or row['role'] not in QUOTAS:
            raise ValueError(f'Invalid or duplicate screening candidate: {i}')
        if row.get('reviewed_skill') and not controls_pass(row.get('offline_controls', {})):
            raise ValueError(f'Reviewed skill lacks judge controls: {i}')
        extra = row['purpose'] == 'additional_four'
        if catalog is not None:
            official = catalog[i]
            if (row['prompt'] != official['instruction'] or official['heldout_instruction']
                    or official['shared_setup_inputs']):
                raise ValueError(f'Candidate differs from official training prompt or overlaps holdout: {i}')
        if extra:
            prior = row['prior_raw_rewards']
            if (not row['prior_job'].isdigit() or len(prior) != 4
                    or not all(type(v) in (float, int) and 0 <= v < 1 for v in prior)
                    or not controls_pass(row.get('offline_controls', {}))):
                raise ValueError(f'Missing evidence for supplemental trials: {i}')
        elif row['purpose'] != 'first_four' or 'prior_raw_rewards' in row:
            raise ValueError(f'Unknown sampling purpose: {i}')
        elif i in previous or prompt in sampled_prompts:
            raise ValueError(f'Previously scheduled task requested as a new candidate: {i}')
        ids.add(i)
        prompts.add(prompt)


class ScreenQueue:
    def __init__(self, directory, training, roster=ROSTER, reviewed=REVIEWED, previous=(), catalog=None):
        self.directory = directory
        self.training = training
        self.roster = roster
        self.reviewed = reviewed
        self.previous = previous
        self.catalog = catalog
        self.tasks = directory / 'tasks'
        self.tasks.mkdir(parents=True, exist_ok=True)
        self.done = {}

    def snapshot(self):
        rows = records(self.roster)
        check_manifest(rows, self.training, self.previous, self.catalog)
        by_id = {row['task_index']: row for row in rows}
        baseline = records(self.reviewed)
        accepted = {}
        prompts = set()
        counts = Counter()
        for r in baseline:
            i, role, prompt = r['task_index'], r['selection_role'], normalized(r['prompt'])
            if (i not in self.training or i in accepted or prompt in prompts or role not in QUOTAS
                    or r['valid_trials'] != 4 or r['skill_evidence'] != 'reviewed'
                    or not r['judge_controls_reviewed'] or counts[role] >= QUOTAS[role]):
                raise ValueError(f'Invalid strict review: {i}')
            accepted[i] = r
            prompts.add(prompt)
            counts[role] += 1
        for folder in self.tasks.iterdir():
            if folder.name not in self.done and (folder / 'outcome.json').is_file():
                self.done[folder.name] = read_rank_status(folder / 'outcome.json')
        # Accept only finished tasks with both a checked judge and a reviewed skill.
        # A stable completion order caps each role even when the last ranks finish together.
        for result in sorted(self.done.values(), key=lambda r: (r['finished'], r['task_index'])):
            original = result['candidate']
            row = by_id.get(original['task_index'])
            if row is None:
                continue
            for field in ('task_index', 'role', 'prompt', 'purpose', 'prior_job', 'prior_raw_rewards'):
                if row.get(field) != original.get(field):
                    raise ValueError(f'Claimed candidate changed: {original["task_index"]} ({field})')
            i, role, prompt = row['task_index'], row['role'], normalized(row['prompt'])
            if i in accepted or prompt in prompts or counts[role] >= QUOTAS[role] or not eligible(row, result):
                continue
            rewards = result['raw_rewards']
            selection_rewards = row.get('prior_raw_rewards', []) + rewards
            accepted[i] = dict(row, selection_role=role, reviewed_role=role, skill_evidence='reviewed',
                               judge_controls_reviewed=True, valid_trials=4, raw_rewards=rewards,
                               full_credit_trials=rewards.count(1), frontier_rate=rewards.count(1)/4,
                               selection_rewards=selection_rewards,
                               selection_full_credit_rate=selection_rewards.count(1) / len(selection_rewards),
                               matched_skill=row['reviewed_skill'], judge_controls=row['offline_controls'],
                               initial_state_reward=0, source=str(self.directory / 'tasks' / i))
            prompts.add(prompt)
            counts[role] += 1
        pending = [r for r in rows if counts[r['role']] < QUOTAS[r['role']]
                   and r['task_index'] not in accepted and normalized(r['prompt']) not in prompts]
        return pending, accepted, {k: QUOTAS[k]-counts[k] for k in QUOTAS}

    def claim(self, rank):
        rows, _, _ = self.snapshot()
        for row in rows:
            folder = self.tasks / row['task_index']
            try:
                folder.mkdir()  # One owner across all hosts, no shared lock or clock lease.
            except FileExistsError:
                continue
            save(folder / 'claim.json', {'rank': rank, 'candidate': row})
            return row
        return None

    def next_task(self, rank):
        deadline = time.monotonic() + int(os.getenv('CUA_FRONTIER_JOIN_SECONDS', '86400'))
        while True:
            row = self.claim(rank)
            if row is not None:
                return row
            pending, _, shortfall = self.snapshot()
            claimed = {p.name for p in self.tasks.iterdir()}
            if any(r['task_index'] not in claimed for r in pending):
                continue  # Roster was replenished between snapshots.
            if not any(shortfall.values()) or claimed == self.done.keys():
                return None
            for path in self.directory.glob('rank-*/status.json'):
                state = read_rank_status(path, missing_ok=True, poll=True)
                if state and state['status'] == 'failed':
                    raise RuntimeError(f"Screening rank failed at {path}: {state.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Screening tasks have no outcome: {sorted(claimed - self.done.keys())}')
            time.sleep(5)

    def publish(self, final=False):
        pending, accepted, deficits = self.snapshot()
        claimed = {p.name for p in self.tasks.iterdir()}
        remaining = sum(r['task_index'] not in claimed for r in pending)
        status = 'running'
        if final:
            if claimed != self.done.keys():
                raise ValueError('Unfinished claims at final publication')
            if remaining:
                return None  # A new batch appeared while the last workers were draining.
            status = 'quotas_met' if not any(deficits.values()) else 'candidates_exhausted'
        result = {'status': status, 'strict_total': len(accepted), 'shortfall': deficits,
                  'claimed': len(claimed), 'completed': len(self.done), 'available': remaining,
                  'in_flight': len(claimed - self.done.keys()),
                  'valid_trials': sum(sum(v is not None for v in r['raw_rewards']) for r in self.done.values())}
        write_lines(self.directory / 'prompts-reviewed.jsonl', accepted.values())
        save(self.directory / 'progress.json', result)
        if final:
            result['results'] = self.done
            save(self.directory / 'summary.json', result)
        return result


def eligible(row, result):
    rewards = result['raw_rewards']
    if (not row.get('reviewed_skill') or not controls_pass(row.get('offline_controls', {}))
            or result['initial_state']['status'] != 'passed' or len(rewards) != 4 or None in rewards):
        return False
    all_trials = row.get('prior_raw_rewards', []) + rewards
    return row['role'] == 'breadth' or 0 < all_trials.count(1) < len(all_trials)


def earlier_jobs(root, job, reuse_from):
    earlier = sorted((p for p in root.iterdir() if p.name.isdigit() and int(p.name) < int(job)
                      and (p / 'campaign.json').is_file()), key=lambda p: int(p.name), reverse=True)
    if reuse_from:
        if not any(p.name == reuse_from for p in earlier):
            raise ValueError('--reuse-from must name an earlier screen-expand queue job')
        earlier = [p for p in earlier if int(p.name) <= int(reuse_from)]
    for p in sorted(root.glob('notebook-*')):
        if (p / 'summary.json').is_file() and read(p / 'summary.json').get('status') == 'notebook_complete':
            earlier.append(p)
    return earlier


def runtime_identity():
    cache = ROOT.parent.parent / 'ENV/cache'
    identity = {k: read(p)['sha256'] for k, p in {
        'worker': Path(os.getenv('CUA_WORKER_MANIFEST', cache / 'ais-osworld-worker/manifest.json')),
        'policy': Path(os.getenv('CUA_POLICY_MANIFEST', cache / 'ais-frontier-vllm/manifest.json')),
        'fixtures': cache / 'ais-sft-osworld-inputs/manifest.json'}.items()}
    return dict(identity, model_revision=read(ROOT / 'configs/sources.lock.json')['sft_model']['revision'],
                initial_state_check='untouched-done-v1')


def complete_task(folder, row, history, runtime, collect):
    i = row['task_index']
    if row['purpose'] == 'additional_four':
        previous = read(OLD_RUNS / row['prior_job'] / 'summary.json')['results'][i]
        if previous['raw_rewards'] != row['prior_raw_rewards'] or previous['full_credit_rate'] != 0:
            raise ValueError(f'Supplemental trials differ from the original four results: {i}')
    indices = folder / 'indices.json'
    save(indices, [i])
    protocol = expected_protocol(indices)
    cohort = dict(runtime, tasks=[i], purpose=row['purpose'], prior_job=row.get('prior_job'))
    save(folder / 'cohort.json', cohort)
    reused = 0
    gate = folder / 'episodes' / i / 'initial-state.json'
    reused_from = {}
    for previous in history:
        source = previous / 'tasks' / i
        if not (source / 'episodes/protocol.json').is_file():
            continue
        kept = restore(source, folder, [i], protocol, cohort, merge_existing=True)
        if kept:
            reused_from[previous.name] = kept
        reused += kept
        if reused == 4 or (gate.is_file() and read(gate)['status'] == 'nonzero_initial_reward'):
            break
    rejected = gate.is_file() and read(gate)['status'] == 'nonzero_initial_reward'
    protocol_file = folder / 'episodes/protocol.json'
    if reused < 4 and not rejected:
        expected = folder / 'expected-protocol.json'
        save(expected, protocol)
        collect(folder, indices, reused_protocol=expected, check_initial_state=True)
    else:
        save(protocol_file, protocol)
    require_equivalent_protocol(read(protocol_file), protocol)
    result = coverage(folder, [i], protocol, overnight=True)[i]
    result.update(task_index=i, candidate=row, finished=time.time(), reused_slots=reused, reused_from=reused_from,
                  eligible=eligible(row, result))
    save(folder / 'outcome.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--reuse-from', metavar='JOB_ID')
    parser.add_argument('--desktops-per-rank', type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    training = sorted({i for ids in read(ROOT / 'configs/research-splits/train-examples.json').values() for i in ids})
    rows = records(ROSTER)
    previous = previous_batches()
    catalog_path = AUDIT / 'index.jsonl'
    catalog = {r['id']: r for r in records(catalog_path)} if catalog_path.is_file() else None
    check_manifest(rows, set(training), previous, catalog)
    if args.plan:
        print(json.dumps({'tasks': len(rows), 'trials': 4*len(rows), 'ranks': 8, 'desktops_per_rank': args.desktops_per_rank,
                          'scheduling': 'shared_queue', 'check_initial_state': True, 'targets': QUOTAS,
                          'purpose': dict(Counter(r['purpose'] for r in rows)),
                          'reviewed_candidates': sum(bool(r.get('reviewed_skill')) for r in rows),
                          'strict_at_start': dict(Counter(r['selection_role'] for r in records(REVIEWED)))}, indent=2))
        return
    job = os.environ.get('JOB_ID', '')
    rank = int(os.environ.get('RANK', '-1'))
    if not job.isdigit() or rank not in range(8) or os.environ.get('WORLD_SIZE') != '8':
        parser.error('screen-expand requires one AIS job with eight single-GPU ranks')
    directory = RUNS / job
    out = directory / f'rank-{rank}'
    out.mkdir(parents=True, exist_ok=False)
    status = {'rank': rank, 'status': 'starting'}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    stop_report = threading.Event()
    try:
        if catalog is None:
            raise ValueError(f'Missing training/holdout audit at {catalog_path}')
        queue = ScreenQueue(directory, set(training), ROSTER, REVIEWED, previous, catalog)
        if rank == 0:
            prior = earlier_jobs(RUNS, job, args.reuse_from)
            save(directory / 'campaign.json', {'history': [p.name for p in prior], 'runtime': runtime_identity(),
                                               'initial_candidates': [r['task_index'] for r in rows]})
        campaign = read_rank_status(directory / 'campaign.json')
        runtime = runtime_identity()
        if runtime != campaign['runtime']:
            raise ValueError('Screening runtime differs between AIS ranks')
        history = [RUNS / n for n in campaign['history'] if read(RUNS / n / 'campaign.json')['runtime'] == runtime]
        if args.reuse_from and RUNS / args.reuse_from not in history:
            raise ValueError('Requested job has a different runtime')
        def report():
            monitor = ScreenQueue(directory, set(training), ROSTER, REVIEWED, previous, catalog)
            while not stop_report.is_set():
                try:
                    monitor.publish()
                except (OSError, ValueError) as exc:
                    print(f'Progress snapshot will retry: {exc}', flush=True)
                stop_report.wait(30)
        if rank == 0:
            reporter = threading.Thread(target=report, daemon=True)
            reporter.start()
        indices = out / 'registered-indices.json'
        save(indices, training)
        with ExitStack() as stack:
            collector = None
            def collect(*args_, **kwargs):
                nonlocal collector
                if collector is None:
                    collector = stack.enter_context(collector_services(out, indices, set(training), status, args.desktops_per_rank))
                collector(*args_, **kwargs)
            while True:
                while (row := queue.next_task(rank)) is not None:
                    i = row['task_index']
                    status.update(status='collecting', current_task=i)
                    save(out / 'status.json', status)
                    print(f'Screening rank {rank} claimed {i}', flush=True)
                    complete_task(queue.tasks / i, row, history, runtime, collect)
                status.update(status='shard_complete', current_task=None)
                save(out / 'status.json', status)
                if rank != 0:
                    break
                wait_for_shards(directory)
                stop_report.set()
                reporter.join()
                summary = queue.publish(final=True)
                if summary:
                    print(f"Screening {summary['status']}: {summary['strict_total']}/256 strict. {directory}", flush=True)
                    break
    except BaseException as exc:
        save(out / 'status.json', dict(status, status='failed', error=repr(exc)))
        raise
    finally:
        stop_report.set()


if __name__ == '__main__':
    main()
