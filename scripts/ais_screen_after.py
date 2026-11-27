"""Review a completed follow-up screen and schedule only its missing episode slots."""
from collections import Counter
import hashlib
import os
from pathlib import Path
import time

from ais_full_screen_pool import extend, load_inventory
from ais_dataset_screen import coverage, expected_protocol
from ais_frontier_shards import read, require_equivalent_protocol
from ais_rank_status import read_rank_status
from ais_screen_followup import BUNDLE, RUN_ROOT, historical_rewards, score_cases, verified_json
from ais_screen_queue import CONFIG, QUOTAS, ROOT, check_manifest, controls_pass, eligible, records, selected_row
from ais_vm_check import save
from ais_worker_cache import stage_worker
from build_overnight_screen import normalized
from overnight_prompts import write_lines
from review_screen_job import TERMINAL, published_selection, upstream_state


def ready(upstream, *, wait):
    deadline = time.monotonic() + int(os.getenv('CUA_FRONTIER_JOIN_SECONDS', '86400'))
    while True:
        state = upstream_state(upstream)
        if state in TERMINAL:
            return True
        if state == 'upstream_failed':
            raise RuntimeError(f'Upstream screen {upstream.name} failed before final export')
        if not wait:
            return False
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Waiting for final export from {upstream.name}')
        time.sleep(10)


def audit(folder, i):
    protocol = expected_protocol(folder / 'indices.json')
    require_equivalent_protocol(read(folder / 'episodes/protocol.json'), protocol)
    return coverage(folder, [i], protocol, overnight=True)[i]


def full_pool_source(upstream, current_job=None):
    anchor = read(CONFIG / 'full-pool.json')['source_job']
    state = read_rank_status(upstream / 'campaign.json')
    if upstream.name != anchor and state.get('upstream_job') != anchor and not state.get('full_pool'):
        return upstream, False
    # Reusing the original Entry after a completed sweep must not sample its tasks again.
    paths = sorted((p for p in upstream.parent.iterdir() if p.name.isdecimal()
                    and int(p.name) > int(upstream.name)
                    and (current_job is None or int(p.name) < int(current_job))), key=lambda p: int(p.name))
    for p in paths:
        record = read_rank_status(p / 'campaign.json', missing_ok=True, poll=True)
        if record and record.get('upstream_job') == upstream.name and upstream_state(p) in TERMINAL:
            upstream = p
    return upstream, True


def sampling_history(upstream, job):
    # A preempted attempt may already have written valid episodes, even without a final selection.
    attempts = []
    for p in upstream.parent.iterdir():
        if not p.name.isdecimal() or not int(upstream.name) < int(p.name) < int(job):
            continue
        record = read_rank_status(p / 'campaign.json', missing_ok=True)
        if record and record.get('upstream_job') == upstream.name:
            attempts.append(p)
    return sorted(attempts, key=lambda p: int(p.name), reverse=True) + [upstream]


def prepare(out, upstream, *, full_pool=False):
    source = read_rank_status(upstream / 'summary.json')
    campaign = read_rank_status(upstream / 'campaign.json')
    manifest = read(BUNDLE / 'manifest.json')
    if source['status'] not in TERMINAL or campaign['runtime'] != manifest['runtime']:
        raise ValueError('Upstream screen is unfinished or has a different runtime')
    if manifest['revision'] != read(ROOT / 'configs/sources.lock.json')['training_data']['revision']:
        raise ValueError('Training task revision changed')
    for name in ('candidates.jsonl', 'protocol.json'):
        if hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError(f'Frozen review input changed: {name}')
    manifest_path = Path(os.getenv('CUA_WORKER_MANIFEST', ROOT.parent.parent / 'ENV/cache/ais-osworld-worker/manifest.json'))
    if read(manifest_path)['sha256'] != manifest['runtime']['worker']:
        raise ValueError('Worker differs from the upstream job')
    worker, _ = stage_worker(manifest_path)
    splits = ROOT / 'configs/research-splits'
    domains = {i: app for app, ids in read(splits / 'train-examples.json').items() for i in ids}
    tasks = worker / 'data/ScaleCUA/osworld/rl_tasks'
    heldout_prompts = {normalized(read(tasks / app / (i.removeprefix(app + '_') + '.json'))['instruction'])
                       for split in ('dev', 'test') for app, ids in read(splits / (split + '-examples.json')).items()
                       for i in ids}
    inputs = {r['task_index']: r for r in records(BUNDLE / 'candidates.jsonl')}
    inventory = None
    prior_ids = set()
    if full_pool or campaign.get('full_pool'):
        inventory = load_inventory(domains, manifest['revision'])
        ancestor = upstream
        while ancestor.name != inventory['source_job']:
            parent = read_rank_status(ancestor / 'campaign.json')
            prior_ids.update(parent['initial_candidates'])
            previous = parent.get('upstream_job', '')
            if not previous.isdecimal() or int(previous) >= int(ancestor.name):
                raise ValueError(f'Full-pool inventory needs a descendant of {inventory["source_job"]}')
            ancestor = upstream.parent / previous
        inputs.update(inventory['candidates'])
    cases = {r['task_index']: r for r in records(CONFIG / 'after-review-controls.jsonl')}
    baseline = published_selection(upstream, source['strict_total'])
    admitted = {r['task_index'] for r in baseline}
    if [r['task_index'] for r in baseline] != read(upstream / 'train-indices.json'):
        raise ValueError('Upstream training IDs differ from the published selection')
    counts = Counter(r['selection_role'] for r in baseline)
    prompts = {normalized(r['prompt']) for r in baseline}
    if {role: limit - counts[role] for role, limit in QUOTAS.items()} != source['shortfall']:
        raise ValueError('Role counts disagree with upstream selection')
    done = source['results']
    roster = {r['task_index']: r for r in campaign['fixed_inputs'][0]}
    if set(done) - roster.keys() or len(done) != source['completed'] or source['in_flight']:
        raise ValueError('Upstream results disagree with frozen roster or completion count')
    pending = {r['task_index']: r for r in records(upstream / 'scored-for-review.jsonl')}
    queue, carry, decisions = [], [], []
    slots = 0
    for i in dict.fromkeys([*roster, *pending]):
        if i in roster:
            row = dict(roster[i])
            folder = upstream / 'tasks' / i
            if i not in done:
                if counts[row['role']] < QUOTAS[row['role']] and i not in admitted and normalized(row['prompt']) not in prompts:
                    raise ValueError(f'Missing final outcome for {i}')
                continue
            result = audit(folder, i)
            if done[i]['candidate'] != row or any(result[k] != done[i][k] for k in result):
                raise ValueError(f'Trial evidence differs from published outcome: {i}')
        else:
            record = pending[i]
            evidence = inputs[i]
            folder = Path(record['source'])
            row = dict(record, purpose='first_four', app=domains[i], reviewed_skill=None, offline_controls={})
            if folder.is_absolute():
                if folder.parents[2] != RUN_ROOT or folder.parent.name != 'tasks' or folder.name != i:
                    raise ValueError(f'Invalid carried evidence path: {folder}')
                result = audit(folder, i)
            else:
                if record['source'] != evidence['source'] or record['role'] != evidence['role']:
                    raise ValueError(f'Historical review source changed: {i}')
                result = dict(raw_rewards=historical_rewards(evidence), initial_state=evidence['receipt']['gate'])
        raw = result['raw_rewards']
        if i in pending and (raw != pending[i]['raw_rewards'] or row['prompt'] != pending[i]['prompt']
                             or row['role'] != pending[i]['role']):
            raise ValueError(f'Pending candidate differs from audited source: {i}')
        role = row['role']
        key = normalized(row['prompt'])
        missing = raw.count(None)
        gate = result['initial_state']['status']
        decision = dict(task_index=i, role=role, missing_slots=missing, initial_state=gate)
        quota_full = counts[role] >= QUOTAS[role]
        retry_for_pool = row.get('full_pool_candidate') and len(baseline) < sum(QUOTAS.values()) and missing
        if i in admitted or key in prompts or (quota_full and not retry_for_pool):
            verdict = 'already_selected_or_quota'
        elif gate == 'nonzero_initial_reward':
            verdict = 'nonzero_initial_reward'
        else:
            task = verified_json(tasks / domains[i] / (i.removeprefix(domains[i] + '_') + '.json'), inputs[i]['sha256'])
            if task['instruction'] != row['prompt']:
                raise ValueError(f'Official prompt differs: {i}')
            case = cases.get(i)
            if case:
                if inputs[i]['sha256'] != case['task_sha256'] or role != case.get('role', 'main'):
                    raise ValueError(f'Offline review belongs to another task or role: {i}')
                scores = score_cases(worker, case, task['evaluator'])
                row.update(reviewed_skill=case['skill'], offline_controls=scores)
                decision['controls'] = scores
            if role == 'main' and (inputs[i]['seed_score'] is None or inputs[i]['seed_score'] >= 1 or key in heldout_prompts):
                verdict = 'missing_weakness_or_heldout_prompt'
            elif case and not controls_pass(scores):
                verdict = 'judge_failed_controls'
            elif missing:
                verdict = 'retry_missing'
                queue.append(row)
                slots += missing
            elif eligible(row, result):
                verdict = 'accepted_on_cpu'
                baseline.append(selected_row(row, result, folder))
                admitted.add(i)
                prompts.add(key)
                counts[role] += 1
            elif not row.get('reviewed_skill'):
                verdict = 'needs_skill_or_judge_controls'
                carry.append(pending[i])
            else:
                verdict = 'outside_reward_criteria'
        decisions.append(dict(decision, decision=verdict))
    queue.sort(key=lambda r: (r['role'] != 'main', not r.get('reviewed_skill')))
    pool_report = dict(retry_tasks=len(queue), missing_trials=slots)
    if inventory:
        prior_ids.update(roster)
        prior_ids.update(pending)
        pool_report.update(extend(inventory, queue, baseline, decisions, prior_ids, domains, heldout_prompts, worker, cases))
        slots += 4 * pool_report['first_time_tasks']
    check_manifest(queue, set(domains))
    write_lines(out / 'roster.jsonl', queue)
    write_lines(out / 'baseline.jsonl', baseline)
    write_lines(out / 'carried-for-review.jsonl', carry)
    write_lines(out / 'cpu-decisions.jsonl', decisions)
    report = dict(upstream_job=upstream.name, upstream_status=source['status'], strict_before=source['strict_total'],
                  strict_after_cpu=len(baseline), by_role=dict(counts), gpu_tasks=len(queue), max_new_trials=slots,
                  decisions=dict(Counter(r['decision'] for r in decisions)), ranks=8, **pool_report)
    save(out / 'cpu-summary.json', report)
    return report, queue, baseline
