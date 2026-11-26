"""Wait for a specific eight-rank screen, verify archived trials and export quota-capped training prompts."""
import argparse
from collections import Counter
import errno
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from ais_dataset_screen import checked_score, expected_protocol, validate_control
from ais_frontier_shards import read, require_equivalent_protocol, verify_attempt
from ais_rank_status import read_rank_status
from ais_screen_queue import ARCHIVE_REVIEW, CONFIG, OLD_RUNS, QUOTAS, REVIEWED, ROOT, ROSTER, RUNS, controls_pass, records
from ais_vm_check import save
from build_overnight_screen import AUDIT, normalized
from overnight_prompts import write_lines

CONTROL_CASES = CONFIG / 'night-review-controls.jsonl'
TERMINAL = {'quotas_met', 'candidates_exhausted'}


def upstream_state(directory):
    summary = read_rank_status(directory / 'summary.json', missing_ok=True, poll=True)
    if summary and summary.get('status') in TERMINAL:
        return summary['status']
    for status in directory.glob('rank-*/status.json'):
        state = read_rank_status(status, missing_ok=True, poll=True)
        if state and state.get('status') == 'failed':
            return 'upstream_failed'
    return None


def published_selection(directory, expected):
    path = directory / 'prompts-reviewed.jsonl'
    for delay in (1, 3, 8, 15, 0):
        try:
            rows = records(path)
            if len(rows) == expected:
                return rows
        except (OSError, json.JSONDecodeError) as exc:
            if isinstance(exc, OSError) and exc.errno not in (errno.ENOENT, errno.EACCES, errno.ESTALE):
                raise
        if delay:
            time.sleep(delay)
    raise ValueError(f'Expected {expected} published prompts in {path}')


def official_prompt(i, catalog, hashes, train):
    row = catalog[i]
    domain = row['domain']
    name = f'osworld/rl_tasks/{domain}/{i.removeprefix(domain + "_")}.json'
    raw = (ROOT / 'data/ScaleCUA' / name).read_bytes()
    if (i not in train or hashlib.sha256(raw).hexdigest() != hashes[name] or
            row['heldout_instruction'] or row['shared_setup_inputs'] or
            json.loads(raw)['instruction'] != row['instruction']):
        raise ValueError(f'Official training source or holdout mismatch for {i}')
    return row['instruction']


@lru_cache(None)
def pilot_shard(lane):
    audit = read(lane.parent / 'verified/audit.json')['ranks'][int(lane.name.split('-')[1])]
    for name, key in [('protocol.json', 'protocol_sha256'), ('results.jsonl', 'results_sha256')]:
        if hashlib.sha256((lane / 'episodes' / name).read_bytes()).hexdigest() != audit[key]:
            raise ValueError(f'Pilot receipt differs from sealed audit at {lane}')
    return {(r['task_index'], r['run_number']): r for r in records(lane / 'episodes/results.jsonl')}


@lru_cache(None)
def old_episode_dirs(source):
    kind, job = source.split('-')
    if not job.isdigit() or kind not in ('pilot', 'screen'):
        raise ValueError(f'Invalid historical source {source}')
    root = ROOT / 'runs/ais-frontier-campaigns' / job if kind == 'pilot' else OLD_RUNS / job
    dirs = {}
    for episodes in root.glob('rank-*/episodes'):
        for task in episodes.iterdir():
            if '_traj_verify_' in task.name and task.is_dir():
                if task.name in dirs:
                    raise ValueError(f'Duplicate archived task {task.name} in {source}')
                dirs[task.name] = task
    return dirs


def verify_trials(i, folder, raw, pilot=False, require_initial=False):
    for delay in (1, 3, 0):
        try:
            return _verify_trials(i, folder, raw, pilot, require_initial)
        except (OSError, json.JSONDecodeError) as exc:
            if not delay or isinstance(exc, OSError) and exc.errno not in (errno.EACCES, errno.ESTALE):
                raise
            time.sleep(delay)


def _verify_trials(i, folder, raw, pilot, require_initial):
    episodes = folder.parent
    protocol = read(episodes / 'protocol.json')
    if pilot:
        signed = pilot_shard(episodes.parent)
    else:
        require_equivalent_protocol(protocol, expected_protocol(episodes.parent / 'indices.json'))
    gate = folder / 'initial-state.json'
    if gate.is_file() or require_initial:
        receipt = read(gate)
        validate_control(receipt, i)
        if receipt['status'] != 'passed':
            return 'initial_state_' + receipt['status']
    if len(raw) != 4 or any(v is None for v in raw):
        return 'needs_missing_trials'
    for n, value in enumerate(raw, 1):
        slot = folder / str(n)
        if pilot:
            row = read(slot / 'result.json')
            if row != signed[i, n]:
                raise ValueError(f'Sealed trial mismatch at {slot}')
            verify_attempt(slot, row, protocol)
        else:
            row = checked_score(slot, i, n, protocol)
        if row['raw_reward'] != value:
            raise ValueError(f'Reward mismatch at {slot}')
    return None


def review(directory, out, trigger):
    catalog = {r['id']: r for r in records(AUDIT / 'index.jsonl')}
    train = {i for ids in read(ROOT / 'configs/research-splits/train-examples.json').values() for i in ids}
    manifest = read(ROOT / 'reports/training-data-manifest.json')
    if manifest['revision'] != read(ROOT / 'configs/sources.lock.json')['training_data']['revision']:
        raise ValueError('Official dataset revision mismatch')
    hashes = {f['name']: f['sha256'] for f in manifest['files']}
    controls = {r['task_index']: r for r in records(CONTROL_CASES)}
    roster = {r['task_index']:r for r in records(ROSTER)}
    archive = {r['task_index']:r for r in records(ARCHIVE_REVIEW)}
    source = read_rank_status(directory / 'summary.json', missing_ok=trigger not in TERMINAL)
    if source and source['status'] in TERMINAL:
        done = source['results']
    else:
        done = {p.parent.name: read_rank_status(p) for p in (directory / 'tasks').glob('*/outcome.json')}
    frozen = records(REVIEWED)
    published = published_selection(directory, source['strict_total']) if source else []
    if source and len(done) != source['completed']:
        raise ValueError('Screening summary and completed task count disagree')
    accepted = {}
    prompts = set()
    counts = Counter()

    def admit(row, role):
        i = row['task_index']
        prompt = official_prompt(i, catalog, hashes, train)
        if row['prompt'] != prompt:
            raise ValueError(f'Changed prompt {i}')
        if normalized(prompt) in prompts or counts[role] >= QUOTAS[role]:
            return False
        accepted[i] = row
        prompts.add(normalized(prompt))
        counts[role] += 1
        return True

    for row in frozen:
        if (row['valid_trials'] != 4 or row['skill_evidence'] != 'reviewed' or
                not row['judge_controls_reviewed'] or not admit(row, row['selection_role'])):
            raise ValueError(f'Invalid frozen row {row["task_index"]}')
    # Keep the screen's existing selection order when reviewing current outcomes.
    order = dict.fromkeys([r['task_index'] for r in published] + list(done) + list(archive))
    audits = []
    for i in order:
        if i in accepted:
            continue
        if i not in done and i not in archive:
            raise ValueError(f'Published row lacks trial source: {i}')
        prompt = official_prompt(i, catalog, hashes, train)
        if i in done:
            result = done[i]
            row = roster[i]
            if any(row.get(k) != result['candidate'].get(k) for k in
                   ('task_index','role','prompt','purpose','prior_job','prior_raw_rewards')):
                raise ValueError(f'Claim changed for {i}')
            raw = result['raw_rewards']
            folder = directory / 'tasks' / i / 'episodes' / i
        else:
            row = archive[i]
            raw = row['raw_rewards']
            folder = old_episode_dirs(row['source']).get(i)
        role = row['role']
        case = controls.get(i, {})
        scores = case.get('scores', row.get('offline_controls', {}))
        skill = case.get('skill', row.get('reviewed_skill'))
        verdict = None
        try:
            if folder is None:
                raise ValueError('Missing archived episode directory')
            pilot = i not in done and row['source'].startswith('pilot-')
            verdict = verify_trials(i, folder, raw, pilot, i in done or row.get('initial_state') == 'passed')
            prior = row.get('prior_raw_rewards', [])
            if prior and not verdict:
                old = old_episode_dirs('screen-' + row['prior_job'])[i]
                verdict = verify_trials(i, old, prior)
            if not verdict and case and case['role'] != role:
                raise ValueError('Role differs from offline case review')
        except (KeyError, OSError, ValueError, IndexError) as exc:
            verdict = 'trial_evidence_unverified'
            error = str(exc)
        review_row = dict(task_index=i, prompt=prompt, role=role, raw_rewards=raw, source=str(folder))
        if verdict == 'trial_evidence_unverified':
            review_row['error'] = error
        all_scores = row.get('prior_raw_rewards', []) + raw
        if not verdict and scores and not controls_pass(scores):
            verdict = 'judge_failed_controls'
        if not verdict and role != 'breadth':
            if 1 not in all_scores:
                verdict = 'review_solvability' if len(all_scores) == 4 else 'no_full_credit_after_eight'
            elif all(v == 1 for v in all_scores):
                verdict = 'all_full_credit'
        if not verdict and (not skill or not controls_pass(scores)):
            verdict = 'needs_skill_or_judge_controls'
        if not verdict:
            item = catalog[i]
            if role == 'main' and (item['seed_score'] is None or item['seed_score'] >= 1):
                raise ValueError(f'Main candidate has no scored seed weakness: {i}')
            selected = dict(row, prompt=prompt, selection_role=role, reviewed_role=role, app=item['domain'],
                            base_id=item['base'], seed_score=item['seed_score'], raw_rewards=raw,
                            source=str(folder), skill_evidence='reviewed', matched_skill=skill,
                            judge_controls_reviewed=True, judge_controls=scores, initial_state_reward=0,
                            initial_state_evidence='desktop' if (folder/'initial-state.json').is_file() else 'offline_control',
                            valid_trials=4, full_credit_trials=raw.count(1), frontier_rate=raw.count(1)/4,
                            selection_rewards=all_scores, selection_full_credit_rate=all_scores.count(1)/len(all_scores))
            verdict = 'accepted' if admit(selected, role) else 'quota_or_duplicate'
        review_row['verdict'] = verdict
        if case:
            review_row['judge_controls'] = scores
        audits.append(review_row)
        if len(audits) % 20 == 0:
            save(out/'status.json', {'status':'reviewing', 'upstream':trigger, 'checked':len(audits),
                                     'strict_total':len(accepted), 'by_role':dict(counts)})
    deficits = {role:cap-counts[role] for role,cap in QUOTAS.items()}
    write_lines(out/'prompts-reviewed.jsonl', accepted.values())
    write_lines(out/'review-decisions.jsonl', audits)
    save(out/'train-indices.json', list(accepted))
    result = dict(status='finished', upstream=trigger, strict_total=len(accepted), by_role=dict(counts),
                  shortfall=deficits, source_job=directory.name, dataset_revision=manifest['revision'],
                  review_decisions=dict(Counter(r['verdict'] for r in audits)),
                  carried_from_frozen=len(frozen), outcomes=len(done), archived_candidates=len(archive))
    save(out/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', help='Existing screen-expand job ID')
    args = parser.parse_args()
    directory = RUNS / args.job
    if not args.job.isdecimal() or not (directory/'campaign.json').is_file():
        parser.error('Expected an existing screen-expand job ID')
    out = directory / 'night-review'
    out.mkdir(exist_ok=True)
    with (out/'review.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Review already queued for {args.job}')
            return
        try:
            save(out/'status.json', dict(status='waiting_for_screen', job=args.job, pid=os.getpid(), host=socket.gethostname()))
            while (trigger := upstream_state(directory)) is None:
                time.sleep(30)
            save(out/'status.json', dict(status='reviewing', upstream=trigger, job=args.job))
            print(json.dumps(review(directory, out, trigger), ensure_ascii=False), flush=True)
            save(out/'status.json', read(out/'summary.json'))
        except Exception as exc:
            save(out/'status.json', dict(status='failed', job=args.job, error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
