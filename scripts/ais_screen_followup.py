"""Run the CPU stages of the follow-up screen on the AIS allocation."""
import __future__
import ast
from collections import Counter
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from ais_dataset_screen import coverage, expected_protocol
from ais_frontier_shards import read, require_equivalent_protocol
from ais_screen_queue import CONFIG, QUOTAS, ROOT, check_manifest, controls_pass, records
from ais_vm_check import save
from ais_worker_cache import stage_worker
from build_overnight_screen import normalized
from overnight_prompts import write_lines

BUNDLE = CONFIG / 'followup'
RUN_ROOT = ROOT / 'runs/ais-screen-followup'
GPU_LIMIT = 48


def verified_json(path, checksum):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError(f'Checksum mismatch: {path}')
    return json.loads(raw)


@lru_cache(maxsize=16)
def functions_in_module(path, checksum):
    # Load just the pinned scoring functions. The full upstream modules import GUI and GPU dependencies.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError(f'Judge checksum mismatch: {path}')
    return {fn.name: fn for fn in ast.parse(raw).body if isinstance(fn, ast.FunctionDef)}


def judge_function(path, checksum, name):
    fn = functions_in_module(path, checksum)[name]
    ns = {'re': re, 'json': json}
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns[name]


def score_cases(worker, case, evaluator):
    score = judge_function(worker / case['module'], case['module_sha256'], evaluator['func'])
    get_state = None
    if 'getter_module' in case:
        get_state = judge_function(worker / case['getter_module'], case['getter_sha256'],
                                   'get_' + evaluator['result']['type'])
    scores = {}
    with redirect_stdout(io.StringIO()):
        for label, state in case['inputs'].items():
            if get_state is not None:
                files = {case['path']: state.encode()}
                desktop = SimpleNamespace(controller=SimpleNamespace(get_file=files.get))
                state = get_state(desktop, evaluator['result'])
            scores[label] = score(state, evaluator['expected']['rules'], **evaluator.get('options', {}))
    return scores

def prepare(out, worker=None):
    manifest = read(BUNDLE / 'manifest.json')
    if manifest['revision'] != read(ROOT / 'configs/sources.lock.json')['training_data']['revision']:
        raise ValueError('Follow-up audit was prepared for another dataset revision')
    for filename, sha in manifest['files'].items():
        if hashlib.sha256((BUNDLE / filename).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Changed follow-up review input: {filename}')
    if worker is None:
        path = Path(os.getenv('CUA_WORKER_MANIFEST', ROOT.parent.parent / 'ENV/cache/ais-osworld-worker/manifest.json'))
        if read(path)['sha256'] != manifest['runtime']['worker']:
            raise ValueError('The worker archive differs from the audited dataset')
        worker, _ = stage_worker(path)
    domains = {i: app for app, ids in read(ROOT / 'configs/research-splits/train-examples.json').items() for i in ids}
    heldout = set(read(ROOT / 'configs/research-splits/dev-indices.json')) | set(read(ROOT / 'configs/research-splits/test-indices.json'))
    source = records(BUNDLE / 'candidates.jsonl')
    baseline = records(BUNDLE / 'baseline.jsonl')
    known = {r['task_index'] for r in baseline + source if r.get('kind') != 'new'}
    known.update(r['task_index'] for name in ('expand-queue', 'expand-review') for r in records(CONFIG / (name + '.jsonl')))
    seen_ids, seen_prompts, accepted, queue, decisions = set(), set(), [], [], []
    counts = Counter()

    def official(r):
        i = r['task_index']
        if i in seen_ids or i not in domains or i in heldout:
            raise ValueError(f'Duplicate or non-training candidate: {i}')
        seen_ids.add(i)
        app = domains[i]
        path = worker / 'data/ScaleCUA/osworld/rl_tasks' / app / (i.removeprefix(app + '_') + '.json')
        return verified_json(path, r['sha256'])

    def admit(row):
        role = row['selection_role']
        prompt = normalized(row['prompt'])
        if counts[role] >= QUOTAS[role] or prompt in seen_prompts:
            return False
        counts[role] += 1
        seen_prompts.add(prompt)
        accepted.append(row)
        return True

    for r in baseline:
        if (official(r)['instruction'] != r['prompt'] or r['valid_trials'] != 4
                or r['skill_evidence'] != 'reviewed' or not r['judge_controls_reviewed'] or not admit(r)):
            raise ValueError(f'Invalid frozen admission: {r["task_index"]}')
    for r in source:
        task = official(r)
        i, kind, role = r['task_index'], r['kind'], r['role']
        prompt = task['instruction']
        verdict = r.get('hold')
        scores = r.get('offline_controls', {})
        skill = r.get('reviewed_skill')
        if 'case' in r:
            skill = r['case']['skill']
            scores = score_cases(worker, r['case'], task['evaluator'])
            if not controls_pass(scores):
                verdict = 'judge_failed_controls'
        if role == 'main' and (r['seed_score'] is None or r['seed_score'] >= 1):
            verdict = 'no_measured_seed_weakness'
        if not verdict:
            if not skill or not controls_pass(scores):
                raise ValueError(f'Missing judge controls for {i}')
            row = dict(task_index=i, prompt=prompt, role=role, reviewed_skill=skill,
                       offline_controls=scores, base_id=task['id'], app=domains[i], seed_score=r['seed_score'])
            if kind != 'new':
                # These are portable receipts from the previous completed CPU trajectory audit.
                receipt = r['receipt']
                raw = r['raw_rewards']
                gate = receipt['gate']
                if (gate['task_index'] != i or gate['status'] != 'passed' or gate['reward'] != 0 or
                        not gate['session_id'] or len(raw) != 4 or len(set(receipt['results'])) != 4 or
                        any(type(v) not in (int, float) or not 0 <= v <= 1 for v in raw)):
                    raise ValueError(f'Invalid historical scores or initial state for {i}')
            if kind == 'scored':
                if role != 'breadth' and not 0 < raw.count(1) < 4:
                    raise ValueError(f'Expected mixed historical rewards for {i}')
                selected = dict(row, selection_role=role, reviewed_role=role, skill_evidence='reviewed',
                                judge_controls_reviewed=True, judge_controls=scores, matched_skill=skill,
                                initial_state_reward=0, raw_rewards=raw, valid_trials=4, full_credit_trials=raw.count(1),
                                frontier_rate=raw.count(1)/4, source=r['source'])
                verdict = 'accepted_on_cpu' if admit(selected) else 'quota_or_duplicate'
            else:
                if kind == 'partial':
                    if not (r['source'].startswith('ais-screen-expand/' + manifest['source_job'] + '/tasks/') and
                            0 < max(raw) < 1):
                        raise ValueError(f'Invalid supplemental history: {i}')
                    row.update(purpose='additional_four', prior_job=manifest['source_job'], prior_raw_rewards=raw)
                else:
                    row['purpose'] = 'first_four'
                verdict = 'gpu_queue'
                if normalized(prompt) in seen_prompts or counts[role] >= QUOTAS[role]:
                    verdict = 'quota_or_duplicate'
                elif len(queue) >= GPU_LIMIT:
                    verdict = 'gpu_limit'
                else:
                    queue.append(row)
        decisions.append(dict(task_index=i, kind=kind, decision=verdict, controls=scores))
    check_manifest(queue, set(domains), known)
    write_lines(out / 'roster.jsonl', queue)
    write_lines(out / 'baseline.jsonl', accepted)
    write_lines(out / 'cpu-decisions.jsonl', decisions)
    summary = dict(inputs=dict(Counter(r['kind'] for r in source)), strict_before=len(baseline),
                   strict_after_cpu=len(accepted), by_role=dict(counts),
                   decisions=dict(Counter(r['decision'] for r in decisions)), gpu_limit=GPU_LIMIT,
                   gpu_tasks=len(queue), gpu_purpose=dict(Counter(r['purpose'] for r in queue)),
                   max_new_trials=4*len(queue), ranks=8)
    save(out / 'cpu-summary.json', summary)
    return summary, queue, accepted


def finish(queue):
    # Re-read the new traces after the eight collectors have closed, before exporting training IDs.
    for i, result in queue.done.items():
        folder = queue.tasks / i
        protocol = expected_protocol(folder / 'indices.json')
        require_equivalent_protocol(read(folder / 'episodes/protocol.json'), protocol)
        audited = coverage(folder, [i], protocol, overnight=True)[i]
        if any(audited[k] != result[k] for k in ('raw_rewards', 'initial_state', 'unscored')):
            raise ValueError(f'Final CPU review differs from collected trials: {i}')
    summary = queue.publish(final=True, export_indices=True)
    if summary is None:
        raise ValueError('Follow-up queue changed during final review')
    return summary
