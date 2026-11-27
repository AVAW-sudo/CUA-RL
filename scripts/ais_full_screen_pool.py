"""Extend the follow-up queue to the remaining official training prompts."""
import hashlib
from collections import Counter

from ais_frontier_shards import read
from ais_screen_followup import verified_json, score_cases
from ais_screen_queue import CONFIG, ROOT, QUOTAS, controls_pass
from build_overnight_screen import normalized


def load_inventory(training, revision):
    data = read(CONFIG / 'full-pool.json')
    path = ROOT / 'configs/research-splits/train-indices.json'
    if data['revision'] != revision or hashlib.sha256(path.read_bytes()).hexdigest() != data['train_sha256']:
        raise ValueError('Full-pool inventory belongs to a different training release or split')
    ids = [*data['candidates'], *(i for group in data['excluded'].values() for i in group)]
    if len(ids) != len(set(ids)) or set(ids) != set(training):
        raise ValueError('Full-pool inventory must account for each training ID once')
    return data


def extend(inventory, rows, baseline, decisions, prior_ids, domains, heldout, worker, cases):
    # The training split is frozen. Only the review/selection state changes between jobs.
    prompts = {normalized(r['prompt']) for r in baseline + rows}
    expanded = []
    if len(baseline) < sum(QUOTAS.values()):
        for i, evidence in inventory['candidates'].items():
            if i in prior_ids:
                continue
            app = domains[i]
            task_path = worker / 'data/ScaleCUA/osworld/rl_tasks' / app / (i.removeprefix(app + '_') + '.json')
            task = verified_json(task_path, evidence['sha256'])
            text = normalized(task['instruction'])
            if text in heldout:
                raise ValueError(f'Full-pool candidate repeats a held-out instruction: {i}')
            if text in prompts:
                continue
            row = dict(task_index=i, prompt=task['instruction'], app=app, base_id=task['id'],
                       role=evidence['role'], seed_score=evidence['seed_score'], shared_setup_inputs=evidence['shared_setup_inputs'],
                       purpose='first_four', full_pool_candidate=True, reviewed_skill=None, offline_controls={})
            case = cases.get(i)
            if case:
                if case['task_sha256'] != evidence['sha256'] or case.get('role', 'main') != row['role']:
                    raise ValueError(f'Judge control belongs to another task or role: {i}')
                scores = score_cases(worker, case, task['evaluator'])
                if not controls_pass(scores):
                    decisions.append(dict(task_index=i, role=row['role'], decision='judge_failed_controls', controls=scores))
                    continue
                row.update(reviewed_skill=case['skill'], offline_controls=scores)
            expanded.append(row)
            prompts.add(text)
    expanded.sort(key=lambda r: (('main', 'component', 'breadth').index(r['role']), not r['reviewed_skill'], r['task_index']))
    rows.extend(expanded)
    return dict(training_pool=len(domains), first_time_tasks=len(expanded),
                first_time_by_role=dict(Counter(r['role'] for r in expanded)),
                first_time_with_controls=sum(bool(r['reviewed_skill']) for r in expanded),
                prior_screened=len(inventory['excluded']['previously_screened']))
