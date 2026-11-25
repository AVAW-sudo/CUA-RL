"""Export official prompts, retaining evidence and keeping pending trials visible."""
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

from build_overnight_screen import AUDIT, MANIFEST, NOTEBOOK, PREVIOUS, ROOT, normalized, read, skill_match

DRAFT = ROOT / 'runs/dataset-screening/overnight-v1'


def publish_latest(directory):
    link = DRAFT.parent / 'latest-overnight'
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    if directory == DRAFT:
        try:
            link.symlink_to(DRAFT.name, target_is_directory=True)
        except FileExistsError:
            pass
        return
    # Each AIS job has one publisher, and a later queued job gets a higher numeric ID.
    if link.is_symlink():
        current = link.resolve().name
        if current.isdigit() and int(current) >= int(directory.name):
            return
    with tempfile.TemporaryDirectory(dir=link.parent) as tmp:
        staged = Path(tmp) / 'latest'
        staged.symlink_to(os.path.relpath(directory, link.parent), target_is_directory=True)
        staged.replace(link)


def write_lines(path, items):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        os.fchmod(f.fileno(), 0o644)
    Path(f.name).replace(path)


def export(directory, job=None, final=None, candidates=None):
    directory.mkdir(parents=True, exist_ok=True)
    catalog = {r['id']: r for line in (AUDIT / 'index.jsonl').read_text().splitlines() if (r := json.loads(line))}
    if candidates is None:
        candidates = read(MANIFEST)
    prior = read(PREVIOUS)['results']
    first = read(NOTEBOOK)['tasks']
    pilot = read(ROOT / 'runs/ais-frontier-campaigns/23500435/verified/initial_pass_rates.json')['osworld-pilot']
    controls = {r['variant'] for n in ['02', '03'] for r in read(AUDIT / f'offline-batch-{n}.json')
                if r['initial'] == 0 and r['positive'] == 1 and 0 <= r['wrong_target'] < 1}
    controls.update(r['variant_id'] for r in first)
    priority = {r['id'] for r in read(AUDIT / 'menu.json')}
    reviewed_skills = {r['variant_id']: (r['skill'], 'main' if r['decision'] == '保留·主练' else 'component')
                       for r in first if r['decision'] in ('保留·主练', '保留·分步')}
    for i, label in read(ROOT / 'configs/dataset-screen/reviewed-weak-skills.json').items():
        if i not in controls:
            raise ValueError(f'Missing fixture controls for reviewed skill {i}')
        reviewed_skills[i] = (label['skill'], label['role'])
    records = {}
    gate_counts = Counter()

    def add(i, source, scores, initial=None):
        r = catalog[i]
        if r['heldout_instruction'] or r['shared_setup_inputs']:
            return
        n = len(scores)
        skill = skill_match(r)
        role = None
        evidence = 'instruction_match' if skill else 'none'
        if i in reviewed_skills:
            skill, role = reviewed_skills[i]
            evidence = 'reviewed'
        records[i] = {'task_index': i, 'prompt': r['instruction'], 'app': r['domain'], 'base_id': r['base'],
                      'seed_app': r['seed_domain'], 'seed_score': r['seed_score'], 'source': source,
                      'seed_action_records': r['seed_actions'], 'matched_skill': skill, 'skill_evidence': evidence,
                      'reviewed_role': role, 'judge_metric': r['metric'], 'judge_getter': r['getter'],
                      'valid_trials': n, 'full_credit_trials': scores.count(1),
                      'frontier_rate': scores.count(1) / 4 if n == 4 else None,
                      'initial_state_reward': initial, 'judge_controls_reviewed': i in controls,
                      'priority_skill_candidate': i in priority}

    for i, rate in pilot.items():
        n = round(rate * 4)
        add(i, 'pilot-23500435', [1] * n + [0] * (4-n))
    for i, row in prior.items():
        if not row['unscored']:
            add(i, 'screen-23527641', row['raw_rewards'], 0)
    for row in first:
        if None not in row['rounds']:
            add(row['variant_id'], 'notebook-first-seven', row['rounds'], 0)
    for index, i in enumerate(candidates):
        initial, scores = None, []
        if job:
            slot = job / f'rank-{index % 8}' / 'episodes' / i
            gate = slot / 'initial-state.json'
            receipt = final[i]['initial_state'] if final is not None else (read(gate) if gate.is_file() else None)
            if receipt is not None:
                if receipt.get('task_index') != i:
                    raise ValueError(f'Misplaced initial state at {gate}')
                gate_counts[receipt['status']] += 1
                if receipt['status'] != 'passed':
                    continue
                initial = receipt['reward']
            if final is not None:
                scores = [v for v in final[i]['raw_rewards'] if v is not None]
            elif receipt is not None:
                for trial in range(1, 5):
                    f = slot / str(trial) / 'result.json'
                    if f.is_file():
                        result = read(f)
                        if result['task_index'] != i or result['run_number'] != trial:
                            raise ValueError(f'Misplaced trial at {f}')
                        scores.append(result['raw_reward'])
            if scores and initial is None:
                raise ValueError(f'Missing initial state for scored task {i}')
        add(i, f'screen-{job.name}' if job else 'overnight-queue', scores, initial)

    # Preserve the disjoint pilot anchors. Fill the remaining slots by score coverage,
    # actionable difficulty, task content and diversity of original task families.
    items = list(records.values())
    anchors = sorted((r for r in items if r['source'].startswith('pilot-')), key=lambda r:r['task_index'])
    chosen = list(anchors)
    used_text = {normalized(r['prompt']) for r in chosen}
    families = Counter(r['base_id'] for r in chosen)
    pool = [r for r in items if not r['source'].startswith('pilot-')]
    apps = Counter(r['app'] for r in anchors)
    def choose(limit, broad):
        while pool and len(chosen) < limit:
            def order(r):
                rate = r['frontier_rate']
                if broad:
                    return (rate is None, bool(r['matched_skill']), apps[r['app']], families[r['base_id']], r['task_index'])
                return (rate is None, r['matched_skill'] is None, r['skill_evidence'] != 'reviewed',
                        rate in (0, 1), not r['judge_controls_reviewed'], not r['priority_skill_candidate'],
                        families[r['base_id']], r['task_index'])
            best = min(pool, key=order)
            pool.remove(best)
            key = normalized(best['prompt'])
            if key not in used_text:
                chosen.append(best)
                used_text.add(key)
                families[best['base_id']] += 1
                apps[best['app']] += 1
    choose(64, broad=True)
    broad_count = len(chosen)
    choose(256, broad=False)
    for idx, r in enumerate(chosen):
        r['selection_role'] = 'breadth' if idx < broad_count else (r['reviewed_role'] or 'pending_skill_review')
    scored = [r for r in chosen if r['frontier_rate'] is not None]
    quotas = {'breadth': 64, 'main': 128, 'component': 64}
    reviewed = []
    reviewed_counts = Counter()
    for r in scored:
        role = r['selection_role']
        if r['judge_controls_reviewed'] and reviewed_counts[role] < quotas.get(role, 0):
            reviewed.append(r)
            reviewed_counts[role] += 1
    write_lines(directory / 'prompts-256.jsonl', chosen)
    write_lines(directory / 'prompts-scored.jsonl', scored)
    write_lines(directory / 'prompts-reviewed.jsonl', reviewed)
    write_lines(directory / 'prompt-pool.jsonl', items)
    from ais_vm_check import save
    save(directory / 'prompt-status.json', {'job': job.name if job else None, 'final': final is not None,
         'requested': 256, 'exported': len(chosen), 'with_four_scores': len(scored),
         'pending_scores': len(chosen) - len(scored), 'scored_shortfall': 256 - len(scored),
         'candidates_before_live_controls': len(items) + sum(n for k, n in gate_counts.items() if k != 'passed'),
         'overnight_queue': len(candidates), 'available_candidates': len(items), 'initial_controls': dict(gate_counts),
         'scored_and_reviewed': len(reviewed), 'reviewed_roles': dict(reviewed_counts),
         'requested_roles': quotas, 'review_shortfall_by_role': {k: n - reviewed_counts[k] for k, n in quotas.items()},
         'pilot_anchors': len(anchors), 'judge_controls_reviewed': sum(r['judge_controls_reviewed'] for r in chosen),
         'apps': dict(Counter(r['app'] for r in chosen))})


if __name__ == '__main__':
    export(DRAFT)
    publish_latest(DRAFT)
