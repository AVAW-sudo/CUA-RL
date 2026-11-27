"""Freeze an inventory for an AIS worker from the notebook's archived screening results."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

from ais_frontier_shards import read
from ais_screen_queue import CONFIG, ROOT, records
from ais_vm_check import save
from build_overnight_screen import AUDIT, NOTEBOOK, normalized, skill_match


def build(source):
    base = ROOT / 'runs'
    catalog = records(AUDIT / 'index.jsonl')
    train = read(ROOT / 'configs/research-splits/train-indices.json')
    if len(catalog) != len(train) or {r['id'] for r in catalog} != set(train):
        raise ValueError('Weakness audit and official research training split disagree')
    manifest = read(ROOT / 'reports/training-data-manifest.json')
    if manifest['revision'] != read(ROOT / 'configs/sources.lock.json')['training_data']['revision']:
        raise ValueError('Official training release changed')
    hashes = {r['name']: r['sha256'] for r in manifest['files']}
    # Count actual outputs, including the seven notebook trials, independently of planned rosters.
    sampled = set(read(base / 'ais-frontier-campaigns/23500435/verified/initial_pass_rates.json')['osworld-pilot'])
    sampled.update(r['variant_id'] for r in read(NOTEBOOK)['tasks'])
    for kind in ('ais-dataset-screen', 'ais-screen-expand', 'ais-screen-followup'):
        for path in sorted((base / kind).glob('*/summary.json')):
            name = path.parent.name
            if name.isdecimal() and int(name) <= int(source):
                sampled.update(read(path)['results'])
            elif name.startswith('notebook-'):
                sampled.update(read(path)['tasks'])
    upstream = base / 'ais-screen-followup' / source
    if read(upstream / 'summary.json')['status'] not in ('candidates_exhausted', 'quotas_met'):
        raise ValueError('Inventory requires a finished upstream job')
    prior = {r['task_index']: r['decision'] for r in records(upstream / 'cpu-decisions.jsonl')}
    audit = {r['variant_id']: r['primary_reason'] for r in read(ROOT / 'docs/CUA-RL Dataset Audit 32.json')['tasks']
             if r['decision'] == '排除'}
    texts = {normalized(r['instruction']) for r in catalog if r['id'] in sampled}
    if sampled - set(train):
        raise ValueError('Archived screen includes non-training tasks')
    exclude = defaultdict(list)
    new = {}
    for r in catalog:
        i, app = r['id'], r['domain']
        key = normalized(r['instruction'])
        earlier = prior.get(i)
        rejected = earlier == 'prior_review_rejected' and audit.get(i) not in (
            'variant_drops_target_skill', 'shared_key_input_with_holdout')
        if i in sampled:
            reason = 'previously_screened'
        elif r['heldout_instruction']:
            reason = 'heldout_instruction'
        elif audit.get(i) == 'wrong_output_full_credit':
            reason = 'judge_failed_controls'
        elif rejected or earlier in ('getter_omits_requested_outcome', 'judge_failed_controls', 'equivalent_candidate'):
            reason = earlier
        elif r['metric'] == 'infeasible':
            reason = 'infeasible'
        elif key in texts:
            reason = 'duplicate_instruction'
        else:
            skill = skill_match(r)
            weak = r['seed_score'] is not None and r['seed_score'] < 1
            if weak and audit.get(i) != 'variant_drops_target_skill':
                role = 'main'
            elif skill:
                role = 'component'
            else:
                role = 'breadth'
            path = 'osworld/rl_tasks/' + app + '/' + i.removeprefix(app + '_') + '.json'
            new[i] = dict(sha256=hashes[path], seed_score=r['seed_score'], role=role,
                          shared_setup_inputs=r['shared_setup_inputs'])
            texts.add(key)
            continue
        exclude[reason].append(i)
    return dict(source_job=source, revision=manifest['revision'],
                train_sha256=hashlib.sha256((ROOT / 'configs/research-splits/train-indices.json').read_bytes()).hexdigest(),
                excluded=dict(exclude), candidates=new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_job', type=int)
    parser.add_argument('--output', type=Path, default=CONFIG / 'full-pool.json')
    args = parser.parse_args()
    inventory = build(str(args.source_job))
    save(args.output, inventory)
    print({k: len(v) for k, v in inventory['excluded'].items()}, 'new', len(inventory['candidates']))
