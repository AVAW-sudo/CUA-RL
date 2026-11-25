"""Freeze the overnight candidate roster from training-side weakness and asset audits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'runs/dataset-screening/weak-skills-20261125'
NOTEBOOK = ROOT / 'runs/dataset-frontier-validation/first-seven-za1t04xm/progress.json'
PREVIOUS = ROOT / 'runs/ais-dataset-screen/23527641/summary.json'
REPAIR = 'libreoffice_calc_4188d3a4-077d-46b7-9c86-23e1a036f6c1_traj_verify_5'
MANIFEST = ROOT / 'configs/dataset-screen/overnight-v1.json'


def read(path):
    return json.loads(path.read_text())


def normalized(text):
    return ' '.join(text.casefold().split())


def skill_match(row):
    """Identify an operation in the variant itself. These matches still need judge review."""
    s = normalized(row['instruction'])
    output = re.search(r'\b(save|write|record|enter|fill|put|populate|add|insert|create|generate)\b', s)
    if output and re.search(r'\b(grade\b.*\bexam|against the reference answers|answer sheets?|invoices?|across (?:all|both|the) (?:files|documents|sheets))\b', s):
        return 'cross_source_writeback'
    if output and re.search(r'\b(vlookup|xlookup|look(?:ed|ing)?[ -]?up)\b', s) and re.search(r'\b(sheets?|tables?|worksheets?)\b', s):
        return 'table_lookup'
    if output and re.search(r'\b(pdf|json|csv|docx|xlsx|folder|directory|text files?)\b', s) and re.search(r'\b(extract|count|read|find|compare|collect|summari[sz]e)\b', s):
        return 'extract_and_record'
    if row['domain'] in ('libreoffice_calc', 'multi_apps') and re.search(
            r'\b(sum(?:if|ifs)?|count(?:if|ifs|a)?|averag(?:e|es)|calculate|compute|highest|lowest|multiply|percentage)\b', s):
        return 'spreadsheet_computation'
    return None


def main():
    rows = {r['id']: r for line in (AUDIT / 'index.jsonl').read_text().splitlines() if (r := json.loads(line))}
    existing = set(read(PREVIOUS)['results'])
    existing.update(t['variant_id'] for t in read(NOTEBOOK)['tasks'])
    pilot = read(ROOT / 'configs/original-pilot/train-indices.json')
    seen_text = {normalized(rows[i]['instruction']) for i in existing.union(pilot)}
    menu = read(AUDIT / 'menu.json')
    # Earlier targeted reviews found contradictory answers, partial outputs or wrong fields receiving full credit.
    rejected = {menu[i]['id'] for i in [0, 2, 4, 16, 17, 19, 20, 24, 53, 82, 83, 87, 88, 89, 93, 94,
                                         95, 96, 97, 98, 100, 109, 110, 135]}
    rejected.add(read(AUDIT / 'excluded-impress-control.json')['variant'])
    # Exclude failed original/positive/negative fixture controls from the extension audit.
    failed_cases = {'1d17.4','1d17.5','1d17.6','357e.4','357e.5','5171.1','deec.1','a53f.2','af23.2','bb83.2','6f81.5'}
    assets = {r['path']: r for r in read(ROOT / 'reports/sft-osworld-setup-assets.json')['files']
              if r['archived_hash_verified']}
    priority = {r['id'] for r in menu}
    candidates = []
    for i, r in rows.items():
        skill = skill_match(r)
        # A solved seed may have a harder sibling targeting a documented weakness.
        # Its own four trials determine difficulty. Keep these behind failed-seed matches.
        related_skill = r['status'] == 'seed_full' and skill is not None
        if (i in existing or i in rejected or (r['status'] != 'eligible' and not related_skill) or not r['inputs']
                or r['heldout_instruction'] or r['shared_setup_inputs']):
            continue
        case = r['base'][:4] + '.' + i.rsplit('_', 1)[1]
        if case in failed_cases or not all(key in assets for key in r['inputs']):
            continue
        dom = r['domain']
        task = read(ROOT / 'data/ScaleCUA/osworld/rl_tasks' / dom / (i[len(dom)+1:] + '.json'))
        instruction = normalized(task['instruction'])
        if instruction in seen_text or task.get('proxy') or not r['getter'] or r['metric'] == 'infeasible':
            continue
        other_setup = [s for s in task['config'] if s['type'] != 'download']
        if re.search(r'https?://', json.dumps(other_setup), re.I) or re.search(
                r'https?://|\b(google scholar|speedtest|online|internet|send an email|send the email|research where|research how)\b',
                instruction):
            continue
        if dom in ('chrome', 'vs_code', 'os', 'thunderbird', 'vlc') and i not in priority:
            continue
        if re.search(r'\b(password|credential|exploit|bypass|vulnerability|phishing|malware)\b', instruction):
            continue
        score = (6 * (i in priority) + 3 * (r['seed_domain'] == 'multi_apps')
                 + 2 * (dom in ('multi_apps', 'libreoffice_calc', 'libreoffice_writer'))
                 + (r['seed_score'] == 0))
        tier = 2 if not skill else int(related_skill)
        candidates.append((tier, -score, i))
    # Screen skill matches first. Within each tier visit distinct families before siblings.
    counts = Counter()
    selected = [REPAIR]
    counts[rows[REPAIR]['base']] += 1
    for priority_tier in range(3):
        tier_tasks = [i for tier, _, i in sorted(candidates) if tier == priority_tier]
        by_family = Counter(rows[i]['base'] for i in tier_tasks)
        for cap in range(1, 1 + max((counts[f] + n for f, n in by_family.items()), default=0)):
            for i in tier_tasks:
                r = rows[i]
                key = normalized(r['instruction'])
                if counts[r['base']] < cap and key not in seen_text:
                    seen_text.add(key)
                    counts[r['base']] += 1
                    selected.append(i)
    if len(selected) <= 256 or len(selected) != len(set(selected)):
        raise ValueError(f'Expected a deduplicated reserve beyond the 256-prompt target, got {len(selected)}')
    MANIFEST.write_text(json.dumps(selected, indent=2) + '\n')
    print(f'{len(selected)} tasks in {len(counts)} families, {Counter(rows[i]["domain"] for i in selected)}')
    print('sha256', hashlib.sha256(MANIFEST.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
