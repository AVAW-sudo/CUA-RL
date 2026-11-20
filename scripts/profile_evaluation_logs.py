"""Read-only phase timing from completed evaluations; no new GPU work.

The gap between environment actions includes inference, image/prompt processing,
serialization and logging. It is not an isolated GPU inference measurement.
"""
import datetime
import json
from pathlib import Path
import re
import statistics

from qemu_runtime import ROOT
from run_benchmark import now, write_json

ANSI = re.compile(r'\x1b\[[0-9;]*m')
STAMP = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) ')


def parse_log(content):
    initialized = setup = first_action = last_reward = active = None
    actions, gaps = [], []
    unmatched = 0
    for raw in content.splitlines():
        line = ANSI.sub('', raw)
        match = STAMP.search(line)
        if not match:
            continue
        timestamp = datetime.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S,%f')
        if 'desktop_env/' in line and 'Initializing...' in line and initialized is None:
            initialized = timestamp
        elif 'desktop_env/' in line and 'Environment setup complete.' in line and setup is None:
            setup = timestamp
        elif 'desktop_env/' in line and re.search(r'Step \d+ in trajectory ', line):
            first_action = first_action or timestamp
            if active is not None:
                unmatched += 1
            if last_reward is not None and active is None:
                gaps.append((timestamp - last_reward).total_seconds())
            active = timestamp
        elif 'lib_run_single/' in line and 'Reward:' in line and active is not None:
            actions.append((timestamp - active).total_seconds())
            last_reward, active = timestamp, None
    if active is not None:
        unmatched += 1
    return {'environment_actions': actions, 'between_action_gaps': gaps,
            'unmatched_actions': unmatched,
            'boot_and_setup_seconds': (setup - initialized).total_seconds() if setup and initialized else None,
            'setup_to_first_action_seconds': (first_action - setup).total_seconds() if first_action and setup else None}


def stats(values):
    values = sorted(x for x in values if x is not None and x >= 0)
    return {'count': len(values), 'sum_seconds': sum(values),
            'median_seconds': statistics.median(values) if values else None,
            'p95_seconds': values[min(len(values) - 1, int(.95 * len(values)))] if values else None}


def main():
    report = {'created_at_utc': now(), 'scope': 'Read-only timings from valid completed SFT attempts',
              'limitations': [
                  'Between-action gaps include model inference and host work; not GPU-only latency.',
                  'Action time includes guest execution, configured sleeps, screenshots and HTTP.',
                  'Initialization includes VM boot, resolution setup and task setup.',
                  'Sums are episode-seconds across concurrent episodes, not campaign wall time.',
                  'Only successful log parses from valid completed attempts are included; failed/running attempts are excluded.'
              ], 'campaigns': []}
    for name in ['osworld-sft__osworld361-tcg-v3', 'osworld-sft__scienceboard169-tcg-v5',
                 'scienceboard-sft__scienceboard169-tcg-v5-pool']:
        campaign = ROOT / 'runs/benchmarks' / name / 'campaign.json'
        state = json.loads(campaign.read_text())
        rows = []
        for key, task in state['tasks'].items():
            if task['status'] != 'completed':
                continue
            attempt = next((a for a in reversed(task['attempts']) if a.get('status') == 'completed'), None)
            if not attempt or not Path(attempt['log']).is_file():
                continue
            row = parse_log(Path(attempt['log']).read_text(errors='replace'))
            row.update(task=key, log=attempt['log'])
            rows.append(row)
        report['campaigns'].append({'campaign': name, 'valid_completed_tasks': sum(t['status'] == 'completed' for t in state['tasks'].values()),
            'parsed_logs': len(rows), 'unmatched_actions': sum(r['unmatched_actions'] for r in rows),
            'environment_action': stats([x for r in rows for x in r['environment_actions']]),
            'between_action_gap': stats([x for r in rows for x in r['between_action_gaps']]),
            'boot_and_setup': stats([r['boot_and_setup_seconds'] for r in rows]),
            'setup_to_first_action': stats([r['setup_to_first_action_seconds'] for r in rows]),
            'tasks': rows})
    write_json(ROOT / 'reports/evaluation-phase-profile.json', report)
    print(json.dumps([{k:v for k,v in c.items() if k != 'tasks'} for c in report['campaigns']], indent=2))


if __name__ == '__main__':
    main()
