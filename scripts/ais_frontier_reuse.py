"""Select individually verified Frontier trials from an earlier AIS campaign."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from ais_frontier_shards import ROOT, read, verify_attempt, verify_protocol


def score_windows(log):
    active = None
    windows = {}
    with log.open(errors='replace') as stream:
        for n, line in enumerate(stream, 1):
            start = re.search(r'Evaluating final score for session_id=(\d+)\.\.\.', line)
            if start:
                # Each worker has concurrency one. An aborted evaluation can have
                # a start with no final score before the next session starts.
                active = {'session': start[1], 'start_line': n, 'errors': [], 'file_404s': 0}
            if active is None:
                continue
            if '[ERROR]' in line:
                reason = line.strip().split(']: ', 1)[-1]
                if reason == 'Failed to get file. Status code: 404':
                    active['file_404s'] += 1
                elif not (reason == 'Failed to get file.' and active['file_404s']):
                    active['errors'].append(reason[:160])
            end = re.search(r'Final score: ([\d.eE+-]+), session_id=(\d+)', line)
            if end:
                if active['session'] != end[2] or end[2] in windows:
                    raise ValueError(f'Ambiguous scoring window at {log}:{n}')
                active.update(score=float(end[1]), end_line=n)
                windows[end[2]] = active
                active = None
    return windows


def plan_rank(source, rank, plan, original, manifest, target=None):
    lane = source / f'rank-{rank}'
    episodes = lane / 'episodes'
    if not (episodes / 'protocol.json').is_file():
        if any(episodes.glob('*/*/result.json')) or any(episodes.glob('*/*/attempt-*.json')):
            raise ValueError(f'Missing sampling protocol for existing attempts: {episodes}')
        shard = plan / f'rank-{rank}.json'
        if hashlib.sha256(shard.read_bytes()).hexdigest() != manifest['sha256'][shard.name]:
            raise ValueError(f'Unexpected shard: {shard}')
        tasks = read(shard)
    else:
        tasks = verify_protocol(episodes, rank, plan, manifest, original)
    if not source.is_dir():
        raise ValueError(f'Missing source campaign: {source}')
    # A reused session ID belongs to its original worker, whose log is rechecked.
    inherited = {}
    if (lane / 'reuse.json').is_file():
        inherited = {(r['task'], r['run']): r for r in read(lane / 'reuse.json')['slots'] if r['decision']=='keep'}
    log = None
    if (lane / 'desktop.log').is_file():
        with (lane / 'desktop.log').open() as stream:
            first = stream.readline().strip()
        if first:
            worker = Path(first.removeprefix('AIS desktop service: '))
            if worker.parent.resolve() != (ROOT / 'runs/ais-desktop-services').resolve():
                raise ValueError(f'Unknown desktop service in {lane}')
            log = worker / 'worker.log'
    logs = {}
    slots = []
    for task in tasks:
        for number in range(1, 5):
            run = episodes / task / str(number)
            slot = {'task': task, 'run': number}
            slots.append(slot)
            if not (run / 'result.json').exists():
                attempts = [read(p) for p in run.glob('attempt-*.json')]
                slot.update(decision='retry', reason='no_terminal', prior_model_calls=sum(a.get('model_calls', 0) for a in attempts))
                continue
            row = read(run / 'result.json')
            attempt = verify_attempt(run, row, original)
            if row['task_index'] != task or row['run_number'] != number:
                raise ValueError(f'Misplaced terminal record {run}')
            origin = inherited.get((task, number))
            path = ROOT / origin['worker_log'] if origin else log
            if path is None or not path.is_file():
                slot.update(decision='retry', reason='unverified_score')
                continue
            if path.parent.parent.resolve() != (ROOT/'runs/ais-desktop-services').resolve() or path.name != 'worker.log':
                raise ValueError(f'Unknown worker log {path}')
            if path not in logs:
                logs[path] = score_windows(path)
            window = logs[path].get(str(attempt['session_id']))
            slot.update(worker_log=str(path.relative_to(ROOT)), session=attempt['session_id'])
            if window is None:
                # The collector enforces the context limit locally, before /interact.
                status = row['terminal_status']
                trace_file = run / f"trace-{row['attempt']:02d}.jsonl"
                last = json.loads(trace_file.read_text().splitlines()[-1]) if trace_file.is_file() else {}
                context_limit = (status == 'agent context limit' and
                                 last.get('usage', {}).get('total_tokens', 0) > original['max_total_len'])
                invalid_action = (status in ('agent invalid action', 'agent validation failed') and
                                  attempt.get('terminal') == {'finish': True, 'status': status, 'reward': 0})
                if row['raw_reward'] != 0 or not (context_limit or invalid_action):
                    slot.update(decision='retry', reason='unverified_score')
                    continue
                slot['termination'] = status
            elif window['score'] != row['raw_reward']:
                slot.update(decision='retry', reason='unverified_score')
                continue
            else:
                slot.update(start_line=window['start_line'], end_line=window['end_line'], file_404s=window['file_404s'])
                if window['errors']:
                    slot.update(decision='retry', reason='scoring_error', errors=window['errors'])
                    continue
            names = [f"attempt-{row['attempt']:02d}.json", 'result.json']
            if (run / f"trace-{row['attempt']:02d}.jsonl").exists():
                names.append(f"trace-{row['attempt']:02d}.jsonl")
            slot.update(decision='keep', source=str(run.relative_to(ROOT)), files={
                name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names})
            if origin and origin['files'] != slot['files']:
                raise ValueError(f'Inherited terminal record changed: {run}')
            if target:
                dest = target / 'episodes' / task / str(number)
                dest.mkdir(parents=True, exist_ok=False)
                for name in names:
                    shutil.copyfile(run/name, dest/name)
                    if hashlib.sha256((dest/name).read_bytes()).hexdigest() != slot['files'][name]:
                        raise ValueError(f'Source changed during copy: {run/name}')
    summary = dict(Counter(s.get('reason', 'keep') for s in slots))
    report = {'source_job':source.name, 'rank':rank, 'counts':summary, 'slots':slots}
    if target:
        (target/'reuse.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
