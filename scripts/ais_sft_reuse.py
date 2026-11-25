"""Carry verified terminal scores between full AIS campaigns."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ais_vm_check import save
from checkpoint_specs import checkpoint_spec
from sft_vm_image import DIGESTS


def read(path):
    return json.loads(path.read_text())


# Exact reviewed transitions for provenance, boot and GUI dispatch. The desktop
# service change affects Frontier only; SFT imports its unchanged stop function.
ADAPTER_TRANSITIONS = {
    'scripts/ais_desktop_service.py': ('9fad6e954b027476b048ecbc4fac2f4c2d32ac5061f752d191cf2192534c7030',
                                       'fb318d1dbbbe5399a5c5d6521f91efa969005c5f096d1435c8602f77cd23db39'),
    'scripts/evaluate_sft.py': ('3ebaaf7cf3dfb51b2d211f109d20d237338039561dae0139194aaef6caeb7b17',
                                '265716797c492bd09a3fd9e9370dde17f4d5f7d35a22a134afe9837c9eea13c8'),
    'scripts/eval_qemu_provider.py': ('76b0f6045026cfcc79beb5e74df5caf4ec11020df7f02d299b2415c8a4b7f28c',
                                      'b4ff58c6c04ced4bb6ce202af98074f6c6af8f0256d445e6dfa1914756b6ba72'),
}


def collect_scores(old, project, source, shards, digest):
    from ais_sft_runtime import checksums, read_rank_status
    archive = old / 'runtime'
    source_hashes = read(archive / 'sha256.json')
    if checksums(archive / 'CODE/CUA-RL', source_hashes) != source_hashes:
        raise ValueError('Source campaign runtime differs from its frozen manifest')
    current = checksums(project)
    scheduling = {'scripts/' + name for name in ('ais_sft_audit.py', 'ais_sft_eval.py', 'ais_sft_reuse.py',
                                                  'ais_vm_check.py', 'sft_gui_input.py')}
    for path in (source_hashes.keys() | current.keys()) - scheduling:
        pair = source_hashes.get(path), current.get(path)
        if pair[0] != pair[1] and pair != ADAPTER_TRANSITIONS.get(path):
            raise ValueError(f'Incompatible source and destination protocols at {path}')
    upstream = read(source / 'sft-source.json')
    report = {'source_job': old.name, 'destination_plan_sha256': digest, 'groups': {}, 'rows': {}}
    inherited = old / 'full/reuse/manifest.json'
    older_rows = read(inherited)['rows'] if inherited.is_file() else {}
    for rank, shard in enumerate(shards):
        place = old / 'full' / f'rank-{rank}'
        state = read_rank_status(place / 'status.json')
        source_digest = state['plan_sha256']
        if state['rank'] != rank or state['assignment'] != shard or report.setdefault('source_plan_sha256', source_digest) != source_digest:
            raise ValueError(f'Incompatible source assignment or protocol at {place}')
        model = checkpoint_spec(project, shard['checkpoint'])
        generation = read(project / 'runs' / model['state_file'])['generation_overrides']
        benchmark = shard['benchmark']
        for profile in shard['profiles']:
            group = benchmark + '-' + profile
            counts = report['groups'].setdefault(group, Counter())
            key = f'rank-{rank}/{profile}'
            imported = report['rows'].setdefault(key, {})
            for task in shard['tasks']:
                folder = place / profile / 'tasks' / task
                receipt = folder / 'outcome.json'
                evidence = folder / 'evaluation'
                counts['expected'] += 1
                if receipt.is_file():
                    row = read(receipt)
                elif task in older_rows.get(key, {}):
                    row = older_rows[key][task]
                elif (evidence / 'summary.json').is_file() and (evidence / 'manifest.json').is_file():
                    # A terminated rank can have a finished episode awaiting its receipt.
                    try:
                        manifest = read(evidence / 'manifest.json')
                    except (ValueError, OSError):
                        manifest = {}
                    if manifest.get('status') != 'finished':
                        counts['unfinished'] += 1
                        continue
                    row, = read(evidence / 'summary.json')['tasks']
                else:
                    counts['unfinished' if evidence.is_dir() else 'not_started'] += 1
                    continue
                if task != row['domain'] + '/' + row['id']:
                    raise ValueError(f'Task identity mismatch at {folder}')
                if row['status'] != 'completed':
                    counts[row['status']] += 1
                    continue
                value = row['score']
                if type(value) not in (float, int) or not 0 <= value <= 1 or row.get('errors') or row.get('model_requested') is not True:
                    raise ValueError(f'Invalid completed score at {folder}')
                if 'reused_from' in row:
                    if older_rows.get(key, {}).get(task) != row:
                        raise ValueError(f'Inherited receipt differs from source ledger at {folder}')
                    imported[task] = row
                else:
                    manifest = read(evidence / 'manifest.json')
                    adapters = manifest['adapter_sha256']
                    if (manifest.get('status') != 'finished' or manifest.get('task_selection') != [task.split('/')]
                            or manifest['checkpoint'] != {'repo_id': model['model_id'], 'revision': model['model_revision']}
                            or manifest['prepared_source'] != upstream or manifest['vm_sha256'] != DIGESTS[benchmark]
                            or manifest['vm_backend'] != 'ais' or manifest['proxy_policy'] != 'direct'
                            or manifest['generation_overrides'] != generation or manifest['upstream_args']['max_steps'] != 50
                            or manifest['scienceboard_setup'] != profile or manifest['summary']['tasks'] != [row]
                            or read(evidence / 'summary.json')['tasks'] != [row]
                            or not {'evaluate_sft.py', 'eval_qemu_provider.py', 'sft_episode_audit.py'} <= adapters.keys()
                            or any(source_hashes.get('scripts/' + k) != v for k, v in adapters.items())):
                        raise ValueError(f'Score or evaluator evidence mismatch at {evidence}')
                    imported[task] = dict(row, reused_from={'job': old.name, 'plan_sha256': source_digest,
                                                           'evidence': str(evidence)})
                counts['reused'] += 1
                counts['score_sum'] += value
                counts['full_credit'] += value == 1
                counts['zero'] += value == 0
    for stats in report['groups'].values():
        stats['to_evaluate'] = stats['expected'] - stats['reused']
        stats['scored_mean'] = stats['score_sum'] / stats['reused'] if stats['reused'] else None
    report['captured_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return report


def handoff(old, new, project, source, shards, digest, preview=False):
    from ais_sft_runtime import publish_once
    if not old.name.isdecimal() or not (old / 'runtime/sha256.json').is_file() or old == new:
        raise ValueError(f'Expected an existing, distinct full source campaign: {old}')
    if preview:
        return collect_scores(old, project, source, shards, digest)
    ledger = new / 'full/reuse/manifest.json'
    with ExitStack() as locks:
        guard = locks.enter_context((new / 'reuse.lock').open('a'))
        fcntl.flock(guard, fcntl.LOCK_EX)
        if not ledger.is_file():
            if any((new / 'full').glob('rank-*/status.json')) and not ledger.is_file():
                raise ValueError('Use a fresh destination campaign to import scores')
            for rank in range(len(shards)):
                lock = locks.enter_context((old / 'full' / f'rank-{rank}/rank.lock').open('rb'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError(f'Source campaign {old.name} rank {rank} is still evaluating. Switch after it stops.') from None
            report = collect_scores(old, project, source, shards, digest)
            with TemporaryDirectory(dir=ledger.parent.parent) as tmp:
                stage = Path(tmp) / 'reuse'
                stage.mkdir()
                save(stage / ledger.name, report)
                publish_once(stage, ledger.parent)
        report = read(ledger)
        if (report['source_job'], report['destination_plan_sha256']) != (old.name, digest):
            raise ValueError('Destination already uses a different score handoff')
        return report


def seed_rank(out, shard, digest):
    ledger = out.parent / 'reuse/manifest.json'
    if not ledger.is_file():
        return
    report = read(ledger)
    if report['destination_plan_sha256'] != digest:
        raise ValueError('Score handoff belongs to a different destination protocol')
    for profile in shard['profiles']:
        rows = report['rows'][f'{out.name}/{profile}']
        if set(rows) - set(shard['tasks']):
            raise ValueError('Score handoff contains unexpected task IDs')
        for task, row in rows.items():
            path = out / profile / 'tasks' / task / 'outcome.json'
            if path.is_file():
                if read(path) != row:
                    raise ValueError(f'Conflicting existing score at {path}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                save(path, row)
