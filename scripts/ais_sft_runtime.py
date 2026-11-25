"""Run a full SFT campaign from its own frozen evaluator and model-service configuration."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from ais_rank_status import read_rank_status


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    'ais_sft_audit.py', 'ais_sft_eval.py', 'evaluate_sft.py', 'ais_frontier_server.py',
    'ais_frontier_preflight.py', 'ais_desktop_service.py', 'ais_worker_cache.py', 'ais_vm_check.py',
    'ais_vm_guest_check.py', 'eval_qemu_provider.py', 'qemu_runtime.py', 'checkpoint_specs.py',
    'sft_local_weights.py', 'sft_vm_image.py', 'sft_episode_audit.py', 'prepare_sft_eval.py',
    'scienceboard_compat.py', 'scienceboard_setup_repairs.json', 'scienceboard_tokenizer_compat/sitecustomize.py',
)
FILES = tuple('scripts/' + name for name in SCRIPTS) + (
    'scripts/sft_gui_input.py', 'scripts/ais_sft_reuse.py',
    'configs/paths.json', 'configs/sources.lock.json', 'configs/sft-eval-weights.json',
    'patches/sft-evaluation.patch', 'runs/inference-service.json', 'runs/scienceboard-inference-service.json',
)


def checksums(root, paths=FILES):
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def publish_once(stage, dest):
    """A populated directory can be published once, including across separate mounts."""
    try:
        stage.rename(dest)
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not dest.is_dir():
            raise


def snapshot(project, campaign):
    """Publish once across ranks. Only data, environments and output are shared with the live project."""
    dest = campaign / 'runtime'
    relative = Path('CODE/CUA-RL')
    with (campaign / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=campaign) as tmp:
                stage = Path(tmp) / 'runtime'
                root = stage / relative
                for path in FILES:
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(project / path, target)
                    target.chmod(0o444)
                for name in ('MODEL', 'ENV'):
                    (stage / name).symlink_to(project.parent.parent / name)
                for name in ('SCALE-CUA', 'data'):
                    (root / name).symlink_to(project / name)
                output = root / 'runs/ais-sft-evaluations' / campaign.name
                output.mkdir(parents=True)
                (output / 'full').symlink_to(campaign / 'full')
                (stage / 'sha256.json').write_text(json.dumps(checksums(root), indent=2) + '\n')
                publish_once(stage, dest)
        root = dest / relative
        archived = json.loads((dest / 'sha256.json').read_text())
        allowed = set(FILES)
        if (set(archived) not in (allowed, allowed - {'scripts/ais_sft_reuse.py'},
                                  allowed - {'scripts/sft_gui_input.py', 'scripts/ais_sft_reuse.py'})
                or checksums(root, archived) != archived):
            raise ValueError(f'Frozen SFT runtime changed: {dest}')
    return root


def campaign_summary(parent, groups, digest):
    from ais_sft_audit import publish_summary
    from ais_vm_check import save
    records = {}
    for rank in range(8):
        path = parent / f'rank-{rank}/status.json'
        record = read_rank_status(path, missing_ok=True)
        if record is None or record['status'] == 'running':
            return False
        records[rank] = record
    # Older frozen evaluators read the files themselves. Give them the terminal
    # records we just read successfully, including the campaign's reuse ledger.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for rank, record in records.items():
            lane = stage / f'rank-{rank}'
            lane.mkdir()
            save(lane / 'status.json', record)
        (stage / 'reuse').symlink_to((parent / 'reuse').resolve())
        if not publish_summary(stage, groups, digest):
            raise RuntimeError('Frozen evaluator did not summarize eight terminal rank records')
        report = json.loads((stage / 'summary.json').read_text())
    save(parent / 'summary.json', report)
    return report


def run_campaign(parent, rank, source, groups, digest, full):
    # The launcher owns cross-Pod coordination, including for previously frozen evaluators.
    from ais_sft_audit import assignment, run_shard
    out = parent / f'rank-{rank}'
    out.mkdir(parents=True, exist_ok=full)
    shard = assignment(groups, rank)
    with (out / 'rank.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / 'status.json'
        previous = read_rank_status(path, missing_ok=True) or {}
        if previous and (previous['plan_sha256'] != digest or previous['assignment'] != shard):
            raise ValueError(f'Existing rank has a different evaluation protocol: {out}')
        if previous.get('status') not in ('finished', 'finished_with_unscored'):
            run_shard(source, out, rank, shard, digest, full)
    if rank != 0:
        return

    # AIS can terminate the other Pods when its master exits.
    seconds = int(os.environ.get('CUA_AUDIT_JOIN_SECONDS', '0' if full else '21600'))
    until = time.monotonic() + seconds
    while not (report := campaign_summary(parent, groups, digest)):
        if seconds and time.monotonic() >= until:
            raise TimeoutError(f'Some audit ranks have not finished. See {parent}/rank-*/status.json')
        time.sleep(5)
    print(f'AIS SFT audit summary: {parent / "summary.json"}', flush=True)
    if report['failed_ranks']:
        raise RuntimeError(f'Failed evaluation ranks: {list(report["failed_ranks"])}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', required=True)
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument('--resume-job', help='Continue the original campaign using its frozen runtime')
    origin.add_argument('--reuse-scored-from', metavar='JOB', help='Import verified scores into a fresh full campaign')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    allocation = os.environ.get('JOB_ID', '')
    if not args.plan and (not allocation.isdecimal() or os.getenv('WORLD_SIZE') != '8'
                          or os.getenv('RANK') not in {str(i) for i in range(8)}):
        parser.error('Full SFT evaluation requires eight AIS ranks')
    campaign_id = args.resume_job if args.resume_job is not None else allocation
    parent = PROJECT / 'runs/ais-sft-evaluations'
    if args.reuse_scored_from is not None:
        if not args.reuse_scored_from.isdecimal() or (args.reuse_scored_from == campaign_id and not args.plan):
            parser.error('--reuse-scored-from expects a distinct numeric source job ID')
        if args.plan:
            from ais_sft_audit import assignment, load_plan, prepare
            from ais_sft_reuse import handoff
            source = prepare()
            groups, digest = load_plan(source, full=True)
            report = handoff(parent / args.reuse_scored_from, None, PROJECT, source,
                             [assignment(groups, r) for r in range(8)], digest, preview=True)
            print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
            return
    if not campaign_id and args.plan and args.resume_job is None:
        os.execv(sys.executable, [sys.executable, str(PROJECT / 'scripts/ais_sft_audit.py'), '--full', '--plan'])
    if not campaign_id.isdecimal():
        parser.error('Expected a numeric campaign ID')
    campaign = parent / campaign_id
    if args.resume_job and not (campaign / 'runtime/sha256.json').is_file():
        parser.error(f'No existing full campaign at {campaign}')
    (campaign / 'full').mkdir(exist_ok=True, parents=True)
    root = snapshot(PROJECT, campaign)
    os.environ['CUA_PROJECT_ROOT'] = str(root)
    sys.path.insert(0, str(root / 'scripts'))
    from ais_sft_audit import prepare, load_plan, assignment
    source = prepare()
    groups, digest = load_plan(source, full=True)
    print(f'AIS allocation {allocation or "(plan)"}, campaign {campaign_id}, protocol {digest}', file=sys.stderr, flush=True)
    if args.reuse_scored_from is not None:
        if not (root / 'scripts/ais_sft_reuse.py').is_file():
            parser.error('Use a new destination with score-import support')
        from ais_sft_reuse import handoff
        report = handoff(parent / args.reuse_scored_from, campaign, root, source,
                         [assignment(groups, r) for r in range(8)], digest)
        print(f"Inherited scores from {report['source_job']}: {report['groups']}", flush=True)
    os.environ['JOB_ID'] = campaign_id
    if args.plan:
        for status in sorted((campaign / 'full').glob('rank-*/status.json')):
            item = read_rank_status(status)
            if item['plan_sha256'] != digest or item['assignment'] != assignment(groups, item['rank']):
                raise ValueError(f'Frozen evaluator or task list differs from {status}')
        print(json.dumps([assignment(groups, r) for r in range(8)], indent=2))
        return
    run_campaign(campaign / 'full', int(os.environ['RANK']), source, groups, digest, full=True)


if __name__ == '__main__':
    main()
