#!/usr/bin/env python3
"""Run frozen OSWorld SFT with upstream benchmark code and an explicit QEMU backend.

Task subsets are integration samples, never full benchmark scores. Failed tasks
are retained. This launcher does not train or change checkpoint weights.
"""
import argparse
from checkpoint_specs import CHECKPOINTS, checkpoint_spec
import datetime
import importlib
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.request
from urllib.parse import urlparse
import uuid

from qemu_runtime import ROOT
from sft_episode_audit import EpisodeAudit, evidence_path
from ais_vm_check import save


def summarize(root, args, selected):
    rows = []
    for domain, identifier in selected:
        task_dir = root / 'results' / args.action_space / args.observation_type / args.model / domain / identifier
        trajectory = task_dir / 'traj.jsonl'
        events = []
        if trajectory.exists():
            for line in trajectory.read_text().splitlines(keepends=True):
                if not line.endswith('\n'):  # Another worker may still be writing its current action.
                    continue
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        events.append({'Error': 'Malformed trajectory record'})
        errors = [e['Error'] for e in events if 'Error' in e]
        evidence = evidence_path(root, domain, identifier)
        audit = json.loads(evidence.read_text()) if evidence.exists() else {}
        errors.extend(audit.get('errors', []))
        result = task_dir / 'result.txt'
        score = None
        if result.exists():
            raw = result.read_text().strip().lower()
            score = {'true': 1., 'false': 0.}.get(raw)
            if score is None:
                try:
                    score = float(raw)
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError('Score must be finite and between zero and one')
                except ValueError:
                    score = None
                    if raw:
                        errors.append('Invalid verifier result: ' + raw[:100])
        raw_score = score
        if errors or audit.get('status', 'finished') != 'finished':
            score = None
        status = 'completed' if score is not None else 'missing_result'
        if errors:
            status = 'verifier_review' if all(e.startswith('Verifier: ') for e in errors) else 'infrastructure_error'
        rows.append({'domain': domain, 'id': identifier, 'score': score, 'upstream_score': raw_score,
                     'status': status,
                     'errors': errors, 'model_requested': audit.get('model_requested'), 'action_records': sum('action' in e for e in events), 'directory': str(task_dir)})
    complete = sum(r['status'] == 'completed' for r in rows)
    return {'tasks': rows, 'selected_count': len(rows), 'completed_count': complete,
            'infrastructure_error_count': sum(r['status'] == 'infrastructure_error' for r in rows),
            'verifier_review_count': sum(r['status'] == 'verifier_review' for r in rows),
            'missing_count': sum(r['status'] == 'missing_result' for r in rows),
            'success_count': sum(r['score'] == 1 for r in rows),
            'success_rate': sum(r['score'] == 1 for r in rows) / len(rows) if complete == len(rows) and rows else None,
            'mean_score': sum(r['score'] for r in rows) / len(rows) if complete == len(rows) else None}


def audited_worker(job):
    runner_name, task = job
    domain, identifier, args = task
    audit = EpisodeAudit(evidence_path(Path(args.result_dir).parent, domain, identifier))
    return audit.run(importlib.import_module(runner_name), task)


def worker_init():
    # Pool termination must not inherit the coordinator's KeyboardInterrupt handler.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['osworld', 'scienceboard'], default='osworld')
    parser.add_argument('--checkpoint', choices=CHECKPOINTS, default='osworld-sft')
    parser.add_argument('--accelerator', choices=['tcg', 'kvm'], default='tcg')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--memory-mb', type=int, default=4096)
    parser.add_argument('--tasks', nargs='+', help='Explicit domain/id subset; omission selects the full official list')
    parser.add_argument('--run-dir', type=Path, help='New output directory; existing results are never overwritten')
    parser.add_argument('--boot-timeout', type=int, default=600)
    parser.add_argument('--proxy-policy', choices=['upstream', 'direct'], default='upstream',
                        help='direct explicitly disables the optional OSWorld residential proxy setup')
    parser.add_argument('--recording-mode', choices=['video', 'steps-only'], default='video',
                        help='video preserves upstream recording; steps-only retains PNG/JSONL without continuous guest H.264 encoding')
    parser.add_argument('--service-state', type=Path, help='Pinned inference replica state selected by the coordinator')
    parser.add_argument('--scienceboard-setup', choices=['released', 'restored'], default='released')
    parser.add_argument('--ais', action='store_true', help='Use isolated node-local QEMU overlays and mapped ports')
    args = parser.parse_args()
    if args.scienceboard_setup == 'restored' and args.benchmark != 'scienceboard':
        parser.error('Restored ScienceBoard setup applies only to ScienceBoard')
    cfg = json.loads((ROOT / 'configs/paths.json').read_text())
    lock = json.loads((ROOT / 'configs/sources.lock.json').read_text())
    selected_checkpoint = checkpoint_spec(ROOT, args.checkpoint)
    checkpoint_source = {'repo_id': selected_checkpoint['model_id'], 'revision': selected_checkpoint['model_revision']}
    cfg.update(selected_checkpoint)
    service_path = ROOT / 'runs' / selected_checkpoint['state_file']
    if args.ais:
        if not args.service_state:
            parser.error('--ais requires an explicit local inference --service-state')
        cfg['model_dir'] = str((ROOT.parent.parent / 'MODEL' / Path(cfg['model_dir']).name).resolve())
    if not args.ais and socket.gethostname() not in cfg['workers']:
        parser.error('Run on worker-0/1')
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8; validate higher concurrency separately')
    from prepare_sft_eval import prepare
    source = prepare()
    name = 'test_nogdrive' if args.benchmark == 'osworld' else 'test_scienceboard'
    meta = json.loads((source / 'evaluation_examples' / (name + '.json')).read_text())
    official = [(domain, identifier) for domain, ids in meta.items() for identifier in ids]
    selected = [tuple(value.split('/')) for value in args.tasks] if args.tasks else official
    if len(set(selected)) != len(selected) or any(t not in official for t in selected):
        parser.error('Tasks must be unique domain/id entries in the official benchmark list')
    service_path = args.service_state or service_path
    service_path = service_path.resolve()
    state = json.loads(service_path.read_text())
    if args.benchmark == 'scienceboard':
        server_args = state['command']
        flag = '--limit-mm-per-prompt'
        image_limit = json.loads(server_args[server_args.index(flag) + 1]).get('image', 0) if flag in server_args else 0
        if image_limit < 20:
            raise RuntimeError('ScienceBoard history may contain 20 images. Reconfigure the SFT endpoint image limit before this benchmark; do not shorten the official history to fit.')
    endpoint = state['base_url']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint + '/models', timeout=10) as response:
        models = json.load(response)['data']
    served_path = cfg['model_dir']
    if args.ais and state.get('served_path'):
        weights = json.loads((ROOT / 'configs/sft-eval-weights.json').read_text())[args.checkpoint]
        if state['model_dir'] != served_path or state.get('weight_sha256') != weights['sha256']:
            raise ValueError('AIS node-local weight identity mismatch')
        served_path = state['served_path']
    matching = [m for m in models if m['id'] == cfg['served_model_name'] and m.get('root') == served_path]
    if not matching or state['model_revision'] != cfg['model_revision']:
        raise RuntimeError('Endpoint is not the selected pinned SFT checkpoint')
    image = ROOT / lock['smoke_vm' if args.benchmark == 'osworld' else 'scienceboard_vm']['extracted_file']
    image_sha256 = None
    if args.ais:
        from sft_vm_image import verified_image
        size = lock['smoke_vm' if args.benchmark == 'osworld' else 'scienceboard_vm']['bytes']
        image, image_sha256 = verified_image(args.benchmark, image, size)
    elif not image.is_file():
        raise FileNotFoundError(image)
    run_id = f'sft-{args.benchmark}-{args.accelerator}-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    output = (args.run_dir or ROOT / 'runs/evaluations' / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    os.environ.update(OPENAI_API_KEY='EMPTY', OPENAI_BASE_URL=endpoint, CUA_VM_RUN_ID=run_id,
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    os.environ.pop('LAN_PROXY', None)  # The model endpoint is an internal cluster service.
    # Keep guest/API traffic local even when a platform-wide HTTP proxy is set.
    local_hosts = ','.join(f'127.0.3.{i}' for i in range(20, 240))
    no_proxy = ','.join([os.environ.get('NO_PROXY', ''), local_hosts, '127.0.0.1',
                         urlparse(endpoint).hostname or ''])
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = no_proxy
    if args.ais:
        (output / 'task-compatibility').mkdir(exist_ok=True)
    os.chdir(source)
    (source / 'logs').mkdir(exist_ok=True)
    sys.path.insert(0, str(source))
    module_name = 'run_scalecua_os' if args.benchmark == 'osworld' else 'run_scalecua_sci'
    runner = importlib.import_module(module_name)
    from eval_qemu_provider import install, cleanup_run
    runner.DesktopEnv = install({'accelerator': args.accelerator, 'ais': args.ais, 'cpus': args.cpus,
                                 'memory_mb': args.memory_mb, 'boot_timeout': args.boot_timeout,
                                 'base_image': str(image), 'recording_mode': args.recording_mode,
                                 'proxy_policy': args.proxy_policy, 'benchmark': args.benchmark,
                                 'scienceboard_setup': args.scienceboard_setup, 'audit_directory': str(output / 'task-compatibility')})
    screen = [1920, 1080] if args.benchmark == 'osworld' else [1280, 800]
    image_size = [1280, 720] if args.benchmark == 'osworld' else screen
    cmd = [module_name, '--model', cfg['served_model_name'], '--path_to_vm', str(image),
           '--provider_name', 'docker', '--screen_width', str(screen[0]), '--screen_height', str(screen[1]),
           '--image_size', *map(str, image_size), '--relative_coordinate', '--max_steps', '50',
           '--max_tokens', '4096', '--result_dir', str(output / 'results'),
           '--recording_dir', str(output / 'recordings'), '--num_workers', str(args.workers),
           '--test_config_base_dir', str(source / 'evaluation_examples' / ('examples' if args.benchmark == 'osworld' else 'scienceboard'))]
    if args.benchmark == 'osworld':
        cmd += ['--temperature', '0.8', '--top_p', '1', '--only_n_most_recent_images', '5', '--min_removal_threshold', '5']
    else:
        cmd += ['--temperature', '0.4', '--only_n_most_recent_images', '10']
    sys.argv = cmd
    upstream_args = runner.config()
    generation_path = Path(cfg['model_dir']) / 'generation_config.json'
    manifest = {'run_id': run_id, 'pid': os.getpid(), 'status': 'running', 'hostname': socket.gethostname(),
                'checkpoint': checkpoint_source, 'upstream': lock['upstream'], 'accelerator': args.accelerator,
                'recording_mode': args.recording_mode,
                'boot_mode': 'snapshot' if args.ais and os.getenv('CUA_SFT_TEMPLATE') else 'cold',
                'resident_gui': args.ais and os.getenv('CUA_SFT_RESIDENT_GUI') == '1',
                'proxy_policy': args.proxy_policy,
                'vm_resources': {'cpus': args.cpus, 'memory_mb': args.memory_mb},
                'task_selection': selected, 'official_task_count': len(official), 'selected_count': len(selected),
                'scope': 'Selected-task integration sample' if len(selected) != len(official) else 'Full official task list',
                'scienceboard_setup': args.scienceboard_setup, 'evaluation_source': str(source),
                'prepared_source': json.loads((source / 'sft-source.json').read_text()) if (source / 'sft-source.json').is_file() else None,
                'vm_backend': 'ais' if args.ais else 'local-worker', 'vm_base': str(image), 'vm_sha256': image_sha256,
                'provider_adaptation': 'docker provider API replaced with local QEMU; upstream agent, action loop and verifier retained; explicit resolution and setup checks',
                'upstream_args': vars(upstream_args), 'server_command': state['command'],
                'generation_config': json.loads(generation_path.read_text()) if generation_path.exists() else None,
                'generation_overrides': state.get('generation_overrides', {}),
                'effective_sampling_note': 'OSWorld sends top_p=1. ScienceBoard upstream omits parsed top_p; the explicitly recorded endpoint override takes precedence over generation_config.',
                'output': str(output)}
    manifest['inference_service'] = {'state_file': str(service_path.resolve()),
        'state_sha256': hashlib.sha256(service_path.read_bytes()).hexdigest(),
        'hostname': state['hostname'], 'gpus': state.get('gpus'), 'base_url': state['base_url']}
    manifest['adapter_sha256'] = {}
    for filename in ['evaluate_sft.py', 'eval_qemu_provider.py', 'qemu_runtime.py', 'scienceboard_compat.py', 'scienceboard_setup_repairs.json', 'sft_episode_audit.py', 'prepare_sft_eval.py', 'sft_vm_image.py'] + (['ais_vm_check.py', 'sft_gui_input.py'] if args.ais else []):
        content = (Path(__file__).resolve().parent / filename).read_bytes()
        manifest['adapter_sha256'][filename] = hashlib.sha256(content).hexdigest()
        (output / filename).write_bytes(content)
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'run_id': run_id, 'output': str(output), 'tasks': len(selected), 'accelerator': args.accelerator}), flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    try:
        with multiprocessing.get_context('fork').Pool(args.workers, initializer=worker_init) as pool:
            jobs = [(module_name, (domain, identifier, upstream_args)) for domain, identifier in selected]
            for row in pool.imap_unordered(audited_worker, jobs):
                summary = summarize(output, upstream_args, selected)
                save(output / 'summary.json', summary)
                print(json.dumps({'completed_task': row, 'accounted': summary['completed_count'],
                                  'errors': summary['infrastructure_error_count']}), flush=True)
            pool.close()
            pool.join()
        manifest['status'] = 'finished' if summary['completed_count'] == len(selected) else 'finished_with_errors'
    except BaseException as exc:
        manifest.update(status='interrupted_or_failed', error=str(exc))
        raise
    finally:
        manifest['cleanup'] = cleanup_run(run_id, output)
        manifest['elapsed_seconds'] = time.monotonic() - started
        summary = summarize(output, upstream_args, selected)
        manifest['summary'] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        save(output / 'summary.json', summary)
        if not args.ais:
            (ROOT / 'reports' / f'latest-sft-{args.benchmark}-evaluation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return 0 if summary['completed_count'] == len(selected) else 2


if __name__ == '__main__':
    raise SystemExit(main())
