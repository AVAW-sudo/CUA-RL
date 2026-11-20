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
        result = task_dir / 'result.txt'
        score = None
        if result.exists() and not errors:
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
        rows.append({'domain': domain, 'id': identifier, 'score': score,
                     'status': 'infrastructure_error' if errors else ('completed' if score is not None else 'missing_result'),
                     'errors': errors, 'action_records': sum('action' in e for e in events), 'directory': str(task_dir)})
    complete = sum(r['status'] == 'completed' for r in rows)
    return {'tasks': rows, 'selected_count': len(rows), 'completed_count': complete,
            'infrastructure_error_count': sum(r['status'] == 'infrastructure_error' for r in rows),
            'missing_count': sum(r['status'] == 'missing_result' for r in rows),
            'success_count': sum(r['score'] == 1 for r in rows),
            'success_rate': sum(r['score'] == 1 for r in rows) / len(rows) if complete == len(rows) and rows else None,
            'mean_score': sum(r['score'] for r in rows) / len(rows) if complete == len(rows) else None}


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
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/paths.json').read_text())
    lock = json.loads((ROOT / 'configs/sources.lock.json').read_text())
    selected_checkpoint = checkpoint_spec(ROOT, args.checkpoint)
    checkpoint_source = {'repo_id': selected_checkpoint['model_id'], 'revision': selected_checkpoint['model_revision']}
    cfg.update(selected_checkpoint)
    service_path = ROOT / 'runs/inference-service.json'
    service_path = ROOT / 'runs' / selected_checkpoint['state_file']
    if socket.gethostname() not in cfg['workers']:
        parser.error('Run on worker-0/1')
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8; validate higher concurrency separately')
    source = ROOT / 'SCALE-CUA/osworld_eval'
    name = 'test_nogdrive' if args.benchmark == 'osworld' else 'test_scienceboard'
    meta = json.loads((source / 'evaluation_examples' / (name + '.json')).read_text())
    official = [(domain, identifier) for domain, ids in meta.items() for identifier in ids]
    selected = [tuple(value.split('/')) for value in args.tasks] if args.tasks else official
    if len(set(selected)) != len(selected) or any(t not in official for t in selected):
        parser.error('Tasks must be unique domain/id entries in the official benchmark list')
    service_path = args.service_state or service_path
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
    matching = [m for m in models if m['id'] == cfg['served_model_name'] and m.get('root') == cfg['model_dir']]
    if not matching or state['model_revision'] != cfg['model_revision']:
        raise RuntimeError('Endpoint is not the selected pinned SFT checkpoint')
    image = ROOT / lock['smoke_vm' if args.benchmark == 'osworld' else 'scienceboard_vm']['extracted_file']
    if not image.is_file():
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
    os.chdir(source)
    (source / 'logs').mkdir(exist_ok=True)
    sys.path.insert(0, str(source))
    module_name = 'run_scalecua_os' if args.benchmark == 'osworld' else 'run_scalecua_sci'
    runner = importlib.import_module(module_name)
    from eval_qemu_provider import install, cleanup_run
    runner.DesktopEnv = install({'accelerator': args.accelerator, 'cpus': args.cpus,
                                 'memory_mb': args.memory_mb, 'boot_timeout': args.boot_timeout,
                                 'base_image': str(image), 'recording_mode': args.recording_mode,
                                 'proxy_policy': args.proxy_policy, 'benchmark': args.benchmark,
                                 'audit_directory': str(output / 'task-compatibility')})
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
                'proxy_policy': args.proxy_policy,
                'vm_resources': {'cpus': args.cpus, 'memory_mb': args.memory_mb},
                'task_selection': selected, 'official_task_count': len(official), 'selected_count': len(selected),
                'scope': 'Selected-task integration sample' if len(selected) != len(official) else 'Full official task list',
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
    for filename in ['evaluate_sft.py', 'eval_qemu_provider.py', 'qemu_runtime.py', 'scienceboard_compat.py']:
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
            jobs = [(domain, identifier, upstream_args) for domain, identifier in selected]
            for row in pool.imap_unordered(runner._worker_run, jobs):
                summary = summarize(output, upstream_args, selected)
                (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                print(json.dumps({'completed_task': row, 'accounted': summary['completed_count'],
                                  'errors': summary['infrastructure_error_count']}), flush=True)
            pool.close()
            pool.join()
        manifest['status'] = 'finished'
    except BaseException as exc:
        manifest.update(status='interrupted_or_failed', error=str(exc))
        raise
    finally:
        manifest['cleanup'] = cleanup_run(run_id)
        manifest['elapsed_seconds'] = time.monotonic() - started
        summary = summarize(output, upstream_args, selected)
        manifest['summary'] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (ROOT / 'reports' / f'latest-sft-{args.benchmark}-evaluation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return 0 if summary['completed_count'] == len(selected) else 2


if __name__ == '__main__':
    raise SystemExit(main())
