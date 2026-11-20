"""Launch the original trainer in a project-owned, single-node Ray cluster."""
import argparse
import datetime
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request
from runtime_env import library_path
from trainer_save_boundary import add_final_save

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'configs/paths.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--child', action='store_true')
parser.add_argument('--config', default='configs/qwen3vl-osworld-local-1step.yaml')
parser.add_argument('--report', default='reports/smoke-online-rl.json')
parser.add_argument('--gpus', default='0,1,2,5,6,7')
parser.add_argument('--run-name', help='Stable run name enables checkpoint resume')
parser.add_argument('--save-final', action='store_true', help='Export the final update even when it is not a save_interval multiple')
args = parser.parse_args()
if socket.gethostname() != 'research-a-worker-1':
    parser.error('This bring-up launcher is scoped to worker-1')
if not args.child:
    gpu_ids = [int(gpu) for gpu in args.gpus.split(',')]
    if len(gpu_ids) != len(set(gpu_ids)) or any(gpu not in range(8) for gpu in gpu_ids):
        parser.error('GPU list must contain distinct IDs in 0..7')
    for gpu in gpu_ids:
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
        processes = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if used > 1024 or processes:
            raise SystemExit(f'GPU {gpu} occupied; launcher will not stop any process')
    prometheus = ROOT / 'runs/prometheus' / str(os.getpid())
    prometheus.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, LD_LIBRARY_PATH=library_path(cfg),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               NO_PROXY='127.0.0.0/8,localhost,.training.svc.cluster.local', NCCL_DEBUG='WARN',
               RAY_ADDRESS='local', RAY_USAGE_STATS_ENABLED='0', VLLM_NO_USAGE_STATS='1',
               PROMETHEUS_MULTIPROC_DIR=str(prometheus), VLLM_CACHE_ROOT=cfg['cache_dir'] + '/vllm019',
               TRITON_CACHE_DIR=cfg['cache_dir'] + '/triton-agentrl')
    command = [str(Path(cfg['rl_env']) / 'bin/python'), str(Path(__file__).resolve()), '--child', '--config', args.config, '--report', args.report, '--gpus', args.gpus]
    if args.run_name:
        command += ['--run-name', args.run_name]
    if args.save_final:
        command += ['--save-final']
    os.execve(command[0], command, env)

import ray
import yaml
config = yaml.safe_load((ROOT / args.config).read_text())
report_path = ROOT / args.report
assert all(config['task']['running_mode'].get(name) == 'custom' for name in config['task']['train_tasks']), 'Preserve the upstream custom visual segmentation loop'
config['run_name'] = args.run_name or config['run_name'] + '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
gpu_ids = [int(gpu) for gpu in args.gpus.split(',')]
assert len(gpu_ids) == config['num_gpu_per_mach'], 'GPU allocation must match the configuration'
# A short local temp path avoids UNIX socket length limits and any shared Ray session.
ray_tmp = '/tmp/cua-rl-ray-' + str(os.getuid()) + '-' + str(os.getpid())
ray.init(address='local', num_gpus=len(gpu_ids), num_cpus=32, resources={'big_mem': 32}, include_dashboard=False,
         object_store_memory=8 * 1024**3, _temp_dir=ray_tmp,
         runtime_env={'env_vars': {k: v for k, v in os.environ.items() if k in {
             'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'WANDB_MODE', 'TOKENIZERS_PARALLELISM', 'PYTHONUNBUFFERED', 'HF_HUB_OFFLINE', 'NO_PROXY', 'NCCL_DEBUG',
             'VLLM_NO_USAGE_STATS', 'PROMETHEUS_MULTIPROC_DIR', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR'}}})
try:
    source = ROOT / 'SCALE-CUA/scalecua_rl/examples/training/async_trainer.py'
    trainer_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.save_final:
        content = add_final_save(source.read_text())
        source = ROOT / 'runs/original-rl-pipeline/trainer-with-final-save.py'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content)
        (source.with_suffix('.audit.json')).write_text(json.dumps({
            'original_sha256': trainer_source_sha256, 'adapted_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope': 'Save final boundary and retain checkpoint saves on skipped empty batches; training and sampling math unchanged'}, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('scalecua_async_trainer', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main(config)
    logs = list(Path(config['metrics_log_dir']).glob('*/' + config['run_name'] + '/train_metrics.jsonl'))
    rows = [json.loads(line) for log in logs for line in log.read_text().splitlines() if line.strip()]
    updates = [row for row in rows if 'actor/grad_norm' in row]
    valid_updates = bool(updates) and all(math.isfinite(float(row['actor/grad_norm'])) for row in updates)
    nonzero_updates = sum(float(row['actor/grad_norm']) > 0 for row in updates)
    report = {'passed': valid_updates and nonzero_updates > 0, 'execution_completed': True, 'updates': updates, 'run_name': config['run_name'],
              'scope': 'Original algorithm with explicitly recorded public-agent and TCG resource adaptations',
              'physical_gpus': gpu_ids,
              'checkpoint_root': str(Path(config['save_path']) / config['project_name'] / config['run_name']),
              'checkpoint_saved': bool(list((Path(config['save_path']) / config['project_name'] / config['run_name']).glob('global_step_*'))),
              'metrics_files': [str(log) for log in logs], 'config': args.config,
              'nonzero_gradient_updates': nonzero_updates,
              'effective_group_fraction': [row.get('rl/overall/effective_uids') for row in updates]}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    assert valid_updates, 'Trainer returned without an actor update with finite gradient norm'
    if not nonzero_updates:
        raise RuntimeError('Trainer completed but every gradient was zero; this is not an effective learning baseline')
except Exception as exc:
    failure = {'passed': False, 'run_name': config['run_name'], 'error': str(exc), 'scope': 'Original GRPO execution'}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('run_name') == config['run_name']:
            failure = {**previous, **failure}
    report_path.write_text(json.dumps(failure, indent=2) + '\n')
    raise
finally:
    ray.shutdown()
    # The original async trainer prefetches episodes beyond the final batch.
    # Cancel only this project's named task sessions after the owned Ray cluster exits.
    try:
        endpoint = config['task']['base_url']
        sessions = json.load(urllib.request.urlopen(endpoint + '/list_sessions', timeout=10))
        for sid, session in sessions.items():
            if session.get('name') in config['task']['train_tasks']:
                req = urllib.request.Request(endpoint + '/cancel', data=b'', headers={'session_id': str(sid)})
                urllib.request.urlopen(req, timeout=20).close()
    except Exception as exc:
        print(f'Owned task cleanup needs inspection: {exc}', flush=True)
