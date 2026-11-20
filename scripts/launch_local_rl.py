"""Launch the original trainer in a project-owned, single-node six-GPU Ray cluster."""
import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request
from runtime_env import library_path

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'configs/paths.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--child', action='store_true')
parser.add_argument('--config', default='configs/qwen3vl-osworld-local-1step.yaml')
parser.add_argument('--report', default='reports/smoke-online-rl.json')
args = parser.parse_args()
if socket.gethostname() != 'research-a-worker-1':
    parser.error('This bring-up launcher is scoped to worker-1')
if not args.child:
    for gpu in [0, 1, 2, 5, 6, 7]:
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
        processes = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if used > 1024 or processes:
            raise SystemExit(f'GPU {gpu} occupied; launcher will not stop any process')
    prometheus = ROOT / 'runs/prometheus' / str(os.getpid())
    prometheus.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,5,6,7', LD_LIBRARY_PATH=library_path(cfg),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               NO_PROXY='127.0.0.0/8,localhost,.training.svc.cluster.local', NCCL_DEBUG='WARN',
               RAY_ADDRESS='local', RAY_USAGE_STATS_ENABLED='0', VLLM_NO_USAGE_STATS='1',
               PROMETHEUS_MULTIPROC_DIR=str(prometheus), VLLM_CACHE_ROOT=cfg['cache_dir'] + '/vllm019',
               TRITON_CACHE_DIR=cfg['cache_dir'] + '/triton-agentrl')
    os.execve(str(Path(cfg['rl_env']) / 'bin/python'), [str(Path(cfg['rl_env']) / 'bin/python'), str(Path(__file__).resolve()), '--child', '--config', args.config, '--report', args.report], env)

import ray
import yaml
config = yaml.safe_load((ROOT / args.config).read_text())
report_path = ROOT / args.report
assert all(config['task']['running_mode'].get(name) == 'custom' for name in config['task']['train_tasks']), 'Preserve the upstream custom visual segmentation loop'
config['run_name'] += '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
# A short local temp path avoids UNIX socket length limits and any shared Ray session.
ray_tmp = '/tmp/cua-rl-ray-' + str(os.getuid()) + '-' + str(os.getpid())
ray.init(address='local', num_gpus=6, num_cpus=24, resources={'big_mem': 24}, include_dashboard=False,
         object_store_memory=8 * 1024**3, _temp_dir=ray_tmp,
         runtime_env={'env_vars': {k: v for k, v in os.environ.items() if k in {
             'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'WANDB_MODE', 'TOKENIZERS_PARALLELISM', 'PYTHONUNBUFFERED', 'HF_HUB_OFFLINE', 'NO_PROXY', 'NCCL_DEBUG',
             'VLLM_NO_USAGE_STATS', 'PROMETHEUS_MULTIPROC_DIR', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR'}}})
try:
    source = ROOT / 'SCALE-CUA/scalecua_rl/examples/training/async_trainer.py'
    spec = importlib.util.spec_from_file_location('scalecua_async_trainer', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main(config)
    logs = list(Path(config['metrics_log_dir']).glob('*/' + config['run_name'] + '/train_metrics.jsonl'))
    rows = [json.loads(line) for log in logs for line in log.read_text().splitlines() if line.strip()]
    updates = [row for row in rows if 'actor/grad_norm' in row]
    valid_updates = bool(updates) and all(math.isfinite(float(row['actor/grad_norm'])) for row in updates)
    report = {'passed': valid_updates, 'updates': updates, 'run_name': config['run_name'],
              'scope': 'Local infrastructure smoke with released eval agent and TCG VM; not exact paper reproduction',
              'physical_gpus': [0, 1, 2, 5, 6, 7], 'checkpoint_saved': False,
              'metrics_files': [str(log) for log in logs], 'config': args.config,
              'nonzero_gradient_updates': sum(float(row['actor/grad_norm']) > 0 for row in updates),
              'effective_group_fraction': [row.get('rl/overall/effective_uids') for row in updates]}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    assert valid_updates, 'Trainer returned without an actor update with finite gradient norm'
except Exception as exc:
    report_path.write_text(json.dumps(
        {'passed': False, 'run_name': config['run_name'], 'error': str(exc), 'scope': 'Local infrastructure smoke'}, indent=2) + '\n')
    raise
finally:
    ray.shutdown()
    # The original async trainer prefetches episodes beyond the final batch.
    # Cancel only this project's named task sessions after the owned Ray cluster exits.
    try:
        endpoint = config['task']['base_url']
        sessions = json.load(urllib.request.urlopen(endpoint + '/list_sessions', timeout=10))
        for sid, session in sessions.items():
            if session.get('name') == 'osworld-local-smoke':
                req = urllib.request.Request(endpoint + '/cancel', data=b'', headers={'session_id': str(sid)})
                urllib.request.urlopen(req, timeout=20).close()
    except Exception as exc:
        print(f'Owned task cleanup needs inspection: {exc}', flush=True)
