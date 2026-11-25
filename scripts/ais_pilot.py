"""Prepare the training stage of the fixed eight-GPU GRPO pilot in an AIS job."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import yaml


def checkpoint_capacity(index, contents, destination, retained=1):
    path = destination.resolve()
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f'Checkpoint destination is not a directory: {path}')
    weight_bytes = index.get('metadata', {}).get('total_size')
    free = shutil.disk_usage(path).free
    copies = len({'model', 'hf_model'}.intersection(contents))
    if weight_bytes and free < retained * copies * weight_bytes:
        raise ValueError(f'Checkpoint disk {path}: {free / 2**30:.2f} GiB free, '
                         f'{copies * weight_bytes / 2**30:.2f} GiB per checkpoint for model copies alone, '
                         f'{retained * copies * weight_bytes / 2**30:.2f} GiB for {retained} retained; '
                         'optimizer and RNG require additional space. Set CUA_CHECKPOINT_DIR.')


def final_checkpoint(directory, step, tp_size):
    marker = directory / 'latest_checkpointed_iteration.txt'
    if not marker.is_file() or marker.read_text().strip() != str(step):
        raise ValueError(f'Missing published final checkpoint at step {step}: {marker}')
    checkpoint = directory / f'global_step_{step}'
    files = [checkpoint / 'model' / f'mp_rank_{rank:02d}' / 'model.pt' for rank in range(tp_size)]
    hf = checkpoint / 'huggingface'
    index = hf / 'model.safetensors.index.json'
    if index.is_file():
        weights = set(json.loads(index.read_text())['weight_map'].values())
        if not weights:
            raise ValueError(f'Empty HF weights index: {index}')
        files.extend(hf / name for name in weights)
    else:
        files.append(hf / 'model.safetensors')
    files.append(hf / 'config.json')
    for subdir in ('optim', 'rng_states'):
        parts = list((checkpoint / subdir).glob('*.pt'))
        if not parts:
            raise ValueError(f'Missing {subdir} at {checkpoint}')
        files.extend(parts)
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing checkpoint payload: {path}')
    return checkpoint


def trainer_source_errors(root):
    trainer = importlib.util.find_spec('agentrl.trainer')
    if trainer is None or not trainer.origin:
        return ['Missing agentrl.trainer in the entry Python']
    package = Path(trainer.origin).parent
    pinned = root / 'SCALE-CUA/scalecua_rl/trainer/src/agentrl/trainer'
    errors = []
    for rel in ('agentic/loops.py', 'components/buffer.py', 'components/nccl_tensor_comm.py',
                'components/task_manager.py', 'workers/collective_handle.py',
                'workers/vllm/async_vllm_worker.py', 'workers/vllm/rlhf_utils.py',
                'checkpoint/megatron_swift_checkpoint_manager.py'):
        if not (package / rel).is_file() or (package / rel).read_bytes() != (pinned / rel).read_bytes():
            errors.append(f'Training runtime differs from the repaired source at {package / rel}; '
                          'install the project trainer as editable in the training environment')
    return errors


def prepare(root):
    pilot = json.loads((root / 'configs/original-rl-pilot-pipeline.json').read_text())
    config = yaml.safe_load((root / pilot['training_config']).read_text())
    task = pilot['task_name']
    model = Path(os.environ.get('CUA_MODEL_DIR', root.parent.parent / 'MODEL' / Path(config['model_path']).name)).resolve()
    rates = Path(os.environ.get('CUA_FRONTIER_RATES', root / 'configs/original-pilot/initial_pass_rates.json')).resolve()
    controller = (os.environ.get('CUA_CONTROLLER_URL') or config['task']['base_url']).rstrip('/')
    expected = set(json.loads((root / 'configs/original-pilot/train-indices.json').read_text()))

    print(f'AIS GRPO pilot | host={socket.gethostname()} | python={sys.executable}', flush=True)
    errors = []
    world_size = os.getenv('WORLD_SIZE', '1')
    if os.getenv('CUA_DISTRIBUTED_RUN'):
        if world_size != '8' or os.getenv('RANK') != '0' or not os.getenv('CUA_RAY_ADDRESS'):
            errors.append('The distributed GRPO driver must run on AIS rank zero of eight')
        config['num_gpu_per_mach'] = 1
        config['rollout']['tp_size'] = 1
        config['rollout']['server_args']['mm_processor_cache_gb'] = 2
    elif world_size != '1':
        errors.append(f'Unsupported allocation WORLD_SIZE={world_size}. Run bash AIS_ENTRY.txt rl on one or eight entries.')
    try:
        import torch
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f'Visible CUDA devices ({len(devices)}): {devices}', flush=True)
        if len(devices) != config['num_gpu_per_mach']:
            errors.append(f"Expected {config['num_gpu_per_mach']} GPUs in this container, found {len(devices)}")
    except (ImportError, RuntimeError, OSError) as exc:
        errors.append(f'PyTorch/CUDA: {exc}')
    missing = [pkg for pkg in ('ray', 'agentrl', 'vllm', 'megatron', 'transformer_engine', 'swift')
               if importlib.util.find_spec(pkg) is None]
    if missing:
        errors.append('Missing training packages in the entry Python: ' + ', '.join(missing))
    if 'agentrl' not in missing:
        errors.extend(trainer_source_errors(root))
    if not (root / 'SCALE-CUA/scalecua_rl/examples/training/async_trainer.py').is_file():
        errors.append('Missing upstream trainer; run scripts/bootstrap_sources.py --local-patch --data')
    index = model / 'model.safetensors.index.json'
    if not index.is_file():
        errors.append(f'Missing SFT checkpoint: {index}')
    else:
        inventory = json.loads(index.read_text())
        shards = set(inventory['weight_map'].values())
        try:
            steps = config['max_steps'] - 1
            saves = (steps + config['save_interval'] - 1) // config['save_interval']
            retained = min(config['actor'].get('max_ckpt_to_keep') or saves, saves)
            checkpoint_capacity(inventory, config['actor']['checkpoint_contents'],
                                Path(os.getenv('CUA_CHECKPOINT_DIR', root / 'runs/ais')).expanduser(), retained)
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
        if not shards or any(not (model / shard).is_file() for shard in shards):
            errors.append(f'Incomplete SFT weight shards: {model}')
    if not rates.is_file():
        errors.append(f'Missing Frontier rates for the pilot: {rates}')
    else:
        values = json.loads(rates.read_text()).get(task, {})
        if (not isinstance(values, dict) or set(values) != expected or
                any(type(v) not in (int, float) or v not in (0, 0.25, 0.5, 0.75, 1) for v in values.values())):
            errors.append(f'Invalid rates for the {len(expected)} pilot tasks: {rates}')
    managed = os.getenv('CUA_LOCAL_DESKTOP_DIR') if not os.getenv('CUA_CONTROLLER_URL') else None
    if not errors or not managed:
        if managed:
            from ais_rl_job import local_controller
            controller = local_controller(Path(managed))
        try:
            with urllib.request.urlopen(f'{controller}/get_indices?name={task}', timeout=5) as response:
                indices = json.load(response)
            if not isinstance(indices, list) or {str(i) for i in indices if i != -1} != expected:
                errors.append(f'Controller does not serve the pilot task set: {controller}')
            with urllib.request.urlopen(f'{controller}/list_workers', timeout=5) as response:
                workers = json.load(response)[task]['workers'].values()
                capacity = sum(w['capacity'] for w in workers if w['status'] == 'ALIVE' and not w['stale'])
            if capacity < 1:
                raise ValueError('No ready OSWorld desktop slots')
            config['train']['concurrency'] = min(capacity, config['train']['concurrency'])
            print(f"AIS OSWorld: {capacity} registered desktop slots, rollout concurrency "
                  f"{config['train']['concurrency']}", flush=True)
        except (OSError, ValueError) as exc:
            errors.append(f'OSWorld controller {controller}: {exc}')
    if errors:
        for msg in errors:
            print(f'[AIS] {msg}', file=sys.stderr)
        raise SystemExit(2)

    # Kernel checks run in a child so the trainer can reserve the full devices afterwards.
    subprocess.run([sys.executable, str(root / 'scripts/smoke_training_runtime.py'), '--ais'], check=True)
    parent = root / 'runs/ais'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='original-grpo-pilot-', dir=parent))
    checkpoint_base = os.getenv('CUA_CHECKPOINT_DIR')
    checkpoint_dir = Path(checkpoint_base).expanduser().resolve() / output.name if checkpoint_base else output / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    print(f'AIS checkpoint storage: {checkpoint_dir} ({shutil.disk_usage(checkpoint_dir).free / 2**30:.1f} GiB free)',
          flush=True)
    config.update(model_path=str(model), run_name=output.name, save_path=str(checkpoint_dir),
                  metrics_log_dir=str(output / 'metrics'))
    config['task'].update(base_url=controller, fallback_base_url=controller)
    config['dynamic_sampling']['initial_pass_rate_file'] = str(rates)
    path = output / 'training.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'AIS training outputs: {output}', flush=True)
    return path
