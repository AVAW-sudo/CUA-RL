"""Compare this AIS instance's GPU allocation before and after its usual Conda launcher."""
import argparse
from datetime import datetime, timedelta, timezone
import faulthandler
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

from audit_node import command, read

ROOT = Path(__file__).resolve().parents[1]
VISIBILITY = ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')
PLACEMENT = ('JOB_ID', 'JOB_NAME', 'VC_JOB_NAME', 'POD_NAME', 'HOSTNAME', 'RANK', 'NODE_RANK',
             'GROUP_RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')


def inventory():
    return {
        'host': socket.gethostname(),
        'python': sys.executable,
        'conda_prefix': os.getenv('CONDA_PREFIX'),
        'placement': {key: os.getenv(key) for key in PLACEMENT},
        'visibility': {key: os.getenv(key) for key in VISIBILITY},
        'nvidia_smi_binary': shutil.which('nvidia-smi'),
        'library_path': os.getenv('LD_LIBRARY_PATH'),
        'device_nodes': sorted(p.name for p in Path('/dev').glob('nvidia[0-9]*') if p.name[6:].isdigit()),
        'nvidia_smi_L': command(['nvidia-smi', '-L']),
        'gpu_identity': command(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total',
                                 '--format=csv,noheader,nounits']),
        'cpu_max': read(Path('/sys/fs/cgroup/cpu.max')),
        'memory_max': read(Path('/sys/fs/cgroup/memory.max')),
    }


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def training_settings():
    from runtime_env import probe_interpreter

    parent = ROOT / 'runs/ais-training-settings'
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix='check-', dir=parent)) / 'settings.json'
    report = inventory()
    report.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), nvcc=command(['nvcc', '--version']),
                  image_build={key: os.environ[key] for key in ('CUDA_VERSION', 'NVIDIA_PYTORCH_VERSION',
                               'NVIDIA_BUILD_ID') if key in os.environ}, interpreters={})
    save(path, report)
    print(f'AIS settings {path} | {json.dumps(report)}', flush=True)
    candidates = {'entry_python': os.getenv('CUA_TRAIN_PYTHON') or 'python',
                  'shared_test_gpu': str(ROOT.parents[1] / 'ENV/conda_envs/test-gpu/bin/python')}
    for label, python in candidates.items():
        probe = probe_interpreter(python, include_versions=True)
        info = {'requested': python, **probe}
        report['interpreters'][label] = info
        save(path, report)
        print(f'AIS settings {label} | {json.dumps(info)}', flush=True)
    return 0


def local_nccl_sum(torch, values):
    if not torch.cuda.nccl.is_available(values):
        raise RuntimeError('Local NCCL all-reduce unavailable')
    torch.cuda.nccl.all_reduce(values)
    expected = len(values) * (len(values) + 1) // 2
    observed = [v.item() for v in values]
    if observed != [expected] * len(values):
        raise RuntimeError(f'NCCL expected {expected} on each device, observed {observed}')
    return observed


def check_all_ranks(torch, device, path, report, *, runtime=None, require_full_a100=True):
    import torch.distributed as dist

    rank = int(os.environ['RANK'])
    gpu = report['devices'][0]
    participant = {'rank': rank, 'host': socket.gethostname(), 'pod': os.getenv('POD_NAME'),
                   'cuda_index': device, 'name': gpu['name'], 'uuid': gpu['uuid']}
    if runtime is not None:
        participant['runtime'] = runtime
    report.update(stage='distributed_nccl', distributed_nccl={'status': 'joining', 'participant': participant})
    save(path, report)
    torch.cuda.set_device(device)
    # AIS already provides one rank per Entry and the common rendezvous address.
    dist.init_process_group('nccl', init_method='env://', timeout=timedelta(seconds=120))
    try:
        value = torch.tensor([rank + 1], dtype=torch.int32, device=f'cuda:{device}')
        dist.all_reduce(value)
        participants = [None] * dist.get_world_size()
        dist.all_gather_object(participants, participant)
        observed = value.item()
        report['distributed_nccl'].update(participants=participants, sum=observed)
        uuids = {p['uuid'] for p in participants}
        if observed != 36 or len(uuids) != 8 or None in uuids or {p['rank'] for p in participants} != set(range(8)):
            raise RuntimeError('Expected ranks 0..7, eight distinct GPU UUIDs and NCCL sum 36; inspect participants')
        if require_full_a100 and not all('A100' in p['name'] and 'MIG' not in p['name'] for p in participants):
            raise RuntimeError('Expected eight full A100 GPUs; inspect participants')
        report['distributed_nccl']['status'] = 'passed'
    finally:
        dist.destroy_process_group()


def after_activation(path):
    report = json.loads(path.read_text())
    report.update(stage='cuda', after_activation=inventory(), devices=[], local_nccl={'status': 'not_run'})
    save(path, report)
    try:
        import torch

        count = torch.cuda.device_count()
        report.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      torch_device_count=count)
        save(path, report)
        if not count:
            raise RuntimeError('PyTorch found no CUDA devices')
        all_ranks = report.get('check_all_ranks', False)
        world_size = int(os.getenv('WORLD_SIZE', '1')) if all_ranks else 1
        indices = range(count)
        if all_ranks:
            if world_size == 8:
                device = 0 if count == 1 else int(os.getenv('LOCAL_RANK', '-1'))
                if device not in indices:
                    raise RuntimeError('Multiple CUDA devices visible; AIS must supply LOCAL_RANK or one GPU per rank')
                indices = [device]
            elif world_size != 1 or count != 8:
                raise RuntimeError(f'Expected eight AIS ranks or one process with eight GPUs; WORLD_SIZE={world_size}, CUDA count={count}')
        values = []
        # A stuck CUDA operation or collective leaves the last stage in the shared report.
        faulthandler.dump_traceback_later(180, exit=True)
        for i in indices:
            props = torch.cuda.get_device_properties(i)
            value = torch.tensor([i], dtype=torch.int32, device=f'cuda:{i}') + 1
            obtained = value.item()
            if obtained != i + 1:
                raise RuntimeError(f'Unexpected arithmetic result on cuda:{i}')
            device_uuid = getattr(props, 'uuid', None)
            report['devices'].append({'cuda_index': i, 'name': props.name, 'memory_bytes': props.total_memory,
                                      'arithmetic_result': obtained,
                                      'uuid': str(device_uuid) if device_uuid is not None else None})
            values.append(value)
            save(path, report)
        if all_ranks and world_size == 8:
            report['local_nccl'] = {'status': 'skipped', 'reason': 'testing one assigned device per AIS rank'}
            faulthandler.dump_traceback_later(240, exit=True)
            check_all_ranks(torch, indices[0], path, report)
        elif count > 1:
            report.update(stage='local_nccl', local_nccl={'status': 'running'})
            save(path, report)
            report['local_nccl'] = {'status': 'passed', 'values': local_nccl_sum(torch, values)}
        else:
            report['local_nccl'] = {'status': 'skipped', 'reason': 'only one CUDA device'}
        report['single_instance_8_a100'] = len(values) == 8 and all(
            'A100' in gpu['name'] and 'MIG' not in gpu['name'] for gpu in report['devices'])
        if all_ranks and world_size == 1 and not report['single_instance_8_a100']:
            raise RuntimeError('Expected eight full A100 GPUs in this instance')
        report.update(status='passed', stage='finished')
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        if report['stage'] == 'local_nccl':
            report['local_nccl']['status'] = 'failed'
        if report['stage'] == 'distributed_nccl':
            report['distributed_nccl']['status'] = 'failed'
    finally:
        faulthandler.cancel_dump_traceback_later()
        save(path, report)
    if 'distributed_nccl' in report:
        collective = report['distributed_nccl']
        print(f"AIS rank {os.getenv('RANK')}/{os.getenv('WORLD_SIZE')}, distributed NCCL: "
              f"{collective['status']}, sum={collective.get('sum')}", flush=True)
    print(f"GPU check: {report['status']}, CUDA count={report.get('torch_device_count')}, "
          f"single-instance eight A100s={report['single_instance_8_a100']} {report.get('error', '')}", flush=True)
    return 0 if report['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-ranks', action='store_true', help='Require and verify a full eight-A100 job')
    parser.add_argument('--settings', action='store_true', help='Log allocation, training package versions and CUDA probes')
    parser.add_argument('--after-activation', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.settings:
        return training_settings()
    if args.after_activation:
        return after_activation(args.after_activation)
    parent = ROOT / 'runs/ais-gpu-checks'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'{stamp}-', dir=parent))
    path = run / 'report.json'
    print(f'AIS GPU report: {path}', flush=True)
    report = {'status': 'running', 'stage': 'conda_activation', 'single_instance_8_a100': False,
              'started_utc': stamp, 'check_all_ranks': args.all_ranks, 'requested_conda_env': os.getenv('CONDA_ENV_NAME'),
              'before_activation': inventory()}
    save(path, report)
    try:
        rc = subprocess.run([str(ROOT.parent / 'run_env.sh'), 'python', '-u', str(Path(__file__).resolve()),
                             '--after-activation', str(path)], cwd=ROOT).returncode
        report = json.loads(path.read_text())
        report['launcher_exit_code'] = rc
        if rc and report['status'] == 'running':
            report.update(status='failed', error=f'Launcher exited with code {rc} during {report["stage"]}')
            if report['stage'] in ('local_nccl', 'distributed_nccl'):
                report[report['stage']]['status'] = 'failed'
    except OSError as exc:
        report.update(status='failed', error=str(exc))
        rc = 2
    save(path, report)
    print(f'{report["status"]}: {path}', flush=True)
    return rc if rc >= 0 else 128 - rc


if __name__ == '__main__':
    raise SystemExit(main())
