#!/usr/bin/env python3
"""Guarded CUDA forward/backward checks for the dedicated training runtime."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from runtime_env import cuda_failure, library_path

ROOT = Path(__file__).resolve().parents[1]


def check_device_kernels():
    import torch
    import flash_attn
    import transformer_engine.pytorch as te
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer = te.Linear(128, 64, params_dtype=torch.bfloat16, device="cuda")
    layer(x).float().square().mean().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError(f'Nonfinite Transformer Engine gradient on {x.device}')
    qkv = [torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    flash_attn.flash_attn_func(*qkv, causal=True).float().square().mean().backward()
    if not all(torch.isfinite(t.grad).all() for t in qkv):
        raise RuntimeError(f'Nonfinite FlashAttention gradient on {qkv[0].device}')
    torch.cuda.synchronize()


def check_kernels(devices=None):
    from importlib.metadata import version
    import torch
    import ray
    import vllm._C
    import megatron.core
    import swift.megatron
    import agentrl.trainer.workers.registry

    if devices is None:
        devices = [torch.cuda.current_device()]
    checked = []
    for device in devices:
        with torch.cuda.device(device):
            check_device_kernels()
            checked.append(device)
    report = {"passed": True, "gpu": torch.cuda.get_device_name(), "tests": ["Transformer Engine BF16 Linear forward/backward", "FlashAttention causal forward/backward", "AgentRL/Megatron/Swift imports"],
              "kernel_cuda_indices": checked,
              "versions": {name: version(name) for name in ["ray", "torch", "vllm", "transformers", "flash-attn", "transformer-engine", "megatron-core", "ms-swift", "mcore-bridge"]},
              "scope": "CUDA kernel/import checks; not yet an 8B RL optimizer update"}
    return report


def verify_core_versions(versions):
    from urllib.parse import unquote, urlsplit
    from packaging.requirements import Requirement
    from packaging.utils import parse_wheel_filename
    from packaging.version import Version

    found = set()
    lock = 'ais-agentrl-cu129-core.txt' if Version(versions['torch']).local == 'cu129' else 'requirements-agentrl.lock.txt'
    for raw in (ROOT / 'configs' / lock).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(('#', '-')):
            continue
        name = line.split('==', 1)[0].split(' @ ', 1)[0]
        if name not in versions:
            continue
        requirement = Requirement(line)
        actual = Version(versions[name])
        if requirement.url:
            wheel = Path(unquote(urlsplit(requirement.url).path)).name
            _, wanted, _, _ = parse_wheel_filename(wheel)
            matches = actual == wanted
        else:
            wanted = str(requirement.specifier)
            matches = actual in requirement.specifier
        if not matches:
            raise RuntimeError(f'Training lock mismatch: {name} installed {actual}, expected {wanted}')
        found.add(name)
    if found != set(versions):
        raise RuntimeError(f'No matching lock entries for {set(versions) - found}')


def check_pilot_placement():
    import ray
    from ray.util import placement_group

    # Match the pilot's local Ray cluster and reserve both training pools at once.
    ray.init(address='local', include_dashboard=False, object_store_memory=8 * 1024**3)
    try:
        resources = ray.cluster_resources()
        if resources.get('GPU') != 8:
            raise RuntimeError(f'Local Ray must discover 8 GPUs, found {resources.get("GPU", 0)}')
        rollout = placement_group([{'CPU': 1, 'GPU': 2}])
        actor_ref = placement_group([{'CPU': 1, 'GPU': 1}] * 6)
        ray.get([rollout.ready(), actor_ref.ready()], timeout=120)
        return {'rollout_gpus': 2, 'actor_reference_gpus': 6, 'available_cpus_before_reservation': resources['CPU']}
    finally:
        ray.shutdown()


def training_stack(devices, verify_source):
    if verify_source:
        from ais_pilot import trainer_source_errors
        errors = trainer_source_errors(ROOT)
        if errors:
            raise RuntimeError('\n'.join(errors))
    kernels = check_kernels(devices)
    verify_core_versions(kernels['versions'])
    return {k: v for k, v in kernels.items() if k != 'passed'}


def distributed_training_check(out, report):
    import faulthandler
    import torch
    from ais_gpu_check import check_all_ranks, save

    count = torch.cuda.device_count()
    device = 0 if count == 1 else int(os.getenv('LOCAL_RANK', '-1'))
    report['torch_device_count'] = count
    if device not in range(count):
        raise RuntimeError(f'Cannot select assigned CUDA device: CUDA count={count}, LOCAL_RANK={os.getenv("LOCAL_RANK")}')
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    uuid = getattr(props, 'uuid', None)
    report['devices'] = [{'name': props.name, 'uuid': str(uuid) if uuid is not None else None}]
    missing = report['missing_modules']
    local = {'python': sys.executable, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
             'missing_modules': missing, 'errors': []}
    report.update(stage='training_runtime', local_runtime=local)
    save(out, report)
    try:
        if missing:
            local['errors'].append('Missing training packages: ' + ', '.join(missing))
            if not {'flash_attn', 'transformer_engine'}.intersection(missing):
                check_device_kernels()
                local['kernel_cuda_indices'] = [device]
        else:
            local.update(training_stack([device], verify_source=True))
    except Exception as exc:
        local['errors'].append(f'{type(exc).__name__}: {exc}')
    # Even a rank with missing training packages participates. All ranks receive
    # the same runtime verdict before the AIS master can finish and release its peers.
    print(f'AIS rank {report["rank"]} joining eight-rank NCCL; runtime errors={local["errors"]}', flush=True)
    faulthandler.dump_traceback_later(240, exit=True)
    try:
        check_all_ranks(torch, device, out, report, runtime=local, require_full_a100=False)
    except Exception:
        if 'distributed_nccl' in report:
            report['distributed_nccl']['status'] = 'failed'
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
    failed = [p['rank'] for p in report['distributed_nccl']['participants'] if p['runtime']['errors']]
    if failed:
        report['stage'] = 'training_runtime'
        raise RuntimeError(f'Training runtime failed on ranks {failed}; see distributed_nccl.participants[].runtime.errors')


def ais_check(eight_gpus=False, all_ranks=False):
    import faulthandler
    import importlib.util
    import tempfile
    from ais_gpu_check import local_nccl_sum, save

    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    parent = ROOT / 'runs/ais-training-runtime'
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='check-', dir=parent)) / 'runtime.json'
    report = dict(passed=False, python=sys.executable, host=socket.gethostname(),
                  ais_job=os.environ.get('JOB_ID'), rank=os.environ.get('RANK'), cuda_visible_devices=visible)
    def stage(name):
        report['stage'] = name
        save(out, report)
        print(f'AIS training runtime [{name}] {out}', flush=True)

    try:
        modules = ('torch', 'ray', 'vllm', 'flash_attn', 'transformer_engine', 'megatron', 'swift', 'agentrl')
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        report['missing_modules'] = missing
        report['world_size'] = os.getenv('WORLD_SIZE', '1')
        if 'CUA_AIS_CUDA_PROBE' in os.environ:
            driver = json.loads(os.environ['CUA_AIS_CUDA_PROBE'])
            report['cuda_driver_probe'] = driver
            selected = driver['attempts'][driver['selected']]
            if not selected['passed']:
                stage('cuda_init')
                raise RuntimeError(cuda_failure(driver))
        if eight_gpus or all_ranks:
            stage('allocation')
            if all_ranks and report['world_size'] == '8':
                report.update(scope='Eight-rank environment check',
                              ray_placement={'status': 'not_run', 'reason': 'Use AIS_ENTRY.txt rl --check-only to verify distributed Ray placement.'})
                distributed_training_check(out, report)
                report.update(stage='finished', passed=True)
                return
            if report['world_size'] != '1':
                raise RuntimeError(f'Expected one AIS instance with 8 visible GPUs, WORLD_SIZE={report["world_size"]}')
            if 'torch' not in missing:
                import torch
                count = torch.cuda.device_count()
                report['cuda_devices'] = [torch.cuda.get_device_name(i) for i in range(count)]
                if count != 8:
                    raise RuntimeError(f'Expected 8 visible GPUs in one instance, found {count}')
                stage('nccl')
                faulthandler.dump_traceback_later(180, exit=True)
                try:
                    values = [torch.tensor([i + 1], dtype=torch.int32, device=f'cuda:{i}') for i in range(count)]
                    report['nccl_all_reduce'] = local_nccl_sum(torch, values)
                finally:
                    faulthandler.cancel_dump_traceback_later()
        stage('training_runtime')
        if missing:
            raise RuntimeError('Missing training packages: ' + ', '.join(missing))
        check_pilot = eight_gpus or all_ranks
        report.update(training_stack(range(8) if check_pilot else None, verify_source=check_pilot))
        if check_pilot:
            stage('ray_placement')
            report['ray_placement'] = check_pilot_placement()
            report['pilot_layout_compatible'] = True
        report.update(stage='finished', passed=True)
    except Exception as exc:
        report['passed'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(out, report)
        print(f'AIS training runtime: {out} | {json.dumps(report)}', flush=True)


def main():
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    if socket.gethostname() != "research-a-worker-1":
        raise SystemExit("Scoped to worker-1 GPU 1")
    if "--child" not in sys.argv:
        for query in ["--query-gpu=memory.used", "--query-compute-apps=pid"]:
            value = subprocess.check_output(["nvidia-smi", "-i", "1", query, "--format=csv,noheader,nounits"], text=True).strip()
            if (query.startswith("--query-gpu") and int(value) > 1024) or (query.startswith("--query-compute") and value):
                raise SystemExit("GPU 1 is occupied")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", LD_LIBRARY_PATH=library_path(cfg), OMP_NUM_THREADS="4")
        subprocess.run([str(Path(cfg["rl_env"]) / "bin/python"), str(Path(__file__).resolve()), "--child"], env=env, check=True, timeout=180)
        return
    report = check_kernels()
    report["gpu"] = "worker-1:1"
    (ROOT / "reports/smoke-training-runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    if "--ais" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--ais', action='store_true')
        layout = parser.add_mutually_exclusive_group()
        layout.add_argument('--eight-gpus', action='store_true', help='Require the single-instance GRPO pilot layout')
        layout.add_argument('--all-ranks', action='store_true', help='Check one eight-GPU instance or eight assigned AIS ranks')
        args = parser.parse_args()
        ais_check(eight_gpus=args.eight_gpus, all_ranks=args.all_ranks)
    else:
        main()
