"""Training library paths and AIS driver selection before the trainer starts."""
import json
import os
from pathlib import Path
import subprocess
import sys


def library_path(cfg):
    site = Path(cfg["rl_env"]) / "lib/python3.12/site-packages"
    return ":".join([cfg["cache_dir"] + "/cxx-runtime", str(site / "nvidia/cudnn/lib"),
                     *map(str, sorted((site / "nvidia").glob("*/lib"))), str(site / "torch/lib")])


def probe_cuda(include_versions=False):
    import ctypes

    kernel = Path('/proc/driver/nvidia/version')
    result = {'passed': False, 'library_path': os.getenv('LD_LIBRARY_PATH', ''),
              'kernel_driver': kernel.read_text().splitlines()[0] if kernel.is_file() else None}
    if include_versions:
        from importlib.metadata import PackageNotFoundError, version
        result.update(python=sys.executable, python_version=sys.version.split()[0], packages={})
        for name in ('torch', 'vllm', 'ray', 'megatron-core', 'ms-swift', 'agentrl-trainer', 'agentrl-worker',
                     'flash-attn', 'transformer-engine', 'transformer-engine-torch', 'mcore-bridge', 'transformers'):
            try:
                result['packages'][name] = version(name)
            except PackageNotFoundError:
                result['packages'][name] = None
    try:
        import torch
        result.update(torch_version=str(torch.__version__), cuda_build=torch.version.cuda)
        torch.cuda.init()
        count = torch.cuda.device_count()
        result['visible_gpu_count'] = count
        index = int(os.getenv('LOCAL_RANK', '0')) if count > 1 and os.getenv('WORLD_SIZE') == '8' else 0
        value = torch.arange(8, device=f'cuda:{index}').sum().item()
        result.update(device=torch.cuda.get_device_name(index), sum=value, passed=value == 28)
        if value != 28:
            result['error'] = f'Expected CUDA sum 28, got {value}'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    # Query after Torch so its actual library choice is visible even on initialization failure.
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        result['cuInit'] = driver.cuInit(0)
    except OSError as exc:
        result['driver_error'] = str(exc)
    result['driver_libraries'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                         if '/libcuda.so' in line})
    return result


def cuda_failure(evidence):
    attempts = evidence['attempts']
    last = attempts[-1]
    if last.get('cuInit') == 0 and 'driver on your system is too old' in last.get('error', '').lower():
        libraries = ', '.join(last.get('driver_libraries', [])) or 'libcuda.so.1'
        return (f"Driver initialization succeeded with {libraries}, but Torch {last.get('torch_version')} "
                f"built for CUDA {last.get('cuda_build')} rejected this driver as too old. "
                "Use a training runtime compatible with the host driver or a compatible GPU node.")
    return ' | '.join(f"Attempt {i + 1}: {a.get('error') or 'CUDA probe failed'}" for i, a in enumerate(attempts))


def probe_interpreter(python, *, include_versions=False):
    def attempt(libraries):
        probe_env = os.environ.copy()
        if libraries is not None:
            probe_env['LD_LIBRARY_PATH'] = libraries
        try:
            args = [python, __file__, '--probe-cuda']
            if include_versions:
                args.append('--versions')
            p = subprocess.run(args, env=probe_env, text=True,
                               stdout=subprocess.PIPE, timeout=90, check=True)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            return {'passed': False, 'error': str(exc), 'library_path': probe_env.get('LD_LIBRARY_PATH', '')}

    initial = attempt(None)
    attempts = [initial]
    selected = 0
    if not initial['passed'] and (initial.get('cuInit') == 803 or 'error 803:' in initial.get('error', '').lower()):
        from ais_frontier_server import policy_environment
        libraries = policy_environment()['LD_LIBRARY_PATH']
        if libraries != os.getenv('LD_LIBRARY_PATH', ''):
            fallback = attempt(libraries)
            attempts.append(fallback)
            if fallback['passed']:
                selected = 1
    return {'selected': selected, 'attempts': attempts}


def launch_ais(args):
    env = os.environ.copy()
    command = [sys.executable, '-u', *args]
    if '--help' in args or '-h' in args:
        os.execve(sys.executable, command, env)
    probe = probe_interpreter(sys.executable)
    selected = probe['attempts'][probe['selected']]
    if probe['selected']:
        env['LD_LIBRARY_PATH'] = selected['library_path']
    evidence = json.dumps(probe)
    print(f'AIS training CUDA probe: {evidence}', flush=True)
    env['CUA_AIS_CUDA_PROBE'] = evidence
    if not selected['passed'] and '--ais-pilot' in args:
        print(cuda_failure(probe), file=sys.stderr, flush=True)
        raise SystemExit(2)
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    if sys.argv[1:] in (['--probe-cuda'], ['--probe-cuda', '--versions']):
        print(json.dumps(probe_cuda(include_versions='--versions' in sys.argv)))
    else:
        launch_ais(sys.argv[1:])
