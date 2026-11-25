"""Run the frozen Frontier SFT vLLM service and optionally verify a desktop image."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from ais_desktop_service import stop
from ais_vm_check import HTTP, ROOT, save
from ais_worker_cache import stage_runtime


def policy_environment():
    env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', VLLM_NO_USAGE_STATS='1')
    # The AIS 8-GPU NCCL entry prioritizes the mounted driver over the image's compat shim.
    mounted = ['/usr/local/nvidia/lib64', '/usr/local/nvidia/lib']
    inherited = env.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    libraries = [p for p in inherited if p and not p.startswith('/usr/local/cuda/compat')]
    env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(mounted + libraries))
    return env


def check_cuda(python_command, env):
    probe = Path(__file__).with_name('ais_frontier_preflight.py')
    result = subprocess.run([*python_command, str(probe)], env=env, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f'Policy CUDA preflight failed: {result.stderr.strip()[-1800:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def probe(url, model_name, image_path):
    image = image_path.read_bytes()
    if not image.startswith(b'\x89PNG'):
        raise ValueError('The probe needs a PNG desktop observation')
    encoded = base64.b64encode(image).decode('ascii')
    content = [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}},
        {'type': 'text', 'text': 'Describe one visible element of this desktop.'},
    ]
    payload = {
        'model': model_name, 'messages': [{'role': 'user', 'content': content}],
        'temperature': 0, 'max_tokens': 256,
    }
    request = urllib.request.Request(
        url + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with HTTP.open(request, timeout=180) as response:
        result = json.load(response)
    choice = result['choices'][0]
    message = choice['message']
    if not (message.get('content') or message.get('reasoning_content')) or result['usage']['completion_tokens'] < 1:
        raise ValueError('Empty multimodal inference response')
    return {
        'status': 'probe_passed', 'image_sha256': hashlib.sha256(image).hexdigest(),
        'probe_message': message, 'usage': result['usage'], 'finish_reason': choice['finish_reason'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--probe-image', type=Path, help='Verify one real screenshot and stop')
    parser.add_argument('--state-dir', type=Path, help='Unique new directory for this service')
    parser.add_argument('--eval-checkpoint', choices=['osworld-sft', 'scienceboard-sft'],
                        help='Use the archived benchmark decoding settings for this SFT checkpoint')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parent = ROOT / 'runs/ais-frontier-services'
    parent.mkdir(exist_ok=True, parents=True)
    if args.state_dir:
        out = args.state_dir.resolve()
        out.mkdir(exist_ok=False, parents=True)
    else:
        out = Path(tempfile.mkdtemp(prefix='sft-', dir=parent))
    lock = json.loads((ROOT / 'configs/sources.lock.json').read_text())
    base = ROOT.parent.parent
    checkpoint = args.eval_checkpoint or 'osworld-sft'
    if args.eval_checkpoint:
        from checkpoint_specs import checkpoint_spec
        spec = checkpoint_spec(ROOT, checkpoint)
        protocol_path = ROOT / 'runs' / spec['state_file']
        primary = json.loads(protocol_path.read_text())
        source = {'repo_id': spec['model_id'], 'revision': spec['model_revision']}
        model = (base / 'MODEL' / Path(spec['model_dir']).name).resolve(strict=True)
        if primary['served_model_name'] != spec['served_model_name']:
            raise ValueError('Archived benchmark model name mismatch')
    else:
        protocol_path = ROOT / 'runs/frontier-pilot-initialization/protocol.json'
        protocol = json.loads(protocol_path.read_text())
        primary = protocol['services'][0]
        source = lock['sft_model']
        if protocol['model_revision'] != source['revision']:
            raise ValueError('Unexpected archived Frontier revision')
        model = Path(os.getenv('CUA_MODEL_DIR', base / 'MODEL/THUDM-ScaleCUA-qwen3-vl-osworld-sft')).resolve(strict=True)
    if primary['model_revision'] != source['revision']:
        raise ValueError('Unexpected archived SFT revision')
    cmd = list(primary['command'])
    if cmd[1:3] != ['-m', 'vllm.entrypoints.openai.api_server'] or cmd[cmd.index('--tensor-parallel-size') + 1] != '1':
        raise ValueError('Unexpected archived single-GPU inference command')
    for name, value in [('--model', str(model)), ('--host', '127.0.0.1'), ('--port', str(args.port))]:
        cmd[cmd.index(name) + 1] = value
    if args.eval_checkpoint:
        # The 65,536-token service rejected an observed 66,529-token benchmark input.
        cmd[cmd.index('--max-model-len') + 1] = '200000'
    url = f'http://127.0.0.1:{args.port}'
    state = {
        'status': 'starting', 'checkpoint': checkpoint, 'generation_overrides': primary.get('generation_overrides', {}), 'model_revision': source['revision'],
        'model_id': source['repo_id'], 'model_dir': str(model), 'served_model_name': primary['served_model_name'],
        'runtime_compatibility': primary.get('runtime_compatibility'), 'base_url': url + '/v1',
        'hostname': socket.gethostname(), 'source_protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    }
    model_view = None
    policy_venv = None
    state_file = out / 'service.json'
    save(state_file, state)
    print(f'AIS Frontier vLLM: {out}', flush=True)
    started = time.monotonic()
    child = None
    try:
        env = policy_environment()
        site = env.get('CUA_POLICY_SITE')
        manifest = Path(env.get('CUA_POLICY_MANIFEST', base / 'ENV/cache/ais-frontier-vllm/manifest.json'))
        if not site and manifest.is_file():
            runtime, state['runtime_cache'] = stage_runtime(manifest, 'frontier')
            site = str(runtime / 'site')
        python = sys.executable
        if site and sys.version_info[:2] != (3, 10):
            python = str(base / 'ENV/conda_envs/test-gpu/bin/python')
        cmd[0] = env.get('CUA_POLICY_PYTHON', python)
        if site:
            env['PYTHONPATH'] = site
            cmd.insert(1, '-S')
        if args.eval_checkpoint:
            from sft_local_weights import prepare_weights
            model_view = tempfile.TemporaryDirectory(prefix='sft-weights-')
            view = Path(model_view.name) / model.name
            local = env.get('CUA_SFT_LOCAL_SHARDS')
            state['weight_sha256'] = prepare_weights(checkpoint, source['revision'], model, Path(local) if local else None,
                                                     view, ROOT / 'configs/sft-eval-weights.json')
            cmd[cmd.index('--model') + 1] = str(view)
            state['served_path'] = str(view)
        if checkpoint == 'scienceboard-sft':
            adapter = ROOT / 'scripts/scienceboard_tokenizer_compat/sitecustomize.py'
            if hashlib.sha256(adapter.read_bytes()).hexdigest() != primary['runtime_compatibility']['sha256']:
                raise ValueError('Unexpected ScienceBoard tokenizer adapter')
            env['CUA_SCI_TOKENIZER_MODEL_ROOT'] = state.get('served_path', str(model))
            env['PYTHONNOUSERSITE'] = '1'
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(adapter.parent), env.get('PYTHONPATH')]))
            if site:
                # sitecustomize must run in vLLM children. An empty venv excludes the image's
                # ABI-incompatible system flash_attn while allowing the pinned tokenizer adapter to load.
                policy_venv = tempfile.TemporaryDirectory(prefix='sft-policy-python-')
                subprocess.run([cmd[0], '-m', 'venv', '--without-pip', policy_venv.name], check=True, timeout=60)
                cmd[0] = str(Path(policy_venv.name) / 'bin/python')
                cmd.remove('-S')

        state['command'] = cmd
        state['driver_search_path'] = env['LD_LIBRARY_PATH']
        state['inherited_triton_ptxas_path'] = env.get('TRITON_PTXAS_PATH')
        save(state_file, state)
        state['cuda_preflight'] = check_cuda(cmd[:cmd.index('-m')], env)
        env['TRITON_PTXAS_PATH'] = state['cuda_preflight']['ptxas_path']
        save(state_file, state)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        with (out / 'server.log').open('wb') as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + int(env.get('CUA_POLICY_STARTUP_SECONDS', '3600'))
        while time.monotonic() < deadline and child.poll() is None:
            try:
                with HTTP.open(url + '/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f'vLLM did not become ready (exit={child.poll()}); see server.log')
        with HTTP.open(url + '/v1/models', timeout=10) as response:
            served = {item['id'] for item in json.load(response)['data']}
        if state['served_model_name'] not in served:
            raise RuntimeError('SFT identity missing from /v1/models')
        state.update(status='ready', startup_seconds=time.monotonic() - started)
        save(state_file, state)
        if args.probe_image:
            state.update(probe(url, state['served_model_name'], args.probe_image))
        else:
            print(f'Frontier SFT ready: {url}/v1', flush=True)
            raise RuntimeError(f'vLLM exited unexpectedly: {child.wait()}')
    except (KeyboardInterrupt, SystemExit):
        state['status'] = 'stopped'
        raise
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        if child is not None:
            stop(child)
        if policy_venv is not None:
            policy_venv.cleanup()
        if model_view is not None:
            model_view.cleanup()
        state['wall_seconds'] = time.monotonic() - started
        save(state_file, state)


if __name__ == '__main__':
    main()
