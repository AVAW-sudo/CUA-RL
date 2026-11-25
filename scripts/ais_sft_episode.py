"""Score one fixed OSWorld pilot task through the real worker using local SFT weights."""
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

from ais_vm_check import ROOT, HTTP, save

TASK = 'os_13584542-872b-42d8-b299-866967b5c3ef_traj_verify_2'


def rpc(api, path, body=None, sid=None, timeout=700):
    headers = {'Content-Type': 'application/json'}
    if sid is not None:
        headers['session_id'] = str(sid)
    req = urllib.request.Request(api + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    with HTTP.open(req, timeout=timeout) as response:
        return json.load(response), response.headers.get('session_id')


def load_model(path):
    started = time.monotonic()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map={'': 0},
                                                            local_files_only=True).eval()
    torch.cuda.synchronize()
    return model, processor, time.monotonic() - started


def generate(model, processor, messages, tools):
    import torch
    from PIL import Image
    images = []
    for msg in messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    url = item['image_url']['url']
                    if not url.startswith('data:image/'):
                        raise ValueError('Expected an inline desktop screenshot')
                    images.append(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB'))
    text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
    ids = output[0, inputs.input_ids.shape[1]:]
    raw = processor.decode(ids, skip_special_tokens=True)
    if len(ids) == 1024 or not raw.strip():
        raise ValueError('Empty or truncated model output')
    call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    calls = []
    for field in re.findall(call_pattern, raw, re.DOTALL):
        call = json.loads(field)
        if call['name'] != 'computer' or not isinstance(call['arguments'], dict):
            raise ValueError('Expected an OSWorld computer action')
        calls.append({'id': uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': call['name'], 'arguments': json.dumps(call['arguments'])}})
    content = re.sub(call_pattern, '', raw, flags=re.DOTALL).strip()
    if '<tool_call>' in content or '</tool_call>' in content:
        raise ValueError('Incomplete tool call')
    message = {'role': 'assistant', 'content': content}
    if calls:
        message['tool_calls'] = calls
    return message, len(ids)


def save_observation(messages, path):
    for message in reversed(messages):
        if isinstance(message.get('content'), list):
            for item in reversed(message['content']):
                if item.get('type') == 'image_url':
                    url = item['image_url']['url']
                    header, _, encoded = url.partition(',')
                    suffix = {'data:image/png;base64': '.png', 'data:image/jpeg;base64': '.jpg'}.get(header)
                    if suffix is None:
                        raise ValueError('Expected an inline PNG or JPEG observation')
                    path = path.with_suffix(suffix)
                    path.write_bytes(base64.b64decode(encoded, validate=True))
                    return path.name
    raise ValueError('The worker supplied no screenshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-worker', action='store_true', help='Start and close a private controller and desktop worker')
    parser.add_argument('--full-worker-trace', action='store_true', help='Also collect the upstream accessibility trace')
    args = parser.parse_args()
    if int(os.getenv('RANK', '0')) != 0:
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    model_path = Path(os.getenv('CUA_MODEL_DIR', ROOT.parent.parent / 'MODEL/THUDM-ScaleCUA-qwen3-vl-osworld-sft'))
    api = os.getenv('CUA_CONTROLLER_URL', 'http://127.0.0.1:18520/api').rstrip('/')
    parent = ROOT / 'runs/ais-sft-episodes'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'), dir=parent))
    report = {'status': 'running', 'stage': 'startup', 'task': TASK, 'controller': api, 'model': str(model_path),
              'expected_model_revision': json.loads((ROOT / 'configs/sources.lock.json').read_text())['sft_model']['revision'],
              'inference': 'Transformers BF16 greedy, max_new_tokens=1024', 'scope': 'AIS integration only',
              'full_worker_trace': args.full_worker_trace, 'reward': None, 'turns': []}
    save(run / 'report.json', report)
    print(f'AIS SFT episode: {run}', flush=True)
    sid = None
    service = None
    start = time.monotonic()
    try:
        spec = importlib.util.spec_from_file_location('server_agent', ROOT / 'SCALE-CUA/scalecua_rl/scripts/server_agent.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        if args.with_worker:
            with socket.socket() as first, socket.socket() as second:
                first.bind(('127.0.0.1', 0))
                second.bind(('127.0.0.1', 0))
                port, worker_port = first.getsockname()[1], second.getsockname()[1]
            api = f'http://127.0.0.1:{port}/api'
            report['controller'] = api
            save(run / 'report.json', report)
            with (run / 'service.log').open('wb') as log:
                service = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/ais_desktop_service.py'),
                                            '--port', str(port), '--worker-port', str(worker_port)],
                                           stdout=log, stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(max_workers=1) as pool:
            loading = pool.submit(load_model, str(model_path))
            expected = set(json.loads((ROOT / 'configs/original-pilot/train-indices.json').read_text()))
            while time.monotonic() - start < 600:
                if service is not None and service.poll() is not None:
                    raise RuntimeError(f'Private desktop service exited with {service.returncode}; see service.log')
                if loading.done():
                    loading.result()
                try:
                    indices, _ = rpc(api, '/get_indices?name=osworld-pilot', timeout=4)
                    if isinstance(indices, list) and {str(i) for i in indices if i != -1} == expected:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Dedicated controller did not register the 64 pilot tasks')
            task = {'index': TASK, 'with_record': args.full_worker_trace, 'image_width': 1280, 'image_height': 720,
                    'screen_width': 1920, 'screen_height': 1080, 'agent_version': 'scalecua_osworld',
                    'env_version': 'anthropic', 'agent_config': {'relative_coordinate': True}, 'max_rounds': 50}
            report['stage'] = 'desktop_setup'
            save(run / 'report.json', report)
            setup = time.monotonic()
            data, sid = rpc(api, '/start_sample', {'name': 'osworld-pilot', 'index': -1, 'custom_task': task})
            report.update(session_id=sid, desktop_setup_seconds=time.monotonic() - setup)
            if sid is None or data.get('finish') or not data.get('messages'):
                raise RuntimeError('No live OSWorld session: ' + str(data.get('status')))
            model, processor, report['model_load_seconds'] = loading.result()
            report.update(stage='acting', ready_seconds=time.monotonic()-start)
            save(run / 'report.json', report)
            messages, tools = data['messages'], data.get('tools', [])
            for turn in range(1, 52):
                messages = upstream._maybe_filter_to_n_most_recent_images(messages, 5, 5)
                observation = save_observation(messages, run / f'observation-{turn:03}.png')
                started = time.monotonic()
                answer, tokens = generate(model, processor, messages, tools)
                inferred = time.monotonic()
                row = {'turn': turn, 'observation': observation, 'message': answer, 'tokens': tokens,
                       'inference_seconds': inferred - started}
                report['turns'].append(row)
                save(run / 'report.json', report)
                data, _ = rpc(api, '/interact', {'messages': [answer]}, sid)
                row['environment_seconds'] = time.monotonic() - inferred
                if data.get('finish'):
                    status = data.get('status')
                    reward = data.get('reward')
                    if status not in ('completed', 'task limit reached') or type(reward) not in (int,float) or not math.isfinite(reward):
                        raise RuntimeError(f'Unscored terminal state: {status}, reward={reward}')
                    report.update(status='scored', stage='finished', terminal_status=status, reward=reward)
                    break
                incoming = data['messages']
                messages = incoming if incoming and incoming[0].get('role') == 'system' else messages + [answer] + incoming
            else:
                raise RuntimeError('Worker did not return a terminal score')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if sid is not None and report['status'] != 'scored':
            try:
                rpc(api, '/cancel', {}, sid, timeout=20)
            except OSError as exc:
                report['cancel_error'] = str(exc)
        if service is not None:
            if service.poll() is None:
                service.terminate()
            try:
                service.wait(timeout=90)
            except subprocess.TimeoutExpired:
                report['service_cleanup_error'] = 'Desktop service did not stop within 90 seconds'
        report['wall_seconds'] = time.monotonic() - start
        save(run / 'report.json', report)
        print(f"{report['status']}: {run}, reward={report['reward']}", flush=True)


if __name__ == '__main__':
    main()
