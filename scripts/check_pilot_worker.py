"""Validate the selected task pool and one real remote initialization/control/cancel."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from run_benchmark import ROOT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--hostname', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    expected = set(json.loads((ROOT / 'configs/original-pilot/train-indices.json').read_text()))
    result = {'status': 'running', 'hostname': args.hostname, 'started_at_utc': now(),
              'worker_config_sha256': hashlib.sha256((ROOT / args.config).read_bytes()).hexdigest(),
              'task_name': 'osworld-pilot', 'gpu_allocation': 0,
              'indices_sha256': hashlib.sha256((ROOT / 'configs/original-pilot/train-indices.json').read_bytes()).hexdigest()}
    def request(route, body=None, sid=None):
        headers = {'Content-Type': 'application/json'}
        if sid is not None: headers['session_id'] = str(sid)
        req = urllib.request.Request(args.address + route, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        with opener.open(req, timeout=900) as response:
            return json.load(response), response.headers.get('session_id')
    sid = None
    try:
        data, _ = request('/get_indices?name=osworld-pilot')
        actual = set(str(x) for x in data if x != -1)
        if actual != expected: raise RuntimeError(f'Expected exact 64 task indices, got {len(actual)}')
        result['indices_count'] = len(actual)
        started = time.monotonic()
        data, sid = request('/start_sample', {'name': 'osworld-pilot', 'index': -1, 'custom_task': {
            'index': sorted(i for i in expected if i.startswith('os_'))[0], 'with_record': False, 'image_width': 1280, 'image_height': 720,
            'screen_width': 1920, 'screen_height': 1080, 'agent_version': 'scalecua_osworld',
            'env_version': 'anthropic', 'agent_config': {'relative_coordinate': True}, 'max_rounds': 50}})
        result['initialization_seconds'] = time.monotonic() - started
        result['session_id'] = sid
        if data.get('finish') or not data.get('messages'): raise RuntimeError(str(data))
        sessions, _ = request('/list_sessions')
        selected = sessions[str(sid)]
        workers, _ = request('/list_workers')
        worker = workers['osworld-pilot']['workers'][str(selected['worker_id'])]
        if args.hostname not in worker['address']:
            raise RuntimeError('Smoke was scheduled on the wrong worker: ' + str(worker))
        result['worker'] = worker
        result['live_observation'] = True
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        if sid:
            try:
                result['cancel_result'], _ = request('/cancel', {}, sid)
            except Exception as exc:
                result.update(status='failed', cancel_error=repr(exc))
        result['finished_at_utc'] = now()
        write_json(args.report, result)


if __name__ == '__main__': main()
