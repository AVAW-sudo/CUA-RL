"""Check cross-node OSWorld initialization and screenshot control without a GPU."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from run_benchmark import ROOT, now, write_json


def main():
    controller = 'http://research-a-worker-1.job-research-a.training.svc.cluster.local:18520/api'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    target = ROOT / 'reports/distributed-osworld-worker0-smoke.json'
    config = ROOT / 'configs/osworld-original-worker0.yaml'
    report = {'status': 'running', 'started_at_utc': now(), 'hostname': 'research-a-worker-0',
              'worker_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'gpu_allocation': 0, 'scope': 'Cross-node environment initialization and one scripted screenshot; not a rollout score'}
    if target.exists():
        archive = target.with_name(target.stem + f'-attempt-{time.time_ns()}.json')
        archive.write_bytes(target.read_bytes())
    write_json(target, report)
    session = None

    def rpc(route, body=None, method='POST', sid=None):
        headers = {'Content-Type': 'application/json'}
        if sid is not None:
            headers['session_id'] = str(sid)
        request = urllib.request.Request(controller + route,
                                         data=json.dumps(body).encode() if body is not None else None,
                                         headers=headers, method=method)
        with opener.open(request, timeout=900) as response:
            return json.load(response), response.headers.get('session_id')

    try:
        workers, _ = rpc('/list_workers', method='GET')
        workers = workers['osworld-rl']['workers']
        alive = [w for w in workers.values() if w['status'] == 'ALIVE' and not w['stale']]
        if len(alive) != 1 or 'research-a-worker-0.' not in alive[0]['address']:
            raise RuntimeError('Run this preflight with only the secondary worker registered')
        smoke_index = json.loads((ROOT / 'configs/osworld-smoke-indices.json').read_text())[0]
        candidates = [i for i in json.loads((ROOT / 'configs/rl-indices.json').read_text())
                      if i == smoke_index or i.endswith('_' + smoke_index)]
        if len(candidates) != 1:
            raise RuntimeError('Smoke task must resolve to one canonical full-pool index')
        index = candidates[0]
        report['task_index'] = index
        started = time.monotonic()
        data, session = rpc('/start_sample', {'name': 'osworld-rl', 'index': -1,
            'custom_task': {'index': index, 'with_record': False, 'image_width': 1280,
                            'image_height': 720, 'screen_width': 1920, 'screen_height': 1080,
                            'agent_version': 'scalecua_osworld', 'env_version': 'anthropic',
                            'agent_config': {'relative_coordinate': True}, 'max_rounds': 50}})
        report.update(session_id=session, initialization_seconds=time.monotonic() - started,
                      worker=alive[0], message_count=len(data.get('messages', [])))
        if not session or not data.get('messages') or data.get('finish'):
            raise RuntimeError('No live initial observation')
        message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'remote-screenshot-check',
                   'type': 'function', 'function': {'name': 'computer',
                   'arguments': json.dumps({'action': 'screenshot'})}}]}
        started = time.monotonic()
        data, _ = rpc('/interact', {'messages': [message]}, sid=session)
        if not data.get('messages') or data.get('finish'):
            raise RuntimeError('Screenshot control did not return a live observation')
        report.update(status='passed', screenshot_control_seconds=time.monotonic() - started)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
    finally:
        if session:
            try:
                rpc('/cancel', sid=session)
                report['cancelled'] = True
            except Exception as exc:
                report.update(status='failed', cleanup_error=str(exc))
        report['finished_at_utc'] = now()
        write_json(target, report)
    print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
