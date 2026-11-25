"""Run a dedicated upstream AgentRL controller and OSWorld pilot worker on AIS."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError

from ais_vm_check import ROOT, HTTP, save


def stop(process, grace_seconds=20):
    # The foreground worker and its QEMU children own this private process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-worker', action='store_true',
                        help='Serve the configured pilot worker capacity to the RL training pod')
    parser.add_argument('--host')
    parser.add_argument('--advertise-host')
    parser.add_argument('--port', type=int, default=18520)
    parser.add_argument('--worker-port', type=int, default=18521)
    parser.add_argument('--controller-url', help='Register only a worker with this job controller')
    parser.add_argument('--task-indices', type=Path, help='Serve this subset of the OSWorld research training split')
    parser.add_argument('--concurrency', type=int, choices=range(1, 33), metavar='1..32')
    parser.add_argument('--ready-file', type=Path, help='Publish the controller address after worker registration')
    args = parser.parse_args()
    if args.rl_worker:
        args.advertise_host = (args.advertise_host or os.getenv('POD_IP')
                               or socket.gethostbyname(socket.gethostname()))
        args.host = args.host or args.advertise_host
    else:
        args.host = args.host or '127.0.0.1'
        args.advertise_host = args.advertise_host or '127.0.0.1'
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    from ais_worker_cache import stage_runtime, stage_worker
    cache_info = None
    cached = None
    manifest = Path(os.getenv('CUA_WORKER_MANIFEST', ROOT.parent.parent / 'ENV/cache/ais-osworld-worker/manifest.json'))
    if not os.getenv('CUA_WORKER_SITE') and manifest.is_file():
        cached, cache_info = stage_worker(manifest)
    base_dir = cached or ROOT
    source = Path(os.getenv('CUA_SOURCE_DIR', base_dir / 'SCALE-CUA')).resolve()
    data = Path(os.getenv('CUA_DATA_DIR', base_dir / 'data/ScaleCUA')).resolve()
    base = Path(os.getenv('CUA_VM_IMAGE', ROOT / 'data/osworld-vm/Ubuntu.qcow2')).resolve(strict=True)
    default_python = ROOT.parent.parent / 'ENV/cua-rl-osworld/bin/python'
    if cached:
        default_python = sys.executable if sys.version_info[:2] == (3, 10) else ROOT.parent.parent / 'ENV/conda_envs/test-gpu/bin/python'
    worker_python = os.getenv('CUA_WORKER_PYTHON', str(default_python))
    parent = ROOT / 'runs/ais-desktop-services'
    parent.mkdir(exist_ok=True, parents=True)
    run = Path(tempfile.mkdtemp(prefix='worker-', dir=parent))
    print(f'AIS desktop service: {run}', flush=True)
    expected = set(json.loads((ROOT / 'configs/original-pilot/train-indices.json').read_text()))
    examples = ROOT / 'configs/original-pilot/train-examples.json'
    if args.task_indices is not None:
        selected = json.loads(args.task_indices.read_text())
        catalogue = json.loads((ROOT / 'configs/research-splits/train-examples.json').read_text())
        expected = set(selected)
        groups = {app: [i for i in ids if i in expected] for app, ids in catalogue.items()}
        if not expected or len(expected) != len(selected) or sum(map(len, groups.values())) != len(expected):
            raise ValueError('Screening requires unique task IDs from the research training split')
        examples = run / 'examples.json'
        save(examples, groups)
    config = yaml.safe_load((ROOT / 'configs/osworld-pilot-worker.yaml').read_text())
    entry = config['osworld-pilot']
    entry['module'] = 'ais_osworld_worker.AisOSWorldTask'
    params = entry['parameters']
    params['first_observation_timeout'] = 240 if args.rl_worker else 600
    if not args.rl_worker:
        params['concurrency'] = 1
    if args.concurrency is not None:
        params['concurrency'] = args.concurrency
    params.pop('warm_reset_report', None)
    env = os.environ.copy()
    paths = [str(ROOT / 'scripts'), str(source / 'scalecua_rl/worker/src'),
             str(source / 'osworld_eval'), str(source / 'osworld_env/worker')]
    worker_site = env.get('CUA_WORKER_SITE') or (str(cached / 'site') if cached else None)
    if worker_site:
        paths.append(worker_site)
    env.update(PYTHONPATH=os.pathsep.join(paths), OSWORLD_HELPER_ASSETS=str(source / 'osworld_env/worker/extra'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env['no_proxy'] = env['NO_PROXY'] = ','.join(['localhost', '127.0.0.0/8', args.advertise_host, env.get('NO_PROXY', '')])
    api = f'http://{args.advertise_host}:{args.port}/api'
    status = {'status': 'starting', 'host': socket.gethostname(), 'controller': api,
              'image': str(base), 'worker_python': worker_python, 'concurrency': params['concurrency'], 'runtime_cache': cache_info}
    binary = cached / 'bin/agentrl-controller' if cached else ROOT / 'runs/bin/agentrl-controller'
    try:
        with ExitStack() as stack:
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='cua-worker-')))
            if os.getenv('CUA_FRONTIER_WARM_GUEST') == '1':
                from ais_vm_check import build_sft_template
                from ais_verifier import prepare_guest_verifier
                from qemu_runtime import require_kvm
                from sft_vm_image import verified_image
                base, status['image_sha256'] = verified_image('osworld', base, 24460197888)
                try:
                    require_kvm()
                except RuntimeError:
                    template = scratch / 'guest-template'
                    template.mkdir()
                    build_sft_template(base, template, run, prepare_guest=prepare_guest_verifier)
                    env['CUA_RL_VM_TEMPLATE'] = str(template)
                status['image'] = str(base)
                status['warm_restore'] = 'CUA_RL_VM_TEMPLATE' in env
            task_cache = os.getenv('CUA_TASK_CACHE')
            fixture_bundle = 'ais-sft-osworld-inputs' if args.task_indices else 'ais-pilot-fixtures'
            fixture_manifest = ROOT.parent.parent / 'ENV/cache' / fixture_bundle / 'manifest.json'
            if not task_cache and fixture_manifest.is_file():
                fixtures, status['setup_assets_cache'] = stage_runtime(fixture_manifest, 'fixtures')
                task_cache = str(fixtures / 'tasks')
            params.update(images_dir=str(base.parent), ais_image=str(base), ais_audit=str(run / 'vms'),
                          data_path=str(data / 'osworld/rl_tasks'), examples_path=str(examples),
                          judge_functions_path=str(data / 'osworld/judge_functions/rl_tasks'),
                          cache_dir=task_cache or str(scratch / 'cache'), record_dir=str(run / 'records'))
            cfg = run / 'worker.yaml'
            cfg.write_text(yaml.safe_dump(config, sort_keys=False))
            # Port zero requests a private port for this job. Reserve the pair
            # together so they are distinct, immediately before starting the servers.
            with ExitStack() as reservations:
                ports = []
                for port in ([args.worker_port] if args.controller_url else [args.port, args.worker_port]):
                    sock = reservations.enter_context(socket.socket())
                    sock.bind((args.host, port))
                    ports.append(sock.getsockname()[1])
            args.worker_port = ports[-1]
            api = args.controller_url or f'http://{args.advertise_host}:{ports[0]}/api'
            worker_url = f'http://{args.advertise_host}:{args.worker_port}/api'
            status.update(controller=api, worker=worker_url)
            save(run / 'status.json', status)
            controller = [str(binary), 'controller', '--host', args.host, '--port', str(ports[0]),
                          '--dashboard=false', '--long-timeout']
            if args.rl_worker or args.task_indices:
                controller += ['--start-timeout', f"{params['first_observation_timeout'] + 30}s"]
            worker = [worker_python, *(['-S'] if worker_site else []), '-m', 'agentrl.worker', '-c', str(cfg), '--controller', api,
                      '--host', args.host, '--port', str(args.worker_port), '--self',
                      worker_url, 'osworld-pilot']
            children = []
            services = [] if args.controller_url else [('controller', controller)]
            for name, cmd in [*services, ('worker', worker)]:
                log = stack.enter_context((run / f'{name}.log').open('wb'))
                child = subprocess.Popen(cmd, env=env, cwd=source / 'osworld_eval', stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                stack.callback(stop, child)
            deadline = time.monotonic() + 600
            while all(p.poll() is None for p in children):
                if status['status'] == 'starting':
                    try:
                        with HTTP.open(api + '/get_indices?name=osworld-pilot', timeout=4) as response:
                            indices = json.load(response)
                        with HTTP.open(api + '/list_workers', timeout=4) as response:
                            workers = json.load(response).get('osworld-pilot', {}).get('workers', {})
                        registered = any(w['address'].rstrip('/') == worker_url and w['status'] == 'ALIVE'
                                         and not w['stale'] and w['capacity'] == params['concurrency']
                                         for w in workers.values())
                        if registered and isinstance(indices, list) and {str(i) for i in indices if i != -1} == expected:
                            status.update(status='ready', pilot_tasks=len(expected))
                            save(run / 'status.json', status)
                            if args.ready_file:
                                save(args.ready_file, status)
                            print(f'Pilot worker ready: {api}, {len(expected)} tasks', flush=True)
                            if args.rl_worker:
                                print(f'CUA_CONTROLLER_URL={api}', flush=True)
                    except (URLError, ValueError, TimeoutError):
                        pass
                    if status['status'] == 'starting' and time.monotonic() > deadline:
                        raise TimeoutError('Pilot task set was not registered within 600 seconds')
                time.sleep(2)
            raise RuntimeError(f'Controller/worker exited: {[p.poll() for p in children]}; see logs in {run}')
    except (KeyboardInterrupt, SystemExit):
        status['status'] = 'stopped'
        raise
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        raise
    finally:
        save(run / 'status.json', status)


if __name__ == '__main__':
    main()
