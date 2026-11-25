"""Run one Ray trainer inside the eight single-GPU entries of an AIS job."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

from ais_desktop_service import stop
from ais_rank_status import read_rank_status
from ais_vm_check import ROOT, save

WORLD = 8


def free_ports(ip, count):
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind((ip, 0))
        return [sock.getsockname()[1] for sock in sockets]


def allocation(receipt):
    import torch
    from ais_gpu_check import check_all_ranks

    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != WORLD or torch.cuda.device_count() != 1:
        raise RuntimeError('Distributed RL requires eight AIS ranks, each with one visible GPU')
    ip = os.getenv('POD_IP') or socket.gethostbyname(socket.gethostname())
    local = {'ip': ip}
    if rank == 0:
        parent = ROOT / 'runs/ais-distributed-rl'
        parent.mkdir(exist_ok=True, parents=True)
        run = Path(tempfile.mkdtemp(dir=parent, prefix='job-' + os.getenv('JOB_ID', 'local') + '-'))
        token = run / 'ray-token'
        with os.fdopen(os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
            output.write(secrets.token_hex(32))
        local.update(run=str(run), address=f'{ip}:{free_ports(ip, 1)[0]}')
    props = torch.cuda.get_device_properties(0)
    report = {'devices': [{'name': props.name, 'uuid': str(props.uuid) if props.uuid else None}]}
    check_all_ranks(torch, 0, receipt, report, runtime=local)
    peers = report['distributed_nccl']['participants']
    allocation_info = dict(peers[0]['runtime'], participants=peers, nccl_sum=report['distributed_nccl']['sum'])
    if rank == 0:
        save(Path(local['run']) / 'allocation.json', allocation_info)
    save(receipt, allocation_info)


def ray_command(rank, ip, address, tmp):
    agent_port, client_port, dashboard_port = free_ports(ip, 3)
    command = [sys.executable, '-u', '-m', 'ray.scripts.scripts', 'start', '--block',
               '--node-ip-address', ip, '--node-name', f'ais-rank-{rank}',
               '--num-gpus', '1', '--num-cpus', '8', '--object-store-memory', str(2 * 1024**3),
               '--resources', json.dumps({'big_mem': 8, f'ais_rank_{rank}': 1}),
               '--min-worker-port', '0', '--max-worker-port', '0', '--dashboard-agent-listen-port', str(agent_port)]
    if rank == 0:
        command += ['--head', '--port', address.rsplit(':', 1)[1], '--temp-dir', tmp,
                    '--include-dashboard=false', '--dashboard-port', str(dashboard_port), '--disable-usage-stats', '--ray-client-server-port', str(client_port)]
    else:
        command += ['--address', address]
    return command


def ray_environment(info):
    ips = [p['runtime']['ip'] for p in info['participants']]
    controller_host = urlsplit(os.getenv('CUA_CONTROLLER_URL', '')).hostname
    bypass = ','.join(dict.fromkeys(filter(None, ['127.0.0.1', 'localhost', *ips, controller_host,
                                                  os.getenv('NO_PROXY'), os.getenv('no_proxy')])))
    inherited = dict(os.environ)
    inherited.pop('RAY_AUTH_TOKEN', None)
    inherited.pop('RAY_ADDRESS', None)
    return dict(inherited, NO_PROXY=bypass, no_proxy=bypass, RAY_AUTH_MODE='token',
                RAY_AUTH_TOKEN_PATH=str(Path(info['run']) / 'ray-token'), RAY_USAGE_STATS_ENABLED='0',
                CUA_DISTRIBUTED_RUN=info['run'], CUA_RAY_ADDRESS=info['address'],
                VLLM_WORKER_MULTIPROC_METHOD='spawn', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')


def gpu_probe():
    import ray
    import torch
    if torch.cuda.device_count() != 1 or torch.arange(8, device='cuda').sum().item() != 28:
        raise RuntimeError('Ray worker did not receive one usable CUDA device')
    return ray.get_runtime_context().get_node_id(), str(torch.cuda.get_device_properties(0).uuid)


def wait_for_head(address):
    from ray._raylet import GcsClient
    from ray.core.generated.gcs_pb2 import GcsNodeInfo

    # The GCS port opens before the head raylet is registered.
    client = GcsClient(address=address)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        nodes = client.get_all_node_info(timeout=5, state_filter=GcsNodeInfo.GcsNodeState.ALIVE)
        if any(node.is_head_node for node in nodes.values()):
            return
        time.sleep(1)
    raise TimeoutError('Ray GCS did not register the head raylet')


def verify_cluster(info):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    wait_for_head(info['address'])
    save(Path(info['run']) / 'head.json', {'address': info['address']})
    ray.init(address=info['address'])
    try:
        probe = ray.remote(num_gpus=1, max_calls=1)(gpu_probe)
        tasks = [probe.options(resources={f"ais_rank_{p['rank']}": 1}).remote() for p in info['participants']]
        results = ray.get(tasks, timeout=240)
        expected = [p['uuid'] for p in info['participants']]
        node_ids = {node for node, _ in results}
        nodes = [n for n in ray.nodes() if n['Alive']]
        if (len(results) != WORLD or len(node_ids) != WORLD or [uuid for _, uuid in results] != expected
                or {n['NodeID'] for n in nodes} != node_ids or any(n['Resources'].get('GPU') != 1 for n in nodes)):
            raise RuntimeError(f'Ray GPU placement differs from AIS allocation: {results}')
        groups = [placement_group([{'CPU': 1, 'GPU': 1}] * count) for count in (2, 6)]
        try:
            ray.get([group.ready() for group in groups], timeout=90)
        finally:
            for group in groups:
                remove_placement_group(group)
        save(Path(info['run']) / 'placement.json', {'nodes_and_uuids': results, 'rollout': 2, 'actor_reference': 6})
        print('Ray placement passed: eight distinct nodes and assigned GPUs, 2 rollout + 6 actor/reference', flush=True)
    finally:
        ray.shutdown()


def supervise(info, rank, local_dir, args):
    run = Path(info['run'])
    print(f'AIS distributed RL rank {rank}: {run}', flush=True)
    env = ray_environment(info)
    # Ray workers create their own process groups after the AIS NCCL rendezvous has exited.
    node_env = {key: value for key, value in env.items()
                if key not in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')
                and not key.startswith('TORCHELASTIC_')}
    ip = info['participants'][rank]['runtime']['ip']
    prom_dir = local_dir / 'prometheus'
    prom_dir.mkdir()
    node_env.update(PROMETHEUS_MULTIPROC_DIR=str(prom_dir), CUA_ROLLOUT_IPC_DIR=str(local_dir))
    env.pop('PROMETHEUS_MULTIPROC_DIR', None)
    endpoint = env.get('CUA_CONTROLLER_URL')
    own_status = run / f'rank-{rank}.json'
    result = run / 'result.json'
    rank_states = [run / f'rank-{i}.json' for i in range(WORLD)]
    watch = rank_states if rank == 0 else rank_states[:1]
    phase = 'startup'
    healthy = {}
    started = time.monotonic()
    last_tick = 0.0

    def wait_for(label, condition, timeout=None):
        nonlocal phase, last_tick
        phase = label
        print(f'AIS rank {rank} waiting for {label}', flush=True)
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            value = condition()
            if value:
                return value
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise TimeoutError(f'rank {rank} timed out during {label}')
            for name, child in healthy.items():
                if child.poll() is not None:
                    raise RuntimeError(f'{name} exited with code {child.returncode}')
            if now - last_tick >= 5:
                os.utime(own_status)
                for path in watch:
                    record = read_rank_status(path, missing_ok=True)
                    if record is None and now - started > 180:
                        raise RuntimeError(f'{path.stem} has no status file')
                    if record:
                        if record['stage'] == 'failed':
                            raise RuntimeError(f"{path.stem}: {record['error']}")
                        if time.time() - path.stat().st_mtime > 180:
                            raise RuntimeError(f'{path.stem} stopped reporting')
                last_tick = now
            time.sleep(1)

    def rank_is_ready(path):
        record = read_rank_status(path, missing_ok=True)
        return record and record['stage'] == 'ready'

    with ExitStack() as stack:
        def start(command, child_env, grace=25):
            child = subprocess.Popen(command, env=child_env, start_new_session=True)
            stack.callback(stop, child, grace)
            return child

        try:
            save(own_status, {'stage': phase})
            if rank:
                wait_for('Ray head', (run / 'head.json').is_file, 300)
            tmp = '/tmp/cua-ray-' + run.name.rsplit('-', 1)[-1]
            healthy['ray'] = start(ray_command(rank, ip, info['address'], tmp), node_env)
            if rank == 0:
                verifier = start([sys.executable, __file__, '--verify', str(run / 'allocation.json')], env)
                wait_for('Ray placement', lambda: verifier.poll() is not None, 600)
                if verifier.returncode:
                    raise RuntimeError(f'Ray GPU placement check exited with code {verifier.returncode}')
            else:
                wait_for('Ray placement', (run / 'placement.json').is_file, 620)

            if args != ['--check-only']:
                if not endpoint:
                    controller = run / 'controller.json'
                    ready = local_dir / 'desktop.json'
                    command = ['bash', str(ROOT / 'AIS_ENTRY.txt'), 'rl-desktop', '--concurrency', '4',
                               '--host', ip, '--advertise-host', ip, '--port', '0', '--worker-port', '0',
                               '--ready-file', str(ready)]
                    if rank:
                        wait_for('desktop controller', controller.is_file, 1800)
                        command += ['--controller-url', read_rank_status(controller)['url']]
                    healthy['desktop'] = start(command, dict(env, CUDA_VISIBLE_DEVICES=''), 70)
                    wait_for('desktop worker registration', ready.is_file, 1800)
                    endpoint = read_rank_status(ready)['controller']
                    if rank == 0:
                        save(controller, {'url': endpoint})
                save(own_status, {'stage': 'ready', 'controller': endpoint})
                if rank == 0:
                    wait_for('all desktop workers', lambda: all(map(rank_is_ready, rank_states)), 1900)
                    trainer = start([sys.executable, '-u', str(ROOT / 'scripts/launch_local_rl.py'),
                                     '--ais-pilot', *args], dict(env, CUA_CONTROLLER_URL=endpoint), 60)
                    wait_for('GRPO driver', lambda: trainer.poll() is not None)
                    if trainer.returncode:
                        raise RuntimeError(f'GRPO driver exited with code {trainer.returncode}')
            # Keep the head alive until every worker has observed the driver outcome.
            finished = run / 'driver-finished.json'
            if rank == 0:
                outcome = {'passed': True, 'stage': 'placement' if args == ['--check-only'] else 'training'}
                save(finished, outcome)
                receipts = [run / f'finished-{i}.json' for i in range(1, WORLD)]
                wait_for('worker completion receipts', lambda: all(p.is_file() for p in receipts), 180)
                save(result, outcome)
            else:
                wait_for('driver completion', finished.is_file)
                save(run / f'finished-{rank}.json', {'rank': rank})
                wait_for('final job status', result.is_file, 240)
                outcome = read_rank_status(result)
                if not outcome['passed']:
                    raise RuntimeError(outcome['error'])
        except BaseException as exc:
            message = f'{phase}: {type(exc).__name__}: {exc}'
            save(own_status, {'stage': 'failed', 'error': message})
            if rank == 0 and not result.exists():
                save(result, {'passed': False, 'error': message})
            raise


def main(args):
    if '--help' in args or '-h' in args:
        print('Usage: bash AIS_ENTRY.txt rl [--check-only]. Eight single-GPU AIS entries share one driver.')
        return
    if '--check-only' in args and args != ['--check-only']:
        raise SystemExit('--check-only must be used alone')
    if args[:1] == ['--allocation']:
        allocation(Path(args[1]))
    elif args[:1] == ['--verify']:
        verify_cluster(read_rank_status(Path(args[1])))
    else:
        rank = int(os.environ['RANK'])
        if not 0 <= rank < WORLD:
            raise ValueError(f'Invalid AIS rank {rank}')
        with tempfile.TemporaryDirectory(prefix='cua-rl-rank-') as directory:
            receipt = Path(directory) / 'allocation.json'
            # Release the CUDA/NCCL context before this rank starts its Ray workers.
            subprocess.run([sys.executable, __file__, '--allocation', str(receipt)], check=True, timeout=300)
            supervise(json.loads(receipt.read_text()), rank, Path(directory), args)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main(sys.argv[1:])
