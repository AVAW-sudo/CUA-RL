"""Run the frozen Frontier collector alongside a private SFT server and one desktop worker."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

from ais_desktop_service import stop
from ais_frontier_reuse import plan_rank
from ais_frontier_shards import merge_campaign, read, write_shard_rates
from ais_rank_status import read_rank_status
from ais_vm_check import HTTP, ROOT, save
from ais_worker_cache import stage_worker

TASK = 'os_13584542-872b-42d8-b299-866967b5c3ef_traj_verify_2'


def check_services(policy, desktop, out):
    for name, process, logs in (
        ('vLLM', policy, ('policy/server.log', 'policy.log')),
        ('OSWorld worker', desktop, ('desktop.log',)),
    ):
        code = process.poll()
        if code is None:
            continue
        message = f'{name} exited with code {code}'
        for relative in logs:
            path = out / relative
            try:
                with path.open('rb') as stream:
                    end = stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, end - 65536))
                    tail = stream.read().decode(errors='replace')
            except OSError:
                continue
            match = re.search(r'\b(?:\w+Error|Exception): [^\r\n]+', tail)
            if match:
                raise RuntimeError(f'{message}. {match.group()[:1800]} ({path})')
        raise RuntimeError(f'{message}. See {out / logs[-1]}')


@contextmanager
def collector_services(out, indices, expected, status, screen_lanes):
    base = ROOT.parent.parent
    binding = []
    with ExitStack() as sockets:
        for _ in range(3):
            sock = sockets.enter_context(socket.socket())
            sock.bind(('127.0.0.1', 0))
            binding.append(sock.getsockname()[1])
    controller_port, worker_port, policy_port = map(str, binding)
    controller = 'http://127.0.0.1:' + controller_port + '/api'
    service_dir = out / 'policy'
    service_state = service_dir / 'service.json'
    pool = out / 'pool.json'
    save(pool, {'checkpoint': 'osworld-sft', 'service_states': [str(service_state)]})
    status['controller'] = controller
    save(out / 'status.json', status)
    with ExitStack() as stack:
        def launch(name, cmd, env, directory=out, owner=stack):
            with (directory / (name + '.log')).open('wb') as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owner.callback(stop, child, 70)
            return child
        policy_command = [sys.executable, '-u', str(ROOT / 'scripts/ais_frontier_server.py'),
                          '--state-dir', str(service_dir), '--port', policy_port]
        if screen_lanes:
            policy_command.append('--local-weights')
        policy = launch('policy', policy_command, os.environ)
        runtime, status['worker_cache'] = stage_worker(Path(os.getenv(
            'CUA_WORKER_MANIFEST', base / 'ENV/cache/ais-osworld-worker/manifest.json')))
        cpu_env = dict(os.environ, PYTHONPATH=str(runtime / 'site'))
        if 'CUA_SCREEN_PROTOCOL' in cpu_env:
            cpu_env.setdefault('CUA_SOURCE_DIR', str(runtime / 'SCALE-CUA'))
        executable = sys.executable if sys.version_info[:2] == (3, 10) else str(base / 'ENV/conda_envs/test-gpu/bin/python')
        cpu = [os.getenv('CUA_WORKER_PYTHON', executable), '-S', '-u']
        cpu_env.setdefault('CUA_FRONTIER_WARM_GUEST', '1')
        desktop_command = cpu + [str(ROOT / 'scripts/ais_desktop_service.py'),
                                 '--port', controller_port, '--worker-port', worker_port]
        if screen_lanes is not None:
            desktop_command += ['--task-indices', str(indices), '--concurrency', str(screen_lanes)]
        desktop = launch('desktop', desktop_command, cpu_env)
        until = time.monotonic() + 3600
        while time.monotonic() < until:
            check_services(policy, desktop, out)
            try:
                if json.loads(service_state.read_text())['status'] == 'ready':
                    with HTTP.open(controller + '/get_indices?name=osworld-pilot', timeout=4) as response:
                        if {str(i) for i in json.load(response) if i != -1} == expected:
                            break
            except (OSError, ValueError):
                pass
            time.sleep(3)
        else:
            raise TimeoutError('Frontier worker and policy were not ready within one hour')
        status['status'] = 'collecting'
        save(out / 'status.json', status)

        def collect(destination, indices, probe=False, reused_protocol=None, check_initial_state=False):
            command = cpu + [str(ROOT / 'scripts/collect_frontier_rates.py'), '--task-name', 'osworld-pilot',
                             '--indices', str(indices), '--primary-service', str(service_state), '--service-pool', str(pool),
                             '--controller', controller, '--output', str(destination / 'episodes'),
                             '--concurrency', str(screen_lanes or 1),
                             '--rates-output', str(destination / 'rates.json')]
            if (screen_lanes or 1) == 1:
                command += ['--private-worker', f'http://127.0.0.1:{worker_port}/api']
            if screen_lanes:
                command.append('--screening')
            if check_initial_state:
                command.append('--check-initial-state')
            if reused_protocol:
                command += ['--reused-protocol', str(reused_protocol)]
            if probe:
                command += ['--probe-one', TASK]
            with ExitStack() as attempt:
                collector = launch('collector', command, cpu_env, directory=destination, owner=attempt)
                next_report = 0
                while collector.poll() is None:
                    check_services(policy, desktop, out)
                    if screen_lanes and time.monotonic() >= next_report:
                        scored = sum(1 for _ in (destination / 'episodes').glob('*/*/result.json'))
                        status.update(scored=scored, trials=4 * len(read(indices)))
                        save(out / 'status.json', status)
                        print(f"Screening rank {status['rank']} | {scored}/{status['trials']} terminal scores", flush=True)
                        next_report = time.monotonic() + 60
                    time.sleep(1)
                if collector.returncode:
                    raise RuntimeError(f'Collector exited with {collector.returncode}; see collector.log and episode attempts')

        yield collect


def run_collector(out, indices, expected, status, probe, reused_protocol, *, screen_lanes=None, check_initial_state=False):
    with collector_services(out, indices, expected, status, screen_lanes) as collect:
        collect(out, indices, probe, reused_protocol, check_initial_state)


def wait_for_shards(directory):
    pending = set(range(8))
    until = time.monotonic() + int(os.environ.get('CUA_FRONTIER_JOIN_SECONDS', '86400'))
    print(f'Waiting for Frontier shards in {directory}', flush=True)
    while pending:
        for rank in sorted(pending):
            path = directory / f'rank-{rank}/status.json'
            record = read_rank_status(path, missing_ok=True)
            if record is None:
                continue
            if record['status'] == 'failed':
                raise RuntimeError(f"Frontier rank {rank} failed: {record.get('error')} ({path})")
            if record['status'] == 'shard_complete':
                pending.remove(rank)
                print(f'Frontier rank {rank} complete, {len(pending)} remaining', flush=True)
        if pending:
            if time.monotonic() >= until:
                raise TimeoutError(f'Waiting for Frontier ranks {sorted(pending)} at {directory}')
            time.sleep(5)


def verify_and_publish(directory, plan, indices, rates_link):
    original_file = ROOT / 'runs/frontier-pilot-initialization/protocol.json'
    original, manifest = read(original_file), read(plan / 'manifest.json')
    for rank in range(8):
        audit = plan_rank(directory, rank, plan, original, manifest)
        if audit['counts'] != {'keep': 32}:
            raise ValueError(f"Rank {rank} still needs verified trials: {audit['counts']}")
    verified = merge_campaign(directory, plan, indices, original_file)
    if rates_link is not None:
        if rates_link.exists() and not rates_link.is_symlink():
            raise FileExistsError(f'Existing rates path is a regular file: {rates_link}')
        with tempfile.TemporaryDirectory(dir=rates_link.parent) as tmp:
            link = Path(tmp) / 'rates'
            link.symlink_to((verified / 'initial_pass_rates.json').resolve())
            link.replace(rates_link)
    print(f'Frontier verified rates: {verified / "initial_pass_rates.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--merge', metavar='JOB_ID', help='Verify and publish the completed eight-shard campaign')
    mode.add_argument('--probe', action='store_true', help='Run one isolated official-protocol desktop trial')
    mode.add_argument('--plan-reuse', metavar='JOB_ID', help='Audit all historical slots without starting services')
    mode.add_argument('--reuse-from', metavar='JOB_ID', help='Start a new campaign with verified slots from this job')
    parser.add_argument('--publish-rates', type=Path, metavar='LINK',
                        help='Atomically point LINK to the successfully verified campaign rates')
    args = parser.parse_args()
    if args.publish_rates and (args.plan_reuse or args.probe):
        parser.error('--publish-rates requires a full campaign or --merge')
    source_job = args.reuse_from or args.plan_reuse
    if source_job and not re.fullmatch(r'[0-9]+', source_job):
        parser.error('Reuse requires a numeric source AIS job ID')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    selection = ROOT / 'configs/original-pilot/train-indices.json'
    indices = selection
    plan = ROOT / 'runs/ais-reproduction/frontier-8pod-plan'
    if args.merge:
        if not re.fullmatch(r'[0-9]+', args.merge):
            parser.error('A numeric AIS job ID is required')
        directory = ROOT / 'runs/ais-frontier-campaigns' / args.merge
        verify_and_publish(directory, plan, selection, args.publish_rates)
        return
    if source_job:
        source = ROOT / 'runs/ais-frontier-campaigns' / source_job
        original = read(ROOT / 'runs/frontier-pilot-initialization/protocol.json')
        manifest = read(plan / 'manifest.json')
        if hashlib.sha256(indices.read_bytes()).hexdigest() != manifest['source_indices_sha256']:
            raise ValueError('Frozen pilot selection has changed')
        if args.plan_reuse:
            print(json.dumps([plan_rank(source, r, plan, original, manifest) for r in range(8)]))
            return
    expected = set(json.loads(indices.read_text()))
    if args.probe:
        if int(os.getenv('RANK', '0')) != 0:
            return
        parent = ROOT / 'runs/ais-frontier-probes'
        parent.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix='trial-', dir=parent))
    else:
        rank = int(os.environ['RANK'])
        job = os.environ['JOB_ID']
        if int(os.environ['WORLD_SIZE']) != 8 or rank not in range(8) or not re.fullmatch(r'[0-9]+', job):
            raise ValueError('Frontier collection requires eight assigned AIS ranks and a numeric JOB_ID')
        manifest = json.loads((plan / 'manifest.json').read_text())
        if hashlib.sha256(indices.read_bytes()).hexdigest() != manifest['source_indices_sha256']:
            raise ValueError('Frozen pilot selection has changed')
        indices = plan / f'rank-{rank}.json'
        if hashlib.sha256(indices.read_bytes()).hexdigest() != manifest['sha256'][indices.name]:
            raise ValueError('Unexpected Frontier shard')
        if job == args.reuse_from:
            raise ValueError('Reuse requires a new AIS job ID')
        out = ROOT / f'runs/ais-frontier-campaigns/{job}/rank-{rank}'
        out.mkdir(parents=True, exist_ok=False)
    print(f'AIS Frontier output: {out}', flush=True)
    status = {'status': 'starting', 'indices': str(indices)}
    save(out / 'status.json', status)
    try:
        kept = 0
        reused_protocol = None
        if args.reuse_from:
            reused = plan_rank(source, rank, plan, original, manifest, out)
            kept = reused['counts'].get('keep', 0)
            if kept:
                reused_protocol = source / f'rank-{rank}/episodes/protocol.json'
            print(f"Frontier rank {rank} audit: {reused['counts']}", flush=True)
        if kept == 32:
            protocol = read(reused_protocol)
            save(out / 'episodes/protocol.json', protocol)
            write_shard_rates(out / 'episodes', out / 'rates.json', protocol, read(indices),
                              protocol['services'][0]['served_model_name'])
        else:
            run_collector(out, indices, expected, status, args.probe, reused_protocol)
        status['status'] = 'scored' if args.probe else 'shard_complete'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save(out / 'status.json', status)
    if not args.probe and rank == 0:
        wait_for_shards(out.parent)
        verify_and_publish(out.parent, plan, selection, args.publish_rates)


if __name__ == '__main__':
    main()
