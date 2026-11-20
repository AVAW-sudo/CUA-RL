"""Executable stages for released-RL evaluation, Frontier initialization, and original GRPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from checkpoint_specs import checkpoint_spec
from pipeline_supervisor import maintain_campaigns, maintain_services, remote, PYTHON
from run_benchmark import ROOT, check_endpoint, now, write_json
from write_evaluation_report import generate

DIRECTORY = ROOT / 'runs/original-rl-pipeline'
CONFIG = ROOT / 'configs/original-rl-pipeline.json'


def call(arguments, timeout=None):
    return subprocess.run(arguments, cwd=ROOT, check=True, timeout=timeout)


def stop_inference():
    paths = set()
    for path in (ROOT / 'runs/inference-pools').glob('*.json'):
        paths.update(json.loads(path.read_text())['service_states'])
    owned = []
    for path in sorted(paths):
        state = json.loads(Path(path).read_text())
        if state['hostname'] != socket.gethostname():
            continue
        proc = Path(f'/proc/{state["pid"]}/cmdline')
        if not proc.exists() or state['model_dir'].encode() not in proc.read_bytes().split(b'\0'):
            continue
        command = [PYTHON, str(ROOT / 'scripts/serve_model.py'), 'stop', '--checkpoint', state['checkpoint']]
        if state.get('replica'):
            command += ['--replica', state['replica'], '--port', state['command'][state['command'].index('--port') + 1]]
        call(command)
        owned.append(state)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if all(not Path(f'/proc/{s["pid"]}/cmdline').exists() or not Path(f'/proc/{s["pid"]}/cmdline').read_bytes() for s in owned):
            return
        time.sleep(2)
    raise RuntimeError('Owned inference services have not exited; do not overwrite their GPU allocations')


def start_pool(checkpoint, frontier=False):
    spec = checkpoint_spec(ROOT, checkpoint)
    paths = []
    pool = ROOT / 'runs/inference-pools' / (checkpoint + '-pipeline.json')
    for gpu in range(8):
        command = [PYTHON, str(ROOT / 'scripts/serve_model.py'), 'start', '--checkpoint', checkpoint,
                   '--gpus', str(gpu), '--port', str(spec['port'] + gpu),
                   '--max-model-len', '200000' if frontier else '65536', '--max-images', '20',
                   '--max-num-seqs', '8', '--gpu-memory-utilization', '.75',
                   '--top-p-default', '1' if frontier else '.5', '--top-k-default', '-1' if frontier else '20']
        path = ROOT / 'runs' / spec['state_file']
        if gpu:
            command += ['--replica', f'r{gpu}']
            path = ROOT / 'runs' / f'inference-{checkpoint}-r{gpu}.json'
        call(command)
        paths.append(str(path))
        write_json(pool, {'checkpoint': checkpoint, 'service_states': paths})
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            for path in paths:
                check_endpoint(json.loads(Path(path).read_text()))
            return pool, paths
        except Exception:
            time.sleep(10)
    raise RuntimeError('Inference pool did not become ready')


def vm_acceptance():
    path = ROOT / 'reports/tcg-osworld-32vm-pipeline.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'passed':
        return path
    for attempt in range(1, 4):
        current = path.with_name(f'tcg-osworld-32vm-pipeline-attempt{attempt}.json')
        if current.exists():
            if json.loads(current.read_text()).get('status') == 'passed':
                write_json(path, json.loads(current.read_text()))
                return path
            continue
        result = subprocess.run([PYTHON, str(ROOT / 'scripts/check_vm_concurrency.py'), '--accelerator', 'tcg',
                                 '--stages', '32', '--cpus', '4', '--memory-mb', '8192', '--rounds', '3',
                                 '--timeout', '300', '--launch-interval', '1', '--report', str(current)], cwd=ROOT)
        if result.returncode == 0:
            write_json(path, json.loads(current.read_text()))
            return path
    raise RuntimeError('OSWorld 32-VM acceptance exhausted its bounded retries; inspect saved diagnostics')


def evaluate(checkpoint):
    existing = [ROOT / 'runs/benchmarks' / f'{checkpoint}__{benchmark}-original-pipeline-v1'
                for benchmark in ['osworld', 'scienceboard']]
    if any((path / 'campaign.json').exists() for path in existing):
        # A crash can occur between the two campaign launches. Keep every
        # existing episode and endpoint alive while creating the missing queue.
        services = json.loads((ROOT / 'runs/inference-pools' / (checkpoint + '-pipeline.json')).read_text())['service_states']
        if len(services) != 8:
            raise RuntimeError('Existing campaign requires its complete eight-replica pool')
        acceptance = ROOT / 'reports/tcg-osworld-32vm-pipeline.json'
    else:
        stop_inference()
        acceptance = vm_acceptance()
        _, services = start_pool(checkpoint)
    paths = []
    for index, benchmark in enumerate(['osworld', 'scienceboard']):
        half = services[index * 4:(index + 1) * 4]
        subpool = ROOT / 'runs/inference-pools' / f'{checkpoint}-{benchmark}.json'
        write_json(subpool, {'checkpoint': checkpoint, 'service_states': half})
        directory = ROOT / 'runs/benchmarks' / f'{checkpoint}__{benchmark}-original-pipeline-v1'
        if not (directory / 'campaign.json').exists():
            report = acceptance if benchmark == 'osworld' else ROOT / 'reports/tcg-scienceboard-32vm-paced-worker-1-20261120.json'
            call([PYTHON, str(ROOT / 'scripts/run_benchmark.py'), 'start', '--benchmark', benchmark,
                  '--checkpoint', checkpoint, '--workers', '16', '--proxy-policy', 'direct',
                  '--concurrency-report', str(report), '--service-pool', str(subpool), '--run-dir', str(directory)])
        paths.append(str(directory))
    state = {}
    while maintain_campaigns(paths, state):
        maintain_services(services, state)
        write_json(DIRECTORY / f'{checkpoint}-evaluation-watch.json', state)
        generate(paths, ROOT / 'reports' / f'{checkpoint}-evaluation-report')
        time.sleep(20)
    generate(paths, ROOT / 'reports' / f'{checkpoint}-evaluation-report', final=True)
    comparison_report()
    stop_inference()


def comparison_report():
    paths = json.loads(CONFIG.read_text())['sft_campaigns']
    for checkpoint in ['osworld-rl-reference', 'reproduced-rl']:
        paths += [str(path) for path in (ROOT / 'runs/benchmarks').glob(checkpoint + '__*-original-pipeline-v1')
                  if (path / 'campaign.json').exists()]
    generate(paths, ROOT / 'reports/sft-and-original-rl-comparison', final=True)


def secondary_worker_start():
    cfg = json.loads(CONFIG.read_text()).get('secondary_environment_worker', {})
    result = {'at_utc': now(), 'enabled': False, 'additional_capacity': 0, 'gpu_allocation': 0}
    try:
        if not cfg.get('enabled'):
            return 0
        host = cfg['hostname']
        smoke = json.loads((ROOT / cfg['rpc_smoke_report']).read_text())
        if smoke.get('status') != 'passed' or smoke.get('hostname') != host:
            raise RuntimeError('Cross-node episode initialization/control has not passed')
        if smoke.get('worker_config_sha256') != hashlib.sha256((ROOT / cfg['config']).read_bytes()).hexdigest():
            raise RuntimeError('Secondary worker configuration changed since its RPC smoke')
        # Validate acceptance on its own host; no GPU is allocated there.
        code = (f'import sys; sys.path.insert(0, {str(ROOT / "scripts")!r}); '
                f'from run_benchmark import check_concurrency; '
                f'check_concurrency({cfg["concurrency"]}, 4, 8192, "osworld", {str(ROOT / cfg["acceptance_report"])!r})')
        remote(host, [PYTHON, '-c', code])
        statefile = ROOT / 'runs/osworld-worker-worker0-service.json'
        live = False
        if statefile.exists():
            state = json.loads(statefile.read_text())
            code = (f'from pathlib import Path; import json; p=Path("/proc/{state["pid"]}/cmdline"); '
                    f'print(json.dumps(p.exists() and {state["identity"]!r}.encode() in p.read_bytes()))')
            live = json.loads(remote(host, [PYTHON, '-c', code]).stdout)
            if live and state['identity'] != cfg['config']:
                raise RuntimeError('A different secondary worker configuration is running')
        if not live:
            remote(host, [PYTHON, str(ROOT / 'scripts/infra_service.py'), 'osworld-worker', 'start',
                          '--config', cfg['config'], '--task-name', 'osworld-rl', '--controller',
                          'http://research-a-worker-1.job-research-a.training.svc.cluster.local:18520/api'])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            with opener.open('http://127.0.0.1:18520/api/list_workers', timeout=5) as response:
                workers = json.load(response).get('osworld-rl', {}).get('workers', {})
            for worker in workers.values():
                if (worker['address'].startswith('http://' + host + '.') and worker['status'] == 'ALIVE'
                        and not worker['stale'] and worker['capacity'] == cfg['concurrency']):
                    result.update(enabled=True, additional_capacity=cfg['concurrency'], worker=worker,
                                  acceptance_report=cfg['acceptance_report'], rpc_smoke_report=cfg['rpc_smoke_report'])
                    return cfg['concurrency']
            time.sleep(5)
        raise RuntimeError('Secondary worker did not register its expected capacity')
    except Exception as exc:
        result['fallback_reason'] = str(exc)
        return 0
    finally:
        write_json(DIRECTORY / 'secondary-worker-selection.json', result)


def infrastructure_start():
    for service in ['controller', 'osworld-worker']:
        path = ROOT / 'runs' / (service + '-service.json')
        if path.exists():
            state = json.loads(path.read_text())
            proc = Path(f'/proc/{state["pid"]}/cmdline')
            if proc.exists() and state['identity'].encode() in proc.read_bytes():
                if service == 'osworld-worker' and 'osworld-original-worker.yaml' not in state['identity']:
                    raise RuntimeError('Another owned worker configuration is still running; inspect before replacing')
                continue
        command = [PYTHON, str(ROOT / 'scripts/infra_service.py'), service, 'start']
        if service == 'osworld-worker':
            command += ['--config', 'configs/osworld-original-worker.yaml', '--task-name', 'osworld-rl']
        call(command)
    deadline = time.monotonic() + 180
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open('http://127.0.0.1:18520/api/get_indices?name=osworld-rl', timeout=5) as response:
                actual = {str(x) for x in json.load(response) if x != -1}
            expected = set(json.loads((ROOT / 'configs/rl-indices.json').read_text()))
            if actual == expected:
                with opener.open('http://127.0.0.1:18520/api/list_workers', timeout=5) as response:
                    workers = json.load(response)['osworld-rl']['workers'].values()
                if any(w['address'] == 'http://127.0.0.1:18521/api' and w['status'] == 'ALIVE'
                       and not w['stale'] and w['capacity'] == 32 for w in workers):
                    return 32 + secondary_worker_start()
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError('Controller does not expose exactly the pinned 2049-task pool')


def frontier_initialization():
    stop_inference()
    vm_acceptance()
    capacity = infrastructure_start()
    pool, services = start_pool('osworld-sft', frontier=True)
    log = (DIRECTORY / 'frontier-collector.log').open('ab', buffering=0)
    proc = subprocess.Popen([PYTHON, str(ROOT / 'scripts/collect_frontier_rates.py'), '--service-pool', str(pool),
                             '--concurrency', str(capacity)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    state = {}
    try:
        while proc.poll() is None:
            maintain_services(services, state)
            write_json(DIRECTORY / 'frontier-services-watch.json', state)
            time.sleep(20)
        if proc.returncode:
            raise RuntimeError('Frontier collection incomplete; preserve missingness and repair before training')
    finally:
        stop_inference()


def original_grpo():
    stop_inference()
    capacity = infrastructure_start()
    cfg = json.loads(CONFIG.read_text())
    # Pin the selected budget/config before launching the trainer.
    source_config = ROOT / cfg['training_config']
    import yaml
    effective = yaml.safe_load(source_config.read_text())
    effective['train']['concurrency'] = capacity
    content = yaml.safe_dump(effective, sort_keys=False).encode()
    pinned = DIRECTORY / ('training-config-' + hashlib.sha256(content).hexdigest()[:12] + '.yaml')
    if not pinned.exists():
        pinned.write_bytes(content)
    cfg['training_config'] = str(pinned)
    write_json(DIRECTORY / 'training-provenance.json', {
        'created_at_utc': now(), 'config': str(pinned), 'config_sha256': hashlib.sha256(pinned.read_bytes()).hexdigest(),
        'upstream': json.loads((ROOT / 'configs/sources.lock.json').read_text())['upstream'],
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'initial_rates_sha256': hashlib.sha256((ROOT / 'configs/initial_pass_rate.osworld-rl.json').read_bytes()).hexdigest(),
        'adaptations': {**cfg['adaptations'], 'environment_concurrency': capacity,
                        'secondary_worker': json.loads((DIRECTORY / 'secondary-worker-selection.json').read_text())},
        'algorithm_changes_enabled': False})
    call([PYTHON, str(ROOT / 'scripts/preflight_rl.py'), '--config', cfg['training_config'], '--gpus', '0,1,2,3,4,5,6,7'])
    command = [PYTHON, str(ROOT / 'scripts/launch_local_rl.py'), '--config', cfg['training_config'],
               '--gpus', '0,1,2,3,4,5,6,7', '--run-name', 'original-grpo-worker1',
               '--save-final', '--report', 'reports/original-grpo-training.json']
    proc = subprocess.Popen(command, cwd=ROOT)
    started = time.time()
    progress = {}
    marker = ROOT / 'runs/checkpoints/thudm-scalecua-osworld/original-grpo-worker1/latest_checkpointed_iteration.txt'
    while proc.poll() is None:
        rows = []
        for path in (ROOT / 'runs/metrics/original-grpo-worker1').glob('*/original-grpo-worker1/train_metrics.jsonl'):
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # A writer may be midway through the final line.
        progress = {'updated_at_utc': now(), 'status': 'running', 'pid': proc.pid,
                    'elapsed_hours': (time.time() - started) / 3600, 'allocated_gpus': 8,
                    'last_checkpoint_step': marker.read_text().strip() if marker.exists() else None,
                    'last_metrics': rows[-1] if rows else None,
                    'finite_update_records': sum('actor/grad_norm' in row for row in rows),
                    'algorithm': 'Original GRPO with Frontier; no TA-GRPO or critic changes'}
        write_json(ROOT / 'reports/original-grpo-progress.json', progress)
        time.sleep(30)
    progress.update(updated_at_utc=now(), status='completed' if proc.returncode == 0 else 'failed',
                    exit_code=proc.returncode, elapsed_hours=(time.time() - started) / 3600)
    write_json(ROOT / 'reports/original-grpo-progress.json', progress)
    if proc.returncode:
        raise RuntimeError(f'Original GRPO trainer exited with code {proc.returncode}; checkpoint resume is enabled')


def register_trained_checkpoint():
    root = ROOT / 'runs/checkpoints/thudm-scalecua-osworld/original-grpo-worker1'
    marker = root / 'latest_checkpointed_iteration.txt'
    step = int(marker.read_text().strip())
    model = root / f'global_step_{step}/huggingface'
    if not (model / 'config.json').is_file() or not list(model.glob('*.safetensors')):
        raise RuntimeError('Latest checkpoint has no complete Hugging Face export')
    index = model / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else [p.name for p in model.glob('*.safetensors')]
    if not all((model / name).is_file() and (model / name).stat().st_size > 0 for name in shards):
        raise RuntimeError('Incomplete checkpoint shards')
    hashes = {}
    for name in shards:
        digest = hashlib.sha256()
        with (model / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {'revision': identity, 'training_step': step, 'model_dir': str(model), 'shards': shards,
                'shard_sha256': hashes, 'identity_scope': 'SHA256 of the ordered full shard hashes'}
    write_json(ROOT / 'reports/reproduced-rl-model-manifest.json', manifest)
    write_json(ROOT / 'runs/reproduced-rl-model.json', {'model_id': 'local/original-grpo-worker1', 'model_revision': identity,
               'model_dir': str(model), 'served_model_name': 'scalecua-qwen3-vl-8b-reproduced-rl', 'port': 18120,
               'state_file': 'reproduced-rl-inference-service.json', 'manifest_file': 'reproduced-rl-model-manifest.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['released_rl_reference', 'frontier_initialization', 'original_grpo', 'reproduced_checkpoint_evaluation'])
    args = parser.parse_args()
    if socket.gethostname() != 'research-a-worker-1':
        parser.error('Original RL pipeline is scoped to worker-1')
    DIRECTORY.mkdir(exist_ok=True)
    receipt = {'stage': args.stage, 'started_at_utc': now(), 'exit_code': 1}
    try:
        if args.stage == 'released_rl_reference':
            evaluate('osworld-rl-reference')
        elif args.stage == 'frontier_initialization':
            frontier_initialization()
        elif args.stage == 'original_grpo':
            original_grpo()
        else:
            register_trained_checkpoint()
            evaluate('reproduced-rl')
        receipt['exit_code'] = 0
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['finished_at_utc'] = now()
        write_json(DIRECTORY / (args.stage + '.receipt.json'), receipt)


if __name__ == '__main__':
    main()
