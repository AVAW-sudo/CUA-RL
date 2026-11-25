"""Run SFT evaluation in the foreground of an AIS entry, one replica per rank."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from ais_desktop_service import stop
from ais_vm_check import ROOT, save
from ais_worker_cache import stage_runtime, stage_worker
from prepare_sft_eval import prepare


def official_tasks(source, benchmark):
    name = 'test_scienceboard' if benchmark == 'scienceboard' else 'test_nogdrive'
    domains = json.loads((source / 'evaluation_examples' / (name + '.json')).read_text())
    return [f'{domain}/{identifier}' for domain, ids in domains.items() for identifier in ids]


def cgroup_cpu_seconds():
    try:
        fields = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        return int(fields['usage_usec']) / 1e6
    except (OSError, KeyError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, choices=['osworld', 'scienceboard'])
    parser.add_argument('--checkpoint', choices=['osworld-sft', 'scienceboard-sft'])
    parser.add_argument('--scienceboard-setup', choices=['released', 'restored'], default='released')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--tasks', nargs='+', metavar='DOMAIN/ID')
    selection.add_argument('--full', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--proxy-policy', choices=['upstream', 'direct'], default='direct')
    args = parser.parse_args()
    rank, count = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    if not 0 <= rank < count or not 1 <= args.workers <= 8:
        parser.error('Invalid rank, world size or worker count')
    if args.benchmark != 'scienceboard' and args.scienceboard_setup == 'restored':
        parser.error('Restored initialization applies only to ScienceBoard')
    source = prepare()
    official = official_tasks(source, args.benchmark)
    selected = args.tasks or official
    if len(set(selected)) != len(selected) or set(selected) - set(official):
        parser.error('Each selected task must occur exactly once in the official list')
    tasks = selected[rank::count]
    if not tasks:
        print(f'Rank {rank} has no assigned tasks', flush=True)
        return
    job = os.getenv('JOB_ID', 'notebook')
    if not job.replace('-', '').isalnum():
        raise ValueError('Invalid AIS job identifier')
    parent = ROOT / 'runs/ais-sft-evaluations' / job
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'rank-{rank}-', dir=parent))
    print(f'AIS SFT evaluation output: {out}', flush=True)
    save(out / 'selection.json', {'rank': rank, 'world_size': count, 'all_selected': selected, 'assigned': tasks})
    checkpoint = args.checkpoint or args.benchmark + '-sft'
    rc = run_profiles(source, args.benchmark, checkpoint, tasks, out,
                      [(args.scienceboard_setup, 'evaluation')], args.workers, args.proxy_policy)
    if rc:
        raise SystemExit(rc)


def task_result(target, task, run, cancelled):
    if cancelled.is_set():
        raise RuntimeError('Rank cancelled')
    domain, identifier = task.split('/')
    folder = target / 'tasks' / domain / identifier
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'outcome.json'
    evaluation = folder / 'evaluation'
    if receipt.exists():
        row = json.loads(receipt.read_text())
    else:
        summary = evaluation / 'summary.json'
        reason = 'Interrupted attempt without a final summary'
        if not evaluation.exists():
            print(f'Starting {target.name} {task}', flush=True)
            with (folder / 'evaluation.log').open('ab') as log:
                code = run(evaluation, [task], log)
            reason = f'Evaluator exit {code} without a final summary'
            if not evaluation.exists():
                raise RuntimeError(f'Evaluator failed before starting {task}. See {folder}/evaluation.log')
        if summary.is_file():
            row, = json.loads(summary.read_text())['tasks']
        else:
            row = {'domain': domain, 'id': identifier, 'score': None, 'status': 'infrastructure_error',
                   'errors': [reason], 'directory': str(evaluation)}
    if (row['domain'], row['id']) != (domain, identifier):
        raise ValueError(f'Unexpected task in {folder}')
    if not receipt.exists():
        save(receipt, row)
    return row


def checkpointed_profile(target, tasks, run, desktops=1, cancelled=None):
    """Workers own individual receipts. Only the coordinator publishes the ordered progress file."""
    if not 1 <= desktops <= 8 or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('Expected 1..8 desktops and distinct nonempty tasks')
    target.mkdir(parents=True, exist_ok=True)
    cancelled = cancelled or threading.Event()
    results = {task: task_result(target, task, run, cancelled) for task in tasks
               if (target / 'tasks' / task / 'outcome.json').is_file()}
    def publish():
        rows = [results[task] for task in tasks if task in results]
        scored = sum(r['status'] == 'completed' for r in rows)
        save(target / 'summary.json', {'expected': len(tasks), 'accounted': len(rows), 'scored': scored, 'tasks': rows})
        print(f'{target.name}: {len(rows)}/{len(tasks)} accounted, {scored} scored', flush=True)
        return scored
    scored = publish() if results else 0
    queue = iter(task for task in tasks if task not in results)
    with ThreadPoolExecutor(max_workers=desktops) as pool:
        pending = set()
        try:
            while True:
                while len(pending) < desktops:
                    task = next(queue, None)
                    if task is None:
                        break
                    pending.add(pool.submit(task_result, target, task, run, cancelled))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    results[row['domain'] + '/' + row['id']] = row
                scored = publish()
        except BaseException:
            cancelled.set()
            for future in pending:
                future.cancel()
            raise
    return 0 if scored == len(tasks) else 2


def run_profiles(source, benchmark, checkpoint, tasks, out, profiles, workers=1, proxy_policy='direct', per_task=False,
                 experiments=None, service_state=None):
    """Run independent desktops for each profile against one pinned local model service."""
    base = ROOT.parent.parent
    runtime, _ = stage_worker(Path(os.getenv('CUA_WORKER_MANIFEST', base / 'ENV/cache/ais-osworld-worker/manifest.json')))
    worker = sys.executable if sys.version_info[:2] == (3, 10) else str(base / 'ENV/conda_envs/test-gpu/bin/python')
    site = runtime / 'site'
    cpu_env = dict(os.environ, PYTHONPATH=str(site), LD_LIBRARY_PATH=os.pathsep.join(
        [str(site / 'torch/lib'), os.environ.get('LD_LIBRARY_PATH', '')]))
    if benchmark == 'osworld':
        fixtures = [json.loads((source / 'evaluation_examples/examples' / (task + '.json')).read_text()) for task in tasks]
        if any(step['type'] == 'download' for fixture in fixtures for step in fixture.get('config', [])):
            inputs, evidence = stage_runtime(base / 'ENV/cache/ais-sft-osworld-inputs/manifest.json', 'fixtures')
            cpu_env['CUA_SFT_SETUP_INPUTS'] = str(inputs / 'tasks')
            save(out / 'setup-inputs.json', evidence)
        if any('compare_image_text' in fixture['evaluator']['func'] for fixture in fixtures):
            ocr_manifest = base / 'ENV/cache/ais-sft-ocr/manifest.json'
            ocr, evidence = stage_runtime(ocr_manifest, 'sft-ocr')
            cpu_env.update(PYTHONPATH=os.pathsep.join([str(ocr / 'site'), str(site)]), EASYOCR_MODULE_PATH=str(ocr / 'models'))
            save(out / 'ocr-runtime.json', evidence)
    service_dir = out
    timeout = None
    if per_task:
        from sft_vm_image import verified_image
        vm = json.loads((ROOT / 'configs/sources.lock.json').read_text())[
            'smoke_vm' if benchmark == 'osworld' else 'scienceboard_vm']
        image, _ = verified_image(benchmark, ROOT / vm['extracted_file'], vm['bytes'])
        service_dir = Path(tempfile.mkdtemp(prefix='service-', dir=out))
        timeout = int(os.getenv('CUA_SFT_TASK_TIMEOUT_SECONDS', '5400'))
        if timeout <= 0:
            raise ValueError('CUA_SFT_TASK_TIMEOUT_SECONDS must be positive')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with ExitStack() as stack:
        policy = None
        if service_state is not None:
            state = service_state.resolve(strict=True)
            record = json.loads(state.read_text())
            if record['status'] != 'ready' or record['hostname'] != socket.gethostname():
                raise ValueError('Expected a ready local SFT inference service')
        else:
            log = stack.enter_context((service_dir / 'policy.log').open('wb'))
            policy = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/ais_frontier_server.py'),
                                       '--eval-checkpoint', checkpoint, '--state-dir', str(service_dir / 'policy'), '--port', str(port)],
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stack.callback(stop, policy, 60)
            state = service_dir / 'policy/service.json'
            deadline = time.monotonic() + int(os.getenv('CUA_POLICY_STARTUP_SECONDS', '3600')) + 120
            while policy.poll() is None and time.monotonic() < deadline:
                if state.exists() and json.loads(state.read_text())['status'] == 'ready':
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f'SFT service startup failed; inspect {service_dir / "policy.log"}')

        cancelled = threading.Event()
        def evaluate(profile, target, selected, output_log=None):
            cmd = [os.getenv('CUA_WORKER_PYTHON', worker), '-S', '-u', str(ROOT / 'scripts/evaluate_sft.py'),
                   '--ais', '--benchmark', benchmark, '--checkpoint', checkpoint, '--scienceboard-setup', profile,
                   '--run-dir', str(target), '--service-state', str(state), '--recording-mode', 'steps-only',
                   '--proxy-policy', proxy_policy, '--cpus', '4', '--memory-mb', '8192', '--workers', str(workers), '--tasks', *selected]
            with ExitStack() as episode:
                evaluation = subprocess.Popen(cmd, env=cpu_env, stdout=output_log,
                                              stderr=subprocess.STDOUT if output_log else None, start_new_session=True)
                episode.callback(stop, evaluation, 60)
                started = time.monotonic()
                while evaluation.poll() is None:
                    if cancelled.is_set() or (policy is not None and policy.poll() is not None):
                        raise RuntimeError(f'SFT service stopped during evaluation; inspect {service_dir / "policy.log"}')
                    if timeout and time.monotonic() - started >= timeout:
                        print(f'Evaluation timed out after {timeout}s: {target}', flush=True)
                        return 124
                    time.sleep(1)
            if cancelled.is_set() or (policy is not None and policy.poll() is not None):
                raise RuntimeError(f'SFT service exited during evaluation; inspect {service_dir / "policy.log"}')
            return evaluation.returncode

        result_code = 0
        desktops = int(os.getenv('CUA_SFT_DESKTOPS', '1'))
        boot = os.getenv('CUA_SFT_BOOT_MODE', 'cold')
        variants = experiments or [(desktops, boot)]
        template = None
        cpu_env.pop('CUA_SFT_TEMPLATE', None)
        for concurrency, mode in variants:
            if mode not in ('cold', 'snapshot', 'cold-gui', 'snapshot-gui') or not 1 <= concurrency <= 8:
                raise ValueError('Unknown SFT performance variant or desktop count outside 1..8')
            cpu_env['CUA_SFT_RESIDENT_GUI'] = '1' if mode.endswith('-gui') else '0'
            if mode.startswith('snapshot'):
                if not per_task:
                    raise ValueError('Snapshot boot requires per-task evaluation')
                if template is None:
                    template = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='sft-template-')))
                    cmd = [os.getenv('CUA_WORKER_PYTHON', worker), '-S', '-u', str(ROOT / 'scripts/ais_vm_check.py'),
                           '--sft-template', str(template), '--image', str(image), '--output-root', str(service_dir)]
                    if experiments and any(boot.endswith('-gui') for _, boot in experiments):
                        cmd.append('--test-gui')
                    with (service_dir / 'template.log').open('wb') as log:
                        with ExitStack() as building:
                            builder = subprocess.Popen(cmd, env=cpu_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            building.callback(stop, builder, 30)
                            if builder.wait(timeout=1500):
                                raise RuntimeError(f'SFT template creation failed. See {service_dir / "template.log"}')
                cpu_env['CUA_SFT_TEMPLATE'] = str(template)
            else:
                cpu_env.pop('CUA_SFT_TEMPLATE', None)
            for profile, directory in profiles:
                target = out / directory if experiments is None else out / f'{mode}-{concurrency}' / directory
                print(f'SFT {benchmark} {profile}, {concurrency} desktops, {mode}: {target}', flush=True)
                cpu_before = cgroup_cpu_seconds() if experiments is not None else None
                started = time.monotonic()
                if per_task:
                    code = checkpointed_profile(target, tasks, partial(evaluate, profile), concurrency, cancelled)
                else:
                    code = evaluate(profile, target, tasks)
                if code == 2 and (target / 'summary.json').is_file():
                    result_code = 2
                elif code:
                    raise RuntimeError(f'Evaluation exited with status {code}: {target}')
                if experiments is not None:
                    elapsed = time.monotonic() - started
                    cpu_after = cgroup_cpu_seconds()
                    mean_cpus = (cpu_after - cpu_before) / elapsed if None not in (cpu_before, cpu_after) else None
                    data = json.loads((target / 'summary.json').read_text())
                    steps = sum(r.get('action_records', 0) for r in data['tasks'])
                    save(target / 'timing.json', {'elapsed_seconds': elapsed, 'desktops': concurrency,
                         'boot': mode, 'actions': steps, 'scored': data['scored'], 'attempted': len(tasks),
                         'mean_cgroup_cpus': mean_cpus,
                         'scored_per_hour': 3600 * data['scored'] / elapsed, 'actions_per_hour': 3600 * steps / elapsed})
        return result_code


if __name__ == '__main__':
    main()
