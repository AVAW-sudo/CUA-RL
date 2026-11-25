"""Compare full real GUI episodes on one local SFT replica and independent task directories."""
import argparse
import json
from pathlib import Path
import tempfile

from ais_sft_eval import official_tasks, run_profiles
from ais_vm_check import ROOT, save
from prepare_sft_eval import prepare


def variant(value):
    try:
        mode, count = value.split(':')
        count = int(count)
        if mode not in ('cold', 'snapshot', 'cold-gui', 'snapshot-gui') or not 1 <= count <= 8:
            raise ValueError
        return count, mode
    except ValueError:
        raise argparse.ArgumentTypeError('Expected cold:N, cold-gui:N, snapshot:N or snapshot-gui:N with N between 1 and 8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['scienceboard', 'osworld'], default='scienceboard')
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--variants', nargs='+', type=variant, default=[(1, 'cold'), (2, 'cold'), (2, 'snapshot')])
    parser.add_argument('--service-state', type=Path, help='Reuse an already running local benchmark service')
    args = parser.parse_args()
    source = prepare()
    if len(set(args.tasks)) != len(args.tasks) or set(args.tasks) - set(official_tasks(source, args.benchmark)):
        parser.error('Expected distinct official tasks')
    if len(set(args.variants)) != len(args.variants) or max(n for n, _ in args.variants) > len(args.tasks):
        parser.error('Use distinct variants and at least one task per concurrent desktop')
    root = ROOT / 'runs/ais-sft-performance'
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=args.benchmark + '-', dir=root))
    print(f'SFT performance experiment: {out}', flush=True)
    report = {'benchmark': args.benchmark, 'tasks': args.tasks, 'variants': args.variants, 'status': 'running'}
    save(out / 'report.json', report)
    try:
        rc = run_profiles(source, args.benchmark, args.benchmark + '-sft', args.tasks, out,
                          [('released', 'released')], per_task=True, experiments=args.variants,
                          service_state=args.service_state)
        report['status'] = 'finished_with_unscored' if rc else 'finished'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['arms'] = []
        for concurrency, mode in args.variants:
            target = out / f'{mode}-{concurrency}' / 'released'
            if (target / 'timing.json').is_file():
                arm = json.loads((target / 'timing.json').read_text())
                arm['tasks'] = [{k: row[k] for k in ('domain', 'id', 'status', 'score', 'action_records') if k in row}
                                for row in json.loads((target / 'summary.json').read_text())['tasks']]
                report['arms'].append(arm)
        report['templates'] = [json.loads(p.read_text()) for p in out.glob('service-*/template/template.json')]
        report['input_checks'] = [json.loads(p.read_text()) for p in out.glob('service-*/template/input-check.json')]
        save(out / 'report.json', report)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
