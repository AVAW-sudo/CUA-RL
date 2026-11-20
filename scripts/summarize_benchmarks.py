#!/usr/bin/env python3
"""Export current campaign metrics to JSON/CSV without dropping unscored tasks."""
import argparse
import csv
import datetime
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def export():
    campaigns = []
    excluded = []
    table = []
    for path in sorted((ROOT / 'runs/benchmarks').glob('*/summary.json')):
        summary = json.loads(path.read_text())
        if summary.get('excluded_from_baseline'):
            excluded.append({'campaign': path.parent.name, 'reason': summary['exclusion_reason'], 'source': str(path)})
            continue
        campaigns.append({'campaign': path.parent.name, 'source': str(path), **summary})
        for domain, data in [('ALL', summary['overall']), *summary['by_domain'].items()]:
            table.append({'campaign': path.parent.name, 'checkpoint': summary['checkpoint'],
                          'benchmark': summary['benchmark'], 'domain': domain,
                          'campaign_status': summary['status'], **{k: v for k, v in data.items() if k != 'note'}})
    document = {'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'metric_definition': {'success_rate': 'Number of official verifier scores equal to 1 / full task count',
                                      'mean_score': 'Sum of official verifier scores / full task count',
                                      'missing_policy': 'Both metrics remain null until all task results are valid; integration samples are excluded.'},
                'campaigns': campaigns, 'excluded_diagnostic_campaigns': excluded}
    target = ROOT / 'reports/sft-benchmark-results.json'
    temp = target.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, target)
    if table:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        target = target.with_suffix('.csv')
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_text(stream.getvalue())
        os.replace(temp, target)
    return campaigns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Refresh every 30 seconds until all active campaigns stop')
    args = parser.parse_args()
    while True:
        campaigns = export()
        if not args.watch or not any(c['status'] == 'running' for c in campaigns):
            for c in campaigns:
                print(json.dumps({'campaign': c['campaign'], 'status': c['status'], **c['overall']}))
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
