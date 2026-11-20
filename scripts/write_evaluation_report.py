"""Create an auditable HTML/JSON evaluation report, including unresolved results."""
import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def summarize_campaign(path):
    state = json.loads((path / 'campaign.json').read_text())
    from run_benchmark import snapshot
    summary = snapshot(state)
    total, valid = summary['overall']['total'], summary['overall']['completed']
    missing = total - valid
    successes = summary['overall']['successes']
    scores = sum(row['score'] for row in state['tasks'].values() if row['status'] == 'completed')
    return {'campaign': path.name, 'source': str(path), **summary,
            'full_denominator_success_bounds': [successes / total, (successes + missing) / total],
            'full_denominator_mean_score_bounds': [scores / total, (scores + missing) / total],
            'bounds_note': 'Missing-score bounds, not a confidence interval; exact metrics stay null until complete.',
            'retries': {task: len(row['attempts']) - 1 for task, row in state['tasks'].items() if len(row['attempts']) > 1},
            'unscored': {task: row for task, row in state['tasks'].items() if row['status'] != 'completed'},
            'adjudications': state.get('adjudications', []),
            'launch_history': state.get('launch_history', []),
            'campaign_sha256': hashlib.sha256((path / 'campaign.json').read_bytes()).hexdigest()}


def generate(paths, output, final=False):
    campaigns = [summarize_campaign(Path(p)) for p in paths]
    if final and any(c['overall']['pending'] or c['overall']['running'] for c in campaigns):
        raise ValueError('Cannot finalize while tasks remain queued or running')
    report = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': ('final_complete' if all(c['overall']['completed'] == c['overall']['total'] for c in campaigns)
                         else 'final_with_unscored_cases') if final else 'provisional',
              'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_worktree_dirty': bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
              'sources': json.loads((ROOT / 'configs/sources.lock.json').read_text()),
              'campaigns': campaigns,
              'limitations': ['TCG software CPU emulation, direct network, and documented setup compatibility fixes.',
                              'Full-denominator exact metrics are null when a verifier result is missing; missing tasks are not dropped or silently assigned zero.',
                              'Published OSWorld RL tasks derive from benchmark seed tasks; these scores do not establish unseen-seed generalization.',
                              'SFT checkpoints and the released RL checkpoint are separate experimental arms.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    esc = html.escape
    rows, detail = [], []
    for c in campaigns:
        m = c['overall']; bounds = c['full_denominator_success_bounds']
        exact = f"{m['success_rate']:.2%}" if m['success_rate'] is not None else '待缺测项补判'
        rows.append(f"<tr><td>{esc(c['checkpoint'])}</td><td>{esc(c['benchmark'])}</td><td>{m['completed']}/{m['total']}</td><td>{m['successes']}</td><td>{exact}</td><td>{bounds[0]:.2%}–{bounds[1]:.2%}</td></tr>")
        domains = ''.join(f"<tr><td>{esc(k)}</td><td>{v['completed']}/{v['total']}</td><td>{v['successes']}</td><td>{v['infrastructure_or_verifier_errors']}</td></tr>" for k, v in c['by_domain'].items())
        errors = ''.join(f"<li><b>{esc(k)}</b>: {esc(str(v.get('error')))}</li>" for k, v in c['unscored'].items()) or '<li>无</li>'
        detail.append(f"<h2>{esc(c['campaign'])}</h2><p>revision: {esc(c['protocol']['model_revision'])}；单题一次采样，保留所有尝试。重试题数：{len(c['retries'])}。</p><table><tr><th>应用</th><th>有效/总数</th><th>完全成功</th><th>异常</th></tr>{domains}</table><h3>缺测和待审计项</h3><ul>{errors}</ul>")
    document = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ScaleCUA 测评报告</title>
<style>body{{font:16px/1.65 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f3f5f8}}li{{overflow-wrap:anywhere}}</style>
<h1>ScaleCUA 测评报告</h1><p>状态：{report['status']}；UTC：{report['created_at_utc']}；代码：{report['source_commit']}</p>
<p>完全成功定义为官方 verifier score=1。下界/上界表示未评分题目全失败/全成功两种极端，不是置信区间。精确平均分及全部运行协议见同名 JSON。</p>
<table><tr><th>Checkpoint</th><th>Benchmark</th><th>有效/总数</th><th>完全成功</th><th>精确成功率</th><th>全分母成功率上下界</th></tr>{''.join(rows)}</table>
{''.join(detail)}<h2>解释限制</h2><ul>{''.join('<li>'+esc(x)+'</li>' for x in report['limitations'])}</ul></html>'''
    output.with_suffix('.html').write_text(document)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaigns', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    report = generate(args.campaigns, args.output, args.final)
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
