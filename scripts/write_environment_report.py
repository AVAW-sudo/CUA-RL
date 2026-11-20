"""Generate an auditable HTML/JSON report from isolated environment experiments."""
import html
import json
from pathlib import Path
import statistics

from qemu_runtime import ROOT
from run_benchmark import now, write_json


def read(name):
    path = ROOT / 'reports' / name
    return json.loads(path.read_text()) if path.is_file() else {}


def median(values):
    return statistics.median(values) if values else None


def number(value):
    return '—' if value is None else f'{value:.2f}'


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def main():
    runtime = read('environment-runtime-comparison.json')
    container = read('userspace-gnome-comparison-v3.json')
    office = read('userspace-office-smoke.json')
    profile = read('evaluation-phase-profile.json')
    applications = read('userspace-applications-v2.json')
    concurrent = [read(f'snapshot-concurrency-worker-{i}-v2.json') for i in [0,1]]
    watchdog = read('watchdog-recovery-validation.json')
    parity = [read('snapshot-task-reset-osworld-v2.json'), read('snapshot-task-reset-scienceboard-v2.json')]
    report = {'created_at_utc': now(), 'production_runtime_changed': False, 'additional_gpus': 0,
              'runtime_status': runtime.get('status'), 'qemu': [], 'task_reset': [],
              'container_status': container.get('status'), 'office_status': office.get('status'),
              'concurrent_snapshot_acceptance': concurrent, 'applications': applications, 'watchdog': watchdog,
              'limitations': [
                  'Infrastructure microbenchmarks and two selected scripted tasks are not full benchmark equivalence or end-to-end RL throughput.',
                  'QEMU uses byte-identical base images, 4 vCPU and 8 GiB; host caches were not cleared. Other CPU workloads were not paused.',
                  'PRoot desktop has different application/system configuration and weak isolation; it is restricted to trusted scripted probes.',
                  'Native desktop startup excludes image download/extraction and does not include original benchmark setup.',
                  'Cold task setup and snapshot task setup are separately recorded; a fast snapshot does not remove task setup or inference.',
                  'All failed attempts are retained. No experimental score is added to SFT benchmark reports.'
              ],
              'sources': ['environment-runtime-comparison.json', 'environment-runtime-comparison.provenance.json',
                  'snapshot-task-reset-osworld.json', 'snapshot-task-reset-osworld-v2.json',
                  'snapshot-task-reset-scienceboard.json', 'snapshot-task-reset-scienceboard-v2.json', 'userspace-gnome-comparison.json',
                  'userspace-gnome-comparison-v2.json', 'userspace-gnome-comparison-v3.json', 'userspace-office-smoke.json',
                  'evaluation-phase-profile.json', 'environment-acceleration-survey.json',
                  'userspace-applications-v1.json', 'userspace-applications-v2.json',
                  'snapshot-concurrency-worker-0-v1.json', 'snapshot-concurrency-worker-1-v1.json',
                  'snapshot-concurrency-worker-0-v2.json', 'snapshot-concurrency-worker-1-v2.json',
                  'snapshot-task-reset-production-osworld.json', 'watchdog-recovery-validation.json']}
    rows = []
    for image in ['osworld', 'scienceboard']:
        for arm in ['shared_disk', 'local_disk']:
            attempts = [r for r in runtime.get('runs', []) if r['image'] == image and r['arm'] == arm]
            finished = [r for r in attempts if r.get('finished_at_utc')]
            valid = [r for r in finished if r['status'] == 'passed' and r.get('cleanup_passed')]
            resets = [s for r in valid for s in r.get('snapshot_resets', []) if s['isolation_passed']]
            result = {'image': image, 'arm': arm, 'finished': len(finished), 'passed': len(valid),
                'failed': sum(r['status'] == 'failed' for r in finished),
                'cold_boot_median_seconds': median([r['cold_boot_seconds'] for r in valid]),
                'control_median_seconds': median([s for r in valid for s in r['guest_control']['screenshot_and_control_seconds']]),
                'snapshot_verified_resets': len(resets),
                'snapshot_restore_median_seconds': median([s['restore_and_ready_seconds'] for s in resets])}
            report['qemu'].append(result)
            rows.append([image, arm, f'{len(valid)}/{len(finished)}', number(result['cold_boot_median_seconds']),
                         number(result['control_median_seconds']), len(resets), number(result['snapshot_restore_median_seconds'])])
    native = [r['result'] for r in container.get('runs', []) if r.get('status') == 'passed' and r['result'].get('cleanup_passed')]
    report['native'] = {'successful_rounds': len(native),
        'desktop_ready_median_seconds': median([r['desktop_ready_seconds'] for r in native]),
        'http_screenshot_control_median_seconds': median([s for r in native for s in r['http_screenshot_fresh_process_control_seconds']]),
        'http_samples': sum(len(r['http_screenshot_fresh_process_control_seconds']) for r in native),
        'image_digest': container.get('image_digest'),
        'office_save_verified': any(r.get('result', {}).get('office_text_saved') for r in office.get('runs', []))}
    for source in parity:
        for task in source.get('tasks', []):
            report['task_reset'].append({k:task.get(k) for k in ['benchmark', 'task_path', 'task_sha256', 'status',
                'cold_environment_seconds', 'iterations', 'error', 'cleanup_passed']})
    report['production_backend_selection'] = read('pilot-backend-selection.json')
    report['production_runtime_changed'] = report['production_backend_selection'].get('enabled',False)
    write_json(ROOT / 'reports/environment-acceleration-report.json', report)
    body = '<h1>CUA-RL 环境加速实验</h1><p>更新时间：' + report['created_at_utc'] + '</p>'
    body += '<p>本报告记录 CPU 侧工程实验。原版 SFT/GRPO 流程、8B 权重及正式评分口径保持不变，实验额外占用 0 张 GPU。</p>'
    body += '<h2>原镜像：磁盘位置与完整快照</h2><p>QEMU 实验状态：' + html.escape(str(report['runtime_status'])) + '。时间单位为秒；成功/完成不包含仍在运行的轮次。</p>'
    body += table(['镜像', '磁盘', '成功/完成', '冷启动中位', '截图+控制中位', '已验证恢复数', '恢复就绪中位'], rows)
    body += '<h2>原任务 setup 与 verifier 的恢复检查</h2>'
    body += table(['镜像', '状态', '冷启动环境', '轮次', '初始分→脚本动作后分', '快照恢复+setup'], [
        [t['benchmark'], t['status'], number(t['cold_environment_seconds']), len(t.get('iterations') or []),
         ', '.join(f"{x.get('initial_score','?')}→{x.get('scripted_score','?')}" for x in t.get('iterations') or []),
         ', '.join(number(x.get('reset_and_setup_seconds')) for x in t.get('iterations') or [] if x['arm'] == 'snapshot')]
        for t in report['task_reset']])
    body += '<p>这是固定脚本的工程检查，不是模型分数。冷启动解法失败时不能据此判断快照等价；早期失败记录保留。</p>'
    native = report['native']
    body += '<h2>用户态 PRoot + Xvnc + GNOME</h2>' + table(['成功轮次', '桌面+测试窗口就绪中位', 'HTTP 截图+新进程控制中位', 'HTTP 样本数', 'Writer 输入并保存验证'], [[
        native['successful_rounds'], number(native['desktop_ready_median_seconds']), number(native['http_screenshot_control_median_seconds']),
        native['http_samples'], native['office_save_verified']]])
    body += '<p>镜像固定 digest：<code>' + html.escape(str(native['image_digest'])) + '</code>。测试包含真实键鼠与截图；使用与 VM 不同的桌面配置和测试服务，不能将速度比直接称为原 benchmark 加速比。PRoot 也不提供足够的隔离来直接托管模型生成的任意动作。</p>'
    body += '<h2>并发快照与恢复验证</h2>' + table(['节点','VM 数','状态','已完成恢复轮数','恢复中位秒'], [[c.get('hostname'),c.get('count'),c.get('status'),len(c.get('rounds',[])),number(c.get('reset_median_seconds'))] for c in concurrent])
    body += '<p>第一版并发测试发现 guest 时钟随快照回退；第二版在 setup 前校准宿主机 epoch，并继续检查磁盘、鼠标、进程和时钟。所有失败报告保留。</p>'
    body += '<h2>用户态真实应用重复检查</h2>' + table(['轮次','状态','Writer 保存','浏览器键盘输入','Calc 数值保存','清理'], [[r.get('iteration'),r.get('status'),r.get('result',{}).get('office_text_saved'),r.get('result',{}).get('browser_keyboard_verified'),r.get('result',{}).get('calc_values_saved'),r.get('result',{}).get('cleanup_passed')] for r in applications.get('runs',[])])
    body += '<p>看门狗故障注入：'+html.escape(str(watchdog.get('status')))+'；杀掉监督进程后，底层命令启动次数：'+str(watchdog.get('command_start_count'))+'。</p>'
    body += '<h2>现有 SFT 日志的耗时拆解</h2>' + table(['队列', '已解析任务', '环境动作中位', '动作间隙中位', '启动+setup中位', 'setup到首动作中位'], [
        [c['campaign'], c['parsed_logs'], number(c['environment_action']['median_seconds']), number(c['between_action_gap']['median_seconds']),
         number(c['boot_and_setup']['median_seconds']), number(c['setup_to_first_action']['median_seconds'])] for c in profile.get('campaigns', [])])
    body += '<p>动作间隙包含推理及客户端处理；环境动作包含等待、截图和 HTTP。以上为历史有效任务统计，不是 GPU 单独耗时，也不是完整训练吞吐。</p>'
    body += '<h2>当前判断</h2><p>近期优先验证同镜像 warm pool + 完整快照，以减少每条轨迹的重复开机。随后需要多应用、并发和原模型 rollout 对照，才能启用到正式训练。用户态容器已展示原生执行的速度潜力，继续作为独立环境研究；保留原任务全集和原 verifier 的对照，不能用筛题后的分数替代原版指标。</p>'
    body += '<h2>边界与原始证据</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in report['limitations']) + '</ul><ul>'
    body += ''.join('<li><a href="' + html.escape(x) + '">' + html.escape(x) + '</a></li>' for x in report['sources']) + '</ul>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CUA-RL 环境加速实验</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font-family:system-ui,sans-serif;line-height:1.65;color:#202b36}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd5dc;padding:9px;text-align:left}th{background:#edf3f7}code{overflow-wrap:anywhere}h2{margin-top:32px}</style>' + body + '</html>'
    (ROOT / 'reports/environment-acceleration-report.html').write_text(page)
    print(json.dumps({'report': str(ROOT / 'reports/environment-acceleration-report.html'), 'qemu_status': report['runtime_status'], 'native': native}))


if __name__ == '__main__':
    main()
