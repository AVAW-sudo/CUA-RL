"""Run one real public task with the upstream server agent and local VM worker."""
import os
import datetime
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if socket.gethostname() != 'research-a-worker-1':
    raise SystemExit('Run on worker-1')
output = ROOT / 'runs' / ('osworld-sft-smoke-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
output.mkdir(exist_ok=True)
cmd = ['/aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python',
       str(ROOT / 'SCALE-CUA/scalecua_rl/scripts/server_agent.py'),
       '-m', 'scalecua-qwen3-vl-8b-sft', '-u', 'http://127.0.0.1:18080/v1',
       '--model-api-key', 'EMPTY', '--provider', 'openai', '-n', '1', '-j', '1',
       '--max-steps', '15', '--max-tokens', '4096', '-t', '0.8', '--top-p', '1',
       '-c', 'http://127.0.0.1:18520/api', '--task_name', 'osworld-local-smoke',
       '--agent_version', 'scalecua_osworld', '--env_version', 'anthropic', '--relative_coordinate',
       '--indices-file', str(ROOT / 'configs/osworld-smoke-indices.json'),
       '--output-dir', str(output), '--output-file', 'results.jsonl', '--store-messages',
       '--recording-dir', str(output / 'recordings'), '--log-dir-base', str(output / 'tasks')]
subprocess.run(cmd, env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4'), check=True, timeout=1800)
rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines() if line.strip()]
valid = len(rows) == 1 and isinstance(rows[0].get('result'), (int, float))
report = {'passed': valid, 'artifacts': str(output), 'scope': 'One public task with released eval agent adapter and QEMU TCG; not the full OSWorld benchmark or exact missing anthropic_rl agent',
          'reward': rows[0].get('result') if rows else None,
          'assistant_turns': sum(m.get('role') == 'assistant' for m in rows[0].get('messages', [])) if rows else 0}
(ROOT / 'reports/smoke-osworld.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
assert valid, 'Upstream CLI may exit zero on failed episodes; require a numeric verifier result'
