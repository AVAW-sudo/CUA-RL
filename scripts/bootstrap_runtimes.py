"""Recreate the dedicated training/desktop environments from the local version locks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'configs/paths.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runtime', choices=['agentrl', 'osworld'])
args = parser.parse_args()
if socket.gethostname() not in cfg['workers']:
    parser.error('Install on an authorized worker')
uv = cfg['uv']
envdir = Path(cfg['rl_env']) if args.runtime == 'agentrl' else Path(cfg['rl_env']).with_name('cua-rl-osworld')
python = str(envdir / 'bin/python')
if not (envdir / 'bin/python').exists():
    subprocess.run([uv, 'venv', '--python', '3.12.16' if args.runtime == 'agentrl' else '/usr/bin/python3', str(envdir)],
                   env=dict(os.environ, UV_PYTHON_INSTALL_DIR=cfg['cache_dir'] + '/python'), check=True)
if args.runtime == 'agentrl':
    for script, wheel in [('repair_decord_wheel.py', 'decord-0.6.0-py3-none-manylinux2010_x86_64.whl'),
                          ('repair_te_wheel.py', 'transformer_engine_cu13-2.12.0-py3-none-manylinux_2_28_x86_64.whl')]:
        if not (Path(cfg['cache_dir']) / 'wheels' / wheel).is_file():
            subprocess.run([sys.executable, str(ROOT / 'scripts' / script)], check=True)
    for item in json.loads((ROOT / 'reports/cxx-runtime.json').read_text()):
        target = Path(item['local'])
        if not target.is_file():
            source = Path(item['source'])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']
index = 'cu130' if args.runtime == 'agentrl' else 'cpu'
subprocess.run([uv, 'pip', 'sync', '--python', python, '--extra-index-url', 'https://download.pytorch.org/whl/' + index,
                '--index-strategy', 'unsafe-best-match', str(ROOT / f'configs/requirements-{args.runtime}.lock.txt')], check=True)
if args.runtime == 'agentrl':
    subprocess.run([uv, 'pip', 'install', '--python', python, '--no-deps',
                    '-e', str(ROOT / 'SCALE-CUA/scalecua_rl/trainer'), '-e', str(ROOT / 'SCALE-CUA/scalecua_rl/worker')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/patch_training_runtime.py')], check=True)
subprocess.run([python, '-m', 'pip', 'check'], check=True)
print(f'{args.runtime} installed; run the hardware smoke separately before training')
