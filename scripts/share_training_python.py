"""Move the training interpreter into shared storage after training processes exit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'configs/paths.json').read_text())
assert socket.gethostname() in cfg['workers']
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        cmd = (proc / 'cmdline').read_bytes().split(b'\0')
        if any(arg.endswith(b'/launch_local_rl.py') for arg in cmd):
            raise SystemExit('Wait for the local RL launcher to exit before relinking Python')
    except (FileNotFoundError, PermissionError):
        pass
envdir = Path(cfg['rl_env'])
source = Path(subprocess.check_output([str(envdir / 'bin/python'), '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()).resolve()
destination = Path(cfg['cache_dir']) / 'python/cpython-3.12.16-linux-x86_64-gnu'
if not destination.exists():
    shutil.copytree(source, destination, symlinks=True)
assert hashlib.sha256((source / 'bin/python3.12').read_bytes()).digest() == hashlib.sha256((destination / 'bin/python3.12').read_bytes()).digest()
link = envdir / 'bin/python.shared-tmp'
link.symlink_to(destination / 'bin/python3.12')
os.replace(link, envdir / 'bin/python')
configfile = envdir / 'pyvenv.cfg'
lines = configfile.read_text().splitlines()
configfile.write_text('\n'.join('home = ' + str(destination / 'bin') if line.startswith('home = ') else line for line in lines) + '\n')
assert subprocess.check_output([str(envdir / 'bin/python'), '-c', 'import sys; print(sys.prefix)'], text=True).strip() == str(envdir)
report = {'source': str(source), 'destination': str(destination), 'version': '3.12.16', 'env_relinked': True}
(ROOT / 'reports/shared-python.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
