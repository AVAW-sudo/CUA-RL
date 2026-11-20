"""Detach a lightweight guard that restarts the watchdog after process exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from persistent_job import alive, identity, write
from run_benchmark import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/original-rl-pilot-pipeline.json')
    p.add_argument('--guard', action='store_true')
    args = p.parse_args()
    if socket.gethostname() != 'research-a-worker-1': p.error('Run on worker-1')
    cfg = json.loads(args.config.read_text())
    directory = ROOT / cfg.get('state_directory', 'runs/original-rl-pipeline')
    directory.mkdir(parents=True, exist_ok=True)
    statefile = directory / 'guard.json'
    if not args.guard:
        if statefile.exists() and alive(json.loads(statefile.read_text()).get('identity')):
            print(json.dumps({'already_running':True, **json.loads(statefile.read_text())}))
            return
        with (directory / 'guard.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--guard', '--config', str(args.config)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'launched':identity(proc.pid),'state':str(statefile)}))
        return
    lease = (directory / 'guard.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'identity':identity(os.getpid()),'config':str(args.config),'watchdog_launches':0}
    write(statefile,state)
    while True:
        with (directory / 'watchdog.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen([sys.executable,str(ROOT/'scripts/pipeline_watchdog.py'),'--config',str(args.config)],
                                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(watchdog=identity(proc.pid),watchdog_launches=state['watchdog_launches']+1,last_started_unix=time.time())
        write(statefile,state)
        state.update(last_exit_code=proc.wait(),last_finished_unix=time.time())
        write(statefile,state)
        time.sleep(5)


if __name__=='__main__':main()
