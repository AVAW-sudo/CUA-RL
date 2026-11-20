#!/usr/bin/env python3
"""Start/stop project-owned controller or local QEMU OSWorld worker processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=["controller", "osworld-worker"])
    parser.add_argument("action", choices=["start", "stop"])
    args = parser.parse_args()
    if socket.gethostname() != "research-a-worker-1":
        parser.error("Local bring-up is scoped to worker-1")
    statefile = ROOT / "runs" / (args.service + "-service.json")
    if statefile.exists():
        state = json.loads(statefile.read_text())
        proc = Path(f"/proc/{state['pid']}/cmdline")
        alive = proc.exists() and state["identity"] in proc.read_bytes().decode(errors="replace")
        if args.action == "stop":
            if state["hostname"] != socket.gethostname():
                parser.error("Service state belongs to a different worker")
            if alive:
                os.killpg(state["pid"], signal.SIGTERM)
            if args.service == 'osworld-worker':
                # Also clean up after a worker crash: QEMU is detached and
                # may outlive its recorded owner. Check each exact VM identity.
                for vmfile in (ROOT / 'runs/vms').glob('*/state.json'):
                    vm = json.loads(vmfile.read_text())
                    if vm.get('owner_pid') != state['pid'] or vm.get('hostname') != socket.gethostname():
                        continue
                    vmproc = Path(f"/proc/{vm['pid']}/cmdline")
                    if vmproc.exists() and ('cua-rl-' + vm['id']) in vmproc.read_bytes().decode(errors='replace'):
                        os.kill(vm['pid'], signal.SIGTERM)
            # Interrupt VM boot waits before waiting for worker shutdown.
            if alive:
                for _ in range(100):
                    if not proc.exists() or state["identity"] not in proc.read_bytes().decode(errors="replace"):
                        break
                    time.sleep(0.1)
            print(json.dumps({"stop_requested": alive, "pid": state["pid"]}))
            return
        if alive:
            parser.error("This project's service is already running")
    elif args.action == "stop":
        parser.error("No owned service state")
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    cwd = ROOT
    if args.service == "controller":
        port = 18520
        identity = str(ROOT / "runs/bin/agentrl-controller")
        cmd = [identity, "controller", "--port", str(port), "--host", "0.0.0.0", "--dashboard=false", "--long-timeout"]
    else:
        port = 18521
        identity = "configs/osworld-local-worker.yaml"
        env["PYTHONPATH"] = ":".join(str(ROOT / p) for p in ["scripts", "SCALE-CUA/osworld_eval", "SCALE-CUA/osworld_env/worker"])
        env["OSWORLD_HELPER_ASSETS"] = str(ROOT / "SCALE-CUA/osworld_env/worker/extra")
        env["NO_PROXY"] = "localhost,127.0.0.0/8,127.0.0.1,.training.svc.cluster.local"
        env["no_proxy"] = env["NO_PROXY"]
        cwd = ROOT / "SCALE-CUA/osworld_eval"
        cmd = ["/aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python", "-m", "agentrl.worker",
               "-c", str(ROOT / identity), "--controller", "http://127.0.0.1:18520/api", "--self", "http://127.0.0.1:18521/api",
               "--host", "127.0.0.1", "--port", str(port), "osworld-local-smoke"]
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    logfile = ROOT / "runs" / (args.service + ".log")
    with logfile.open("ab", buffering=0) as log:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    state = {"pid": process.pid, "hostname": socket.gethostname(), "port": port, "identity": identity, "command": cmd, "log": str(logfile)}
    statefile.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state))


if __name__ == "__main__":
    main()
