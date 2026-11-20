#!/usr/bin/env python3
"""Read-only GPU, resource and environment capability inventory; no environment secrets."""
import csv
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stderr": str(exc)}


def read(path):
    p = Path(path)
    return p.read_text().strip() if p.exists() else None


def main():
    fields = ["index", "name", "uuid", "driver_version", "memory.total", "memory.used", "utilization.gpu"]
    gpu = command(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "uid": os.getuid(),
        "gpus": [dict(zip(fields, [s.strip() for s in row])) for row in csv.reader(io.StringIO(gpu.get("stdout", "")))],
        "gpu_query": gpu,
        "gpu_processes": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
        "cpu_visible": os.cpu_count(),
        "cpu_quota": read("/sys/fs/cgroup/cpu.max"),
        "memory_limit_bytes": read("/sys/fs/cgroup/memory.max"),
        "memory": command(["free", "-b"]),
        "filesystems": command(["df", "-B1", "/aigc", "/tmp", "/dev/shm"]),
        "tools": {n: shutil.which(n) for n in ["docker", "podman", "unshare", "qemu-system-x86_64"]},
        "devices": {p: {"exists": Path(p).exists(), "readable": os.access(p, os.R_OK), "writable": os.access(p, os.W_OK)} for p in ["/dev/kvm", "/dev/fuse", "/dev/net/tun", "/var/run/docker.sock"]},
        "user_namespace": command(["unshare", "-Ur", "true"]),
        "user_mount_namespace": command(["unshare", "-Urm", "true"]),
        "rdma": command(["ls", "/dev/infiniband"]),
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
