#!/usr/bin/env python3
"""Short, guarded two-GPU NCCL correctness check on worker-1; not a throughput benchmark."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    if socket.gethostname() != "research-a-worker-1":
        raise SystemExit("This check is scoped to worker-1")
    if "LOCAL_RANK" not in os.environ:
        for gpu in [1, 2]:
            used = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            processes = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
            if int(used.strip()) > 1024 or processes.strip():
                raise SystemExit(f"GPU {gpu} is occupied; refusing to run")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,2", OMP_NUM_THREADS="2", NCCL_DEBUG="WARN")
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2", str(Path(__file__).resolve())], env=env, check=True, timeout=120)
        return
    import torch
    import torch.distributed as dist
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
    try:
        data = torch.full((1024 * 1024,), rank + 1.0, device=f"cuda:{rank}")
        start = time.monotonic()
        dist.all_reduce(data)
        torch.cuda.synchronize()
        assert torch.all(data == 3.0).item()
        dist.barrier(device_ids=[rank])
        if rank == 0:
            report = {"passed": True, "worker": socket.gethostname(), "physical_gpus": [1, 2], "torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(), "payload_bytes_per_rank": data.numel() * data.element_size(), "seconds_including_initialization": time.monotonic() - start, "scope": "intra-node correctness only; inter-node NCCL not tested while worker-0 is occupied"}
            (ROOT / "reports/smoke-nccl.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
