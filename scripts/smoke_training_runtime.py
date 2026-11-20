#!/usr/bin/env python3
"""Guarded CUDA forward/backward checks for the dedicated training runtime."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from runtime_env import library_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    if socket.gethostname() != "research-a-worker-1":
        raise SystemExit("Scoped to worker-1 GPU 1")
    if "--child" not in sys.argv:
        for query in ["--query-gpu=memory.used", "--query-compute-apps=pid"]:
            value = subprocess.check_output(["nvidia-smi", "-i", "1", query, "--format=csv,noheader,nounits"], text=True).strip()
            if (query.startswith("--query-gpu") and int(value) > 1024) or (query.startswith("--query-compute") and value):
                raise SystemExit("GPU 1 is occupied")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", LD_LIBRARY_PATH=library_path(cfg), OMP_NUM_THREADS="4")
        subprocess.run([str(Path(cfg["rl_env"]) / "bin/python"), str(Path(__file__).resolve()), "--child"], env=env, check=True, timeout=180)
        return
    import torch
    import flash_attn
    import transformer_engine.pytorch as te
    import megatron.core
    import swift.megatron
    import agentrl.trainer.workers.registry
    from importlib.metadata import version
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer = te.Linear(128, 64, params_dtype=torch.bfloat16, device="cuda")
    layer(x).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    qkv = [torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    flash_attn.flash_attn_func(*qkv, causal=True).float().square().mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in qkv)
    torch.cuda.synchronize()
    report = {"passed": True, "gpu": "worker-1:1", "tests": ["Transformer Engine BF16 Linear forward/backward", "FlashAttention causal forward/backward", "AgentRL/Megatron/Swift imports"],
              "versions": {name: version(name) for name in ["torch", "vllm", "transformers", "flash-attn", "transformer-engine", "megatron-core", "ms-swift", "mcore-bridge"]},
              "scope": "CUDA kernel/import checks; not yet an 8B RL optimizer update"}
    (ROOT / "reports/smoke-training-runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
