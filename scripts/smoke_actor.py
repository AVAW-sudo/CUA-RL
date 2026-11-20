#!/usr/bin/env python3
"""Load the actual 8B actor and perform one disposable multimodal optimizer step."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from runtime_env import library_path

ROOT = Path(__file__).resolve().parents[1]


def child():
    import torch
    import torch.distributed as dist
    import yaml
    from PIL import Image
    from transformers import AutoProcessor
    from agentrl.trainer.workers.megatron.megatron_worker import MegatronWorker
    from agentrl.trainer.algorithms.rope_index import get_rope_index_qwen3vl
    from agentrl.trainer.algorithms.loss_funcs import compute_log_probs_with_cp
    cfg = yaml.safe_load((ROOT / "configs/qwen3vl-osworld-smoke-6gpu.yaml").read_text())
    actor_cfg = cfg["actor"]
    actor_cfg["max_tokens_per_micro_batch"] = 1500
    worker = MegatronWorker(actor_cfg)
    worker.init_distributed("127.0.0.1", int(os.environ["MASTER_PORT"]))
    try:
        worker.build_model(cfg["model_path"], is_vision_model=True, model_type="qwen3_vl")
        worker.build_optimizer()
        print("Actor and optimizer loaded", flush=True)
        processor = AutoProcessor.from_pretrained(cfg["model_path"])
        smoke = json.loads((ROOT / "reports/smoke-thudm-sft.json").read_text())
        screenshot = Image.open(Path(smoke["artifacts"]) / "step-0.png")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Which button is visible?"}]},
                    {"role": "assistant", "content": "The Start button is visible."}]
        encoded = processor(text=processor.apply_chat_template(messages, tokenize=False), images=[screenshot], return_tensors="pt")
        ids = encoded["input_ids"]
        pos, _ = get_rope_index_qwen3vl(processor, input_ids=ids, image_grid_thw=encoded["image_grid_thw"])
        mask = torch.zeros_like(ids)
        mask[:, -8:-1] = 1
        item = {"input_ids": ids, "loss_mask": mask, "position_ids": torch.cat([torch.arange(ids.shape[1]).view(1, -1), pos.squeeze(1)], dim=0).unsqueeze(0),
                "seq_len": ids.shape[1], "loss_tokens": int(mask.sum()), "loss_weight": 1.0,
                "multi_modal_inputs": {"pixel_values": encoded["pixel_values"], "image_grid_thw": encoded["image_grid_thw"]}}
        def loss_fn(output, model_inputs, use_megatron=False, cp_rank=0, cp_world_size=1):
            logits = output["logits"] if isinstance(output, dict) else output
            logp, _ = compute_log_probs_with_cp(logits, model_inputs["input_ids"], use_megatron=True)
            mask = model_inputs["loss_mask"]
            loss = -(logp * mask).sum() / mask.sum()
            return loss, {"negative_log_likelihood": loss.detach().item()}
        metrics = worker.forward_backward([copy.deepcopy(item), copy.deepcopy(item)], loss_fn, unpack=True)
        grad_norm = worker.step()
        norm = float(grad_norm)
        assert torch.isfinite(torch.tensor(norm)) and norm > 0, norm
        dist.barrier()
        if worker.rank == 0:
            report = {"passed": True, "model_path": cfg["model_path"], "physical_gpus": [1, 2, 5, 6], "tp": 2, "dp": 2,
                      "grad_norm": norm, "sequence_length": ids.shape[1], "scope": "Disposable multimodal negative-log-likelihood optimizer smoke, NOT an online RL update; no checkpoint saved",
                      "peak_memory_gib_rank0": torch.cuda.max_memory_allocated() / 2**30, "metrics": str(metrics)}
            (ROOT / "reports/smoke-actor.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    if "--child" in sys.argv:
        child()
        return
    if socket.gethostname() != "research-a-worker-1":
        raise SystemExit("Scoped to worker-1")
    gpus = [1, 2, 5, 6]
    for gpu in gpus:
        used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        active = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if used > 1024 or active:
            raise SystemExit(f"GPU {gpu} occupied")
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = []
    try:
        for rank, gpu in enumerate(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RANK=str(rank), WORLD_SIZE="4", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1",
                       MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="4", WANDB_MODE="disabled",
                       LD_LIBRARY_PATH=library_path(cfg), HF_HUB_OFFLINE="1", NCCL_DEBUG="WARN", PYTHONUNBUFFERED="1",
                       NVTE_DEBUG="1", NVTE_DEBUG_LEVEL="2")
            with (ROOT / f"runs/actor-rank{rank}.log").open("wb") as log:
                processes.append(subprocess.Popen([str(Path(cfg["rl_env"]) / "bin/python"), str(Path(__file__).resolve()), "--child"], env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            codes = [p.poll() for p in processes]
            if any(c not in {None, 0} for c in codes):
                raise RuntimeError(f"Actor smoke failed: {codes}; inspect runs/actor-rank*.log")
            if all(c == 0 for c in codes):
                print("Multimodal actor optimizer smoke passed")
                return
            time.sleep(3)
        raise TimeoutError("Actor smoke exceeded 15 minutes")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
