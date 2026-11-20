#!/usr/bin/env python3
"""Start or stop this project's single-worker SFT inference service."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
from checkpoint_specs import CHECKPOINTS, checkpoint_spec

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs/inference-service.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("--gpus", default="0")
    parser.add_argument('--checkpoint', choices=CHECKPOINTS, default='osworld-sft')
    parser.add_argument('--replica', help='Independent replica name; keeps the primary service state intact')
    parser.add_argument('--port', type=int, help='Explicit port for an independent replica')
    parser.add_argument('--max-model-len', type=int, default=40960)
    parser.add_argument('--max-images', type=int, default=10)
    parser.add_argument('--max-num-seqs', type=int, default=4)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.5)
    parser.add_argument('--top-p-default', type=float)
    parser.add_argument('--top-k-default', type=int)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    state_path = STATE
    manifest = ROOT / 'reports/model_manifest.json'
    log_name = 'inference-thudm-sft.log'
    spec = checkpoint_spec(ROOT, args.checkpoint)
    cfg.update(spec)
    state_path = ROOT / 'runs' / spec['state_file']
    manifest = ROOT / 'reports' / spec['manifest_file']
    log_name = 'inference-' + args.checkpoint + '.log'
    if args.replica:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', args.replica) or args.port is None:
            parser.error('A replica requires a short lowercase name and an explicit --port')
        state_path = ROOT / 'runs' / f'inference-{args.checkpoint}-{args.replica}.json'
        log_name = f'inference-{args.checkpoint}-{args.replica}.log'
    if args.port is not None:
        if not 1024 <= args.port <= 65535:
            parser.error('--port must be between 1024 and 65535')
        cfg['port'] = args.port
    hostname = socket.gethostname()
    if hostname not in cfg["workers"]:
        parser.error("run on an authorized worker through worker_exec.py")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        process = Path(f"/proc/{state['pid']}/cmdline")
        alive = state["hostname"] == hostname and process.exists() and cfg["model_dir"] in process.read_bytes().decode(errors="replace")
        if args.action == "stop":
            if state["hostname"] != hostname:
                parser.error("stop must run on the worker that owns the service")
            if alive:
                os.killpg(state["pid"], signal.SIGTERM)
            state["stop_requested_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps({"stop_requested": alive, "pid": state["pid"]}))
            return
        if alive:
            parser.error("this project already has a live inference service")
        if state["hostname"] != hostname:
            parser.error("existing service state belongs to another worker; inspect it first")
    elif args.action == "stop":
        parser.error("no recorded service to stop")
    ids = args.gpus.split(",")
    if not all(s.isdigit() and 0 <= int(s) < 8 for s in ids) or len(set(ids)) != len(ids):
        parser.error("GPU IDs must be distinct numbers in 0..7")
    for device in ids:
        raw = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(raw.strip()) > 1024:
            parser.error(f"GPU {device} is occupied ({raw.strip()} MiB); refusing to share it")
        raw = subprocess.check_output(["nvidia-smi", "-i", device, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if raw.strip():
            parser.error(f"GPU {device} has an existing compute process")
    if not manifest.exists() or json.loads(manifest.read_text())["revision"] != cfg["model_revision"]:
        parser.error("run download_model.py and verify the checkpoint first")
    with socket.socket() as s:
        s.bind(("0.0.0.0", cfg["port"]))
    run_dir = ROOT / "runs"
    run_dir.mkdir(exist_ok=True)
    cache = Path(cfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, VLLM_USE_V1="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8", VLLM_CACHE_ROOT=str(cache / "vllm"), TRITON_CACHE_DIR=str(cache / "triton"))
    compatibility = None
    if args.checkpoint == 'scienceboard-sft':
        adapter_dir = ROOT / 'scripts/scienceboard_tokenizer_compat'
        adapter = adapter_dir / 'sitecustomize.py'
        env['PYTHONPATH'] = str(adapter_dir) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['CUA_SCI_TOKENIZER_MODEL_ROOT'] = cfg['model_dir']
        compatibility = {'kind': 'extra_special_tokens_list_to_transformers4_schema',
                         'adapter': str(adapter), 'sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
                         'checkpoint_files_modified': False}
    command = [str(Path(cfg["policy_env"]) / "bin/python"), "-m", "vllm.entrypoints.openai.api_server", "--model", cfg["model_dir"], "--served-model-name", cfg["served_model_name"], "--host", "0.0.0.0", "--port", str(cfg["port"]), "--tensor-parallel-size", str(len(ids)), "--dtype", "bfloat16", "--max-model-len", "40960", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.5", "--limit-mm-per-prompt", '{"image":10}', "--enforce-eager", "--disable-log-requests", "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3", "--chat-template-content-format", "openai"]
    for flag, value in [('--max-model-len', args.max_model_len), ('--max-num-seqs', args.max_num_seqs),
                        ('--gpu-memory-utilization', args.gpu_memory_utilization),
                        ('--limit-mm-per-prompt', json.dumps({'image': args.max_images}))]:
        command[command.index(flag) + 1] = str(value)
    overrides = {}
    if args.top_p_default is not None:
        if not 0 < args.top_p_default <= 1:
            parser.error('--top-p-default must be in (0,1]')
        overrides['top_p'] = args.top_p_default
    if args.top_k_default is not None:
        overrides['top_k'] = args.top_k_default
    if overrides:
        command += ['--override-generation-config', json.dumps(overrides)]
    log_path = run_dir / log_name
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    state = {"hostname": hostname, "pid": process.pid, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpus": ids, "base_url": f"http://{hostname}.job-research-a.training.svc.cluster.local:{cfg['port']}/v1", "command": command, "model_revision": cfg["model_revision"], "log": str(log_path), "status": "starting; readiness must be checked via /health"}
    state.update(checkpoint=args.checkpoint, replica=args.replica, model_id=cfg['model_id'], model_dir=cfg['model_dir'],
                 served_model_name=cfg['served_model_name'], generation_overrides=overrides,
                 runtime_compatibility=compatibility)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
