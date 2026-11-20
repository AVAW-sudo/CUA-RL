#!/usr/bin/env python3
"""Read-only RL readiness audit; exit 2 until the required runtime/controller are ready."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import urllib.parse
import urllib.request

import yaml
from runtime_env import library_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3vl-osworld-smoke-6gpu.yaml")
    parser.add_argument("--gpus", default="0,1,2,5,6,7")
    args = parser.parse_args()
    paths = json.loads((ROOT / "configs/paths.json").read_text())
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "worker": socket.gethostname(),
              "config": args.config, "checks": {}, "blockers": []}
    checks, blockers = report["checks"], report["blockers"]
    if report["worker"] not in paths["workers"]:
        parser.error("run on worker-0 or worker-1 via worker_exec.py")
    manifest = ROOT / "reports/model_manifest.json"
    checks["model_verified"] = manifest.exists() and json.loads(manifest.read_text()).get("revision") == paths["model_revision"]
    if not checks["model_verified"]:
        blockers.append("SFT checkpoint verification missing")
    data = ROOT / "reports/training-data-manifest.json"
    checks["task_pool_verified"] = data.exists() and json.loads(data.read_text()).get("task_count") == 2049
    if not checks["task_pool_verified"]:
        blockers.append("RL task pool verification missing")
    checks["gpus"] = []
    if len(set(args.gpus.split(','))) != cfg.get('num_gpu_per_mach', 8):
        blockers.append('GPU list size does not match configured GPUs per node')
    for gpu in args.gpus.split(","):
        if not gpu.isdigit() or not 0 <= int(gpu) < 8:
            parser.error("invalid GPU id")
        memory = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        processes = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        checks["gpus"].append({"id": int(gpu), "memory_mib": memory, "has_process": bool(processes)})
        if memory > 1024 or processes:
            blockers.append(f"GPU {gpu} occupied; if this is the SFT service, stop it with serve_model.py stop before training")
    checks["container_kvm"] = Path("/dev/kvm").exists()
    agent_version = cfg['task'].get('task_kwargs', {}).get('agent_version')
    if agent_version == 'anthropic_rl':
        agent_file = ROOT / 'SCALE-CUA/osworld_eval/mm_agents/anthropic_rl/main.py'
        checks['original_rl_agent_exists'] = agent_file.is_file()
        if not agent_file.is_file():
            blockers.append('Public source references anthropic_rl but omits its module. The separately named scalecua_osworld eval-agent adapter is available only for bring-up, not exact reproduction.')
    task_names = cfg["task"].get("train_tasks", ["osworld-rl"])
    real_indices = []
    endpoint = cfg["task"]["base_url"].rstrip("/") + "/get_indices?" + urllib.parse.urlencode({"name": task_names[0]})
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            indices = json.load(response)
        # AgentRL's Go controller returns the index list directly.
        real_indices = [i for i in indices if i != -1] if isinstance(indices, list) else []
        checks["controller_indices_count"] = len(real_indices)
        if not checks["controller_indices_count"]:
            blockers.append("Controller returned no OSWorld RL indices")
        elif task_names == ['osworld-rl'] and len(set(real_indices)) != 2049:
            blockers.append('Full-pool baseline requires exactly 2049 unique controller task indices')
    except Exception as exc:
        checks["controller_error"] = str(exc)
        blockers.append("OSWorld controller/VM workers are not reachable")
    python = Path(paths["rl_env"]) / "bin/python"
    checks["rl_python_exists"] = python.exists()
    checks["runtime_imports"] = False
    if python.exists():
        run = subprocess.run([str(python), "-c", "import agentrl.trainer.workers.registry; import megatron.core; import transformer_engine; import swift; import vllm; print('ok')"], capture_output=True, text=True, timeout=120,
                             env=dict(os.environ, LD_LIBRARY_PATH=library_path(paths)))
        checks["runtime_imports"] = run.returncode == 0
        checks["runtime_error"] = run.stderr[-3000:] if run.returncode else None
    if not checks["runtime_imports"]:
        blockers.append("Dedicated AgentRL/Megatron runtime is not ready; inference environment is not a training substitute")
    if cfg.get("dynamic_sampling", {}).get("enabled"):
        rate = cfg["dynamic_sampling"].get("initial_pass_rate_file")
        checks["frontier_initial_rates_exists"] = bool(rate and Path(rate).is_file())
        if not checks["frontier_initial_rates_exists"]:
            blockers.append("Frontier Sampling needs four SFT rollouts per public RL task to initialize rates")
        else:
            try:
                rates = json.loads(Path(rate).read_text())
                for name in task_names:
                    values = rates[name]
                    assert isinstance(values, dict) and values
                    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1 for v in values.values())
                    if name == 'osworld-rl':
                        expected = json.loads((ROOT / 'configs/rl-indices.json').read_text())
                        assert set(values) == set(expected), 'Rates must cover the exact 2049-task pool'
                    if name == task_names[0] and real_indices:
                        assert {str(i) for i in real_indices} <= set(values), 'Rates omit controller tasks'
                checks['frontier_rates_schema_and_coverage'] = True
            except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
                checks['frontier_rates_schema_and_coverage'] = False
                blockers.append(f'Frontier initial rates have invalid values or incomplete coverage: {exc}')
    report["ready"] = not blockers
    (ROOT / "reports/rl-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
