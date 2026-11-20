"""Shared runtime library selection for the isolated CUDA 13 training environment."""
from pathlib import Path


def library_path(cfg):
    site = Path(cfg["rl_env"]) / "lib/python3.12/site-packages"
    return ":".join([cfg["cache_dir"] + "/cxx-runtime", str(site / "nvidia/cudnn/lib"),
                     *map(str, sorted((site / "nvidia").glob("*/lib"))), str(site / "torch/lib")])
