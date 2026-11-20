#!/usr/bin/env python3
"""Build the upstream Go controller with pinned tools; dashboard is disabled."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    tools = Path(cfg["cache_dir"]) / "host-tools"
    env = dict(os.environ, GOPATH=str(tools / "gopath"), GOCACHE=str(tools / "gocache"), GOMAXPROCS="8")
    env["PATH"] = ":".join([str(tools / "go/bin"), str(tools / "gopath/bin"), str(tools / "protoc/bin"), env["PATH"]])
    for package in ["google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.10", "google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1"]:
        subprocess.run(["go", "install", package], env=env, check=True)
    source = ROOT / "SCALE-CUA/scalecua_rl"
    dest = source / "controller/internal/pb"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["protoc", f"--proto_path={source / 'proto'}", f"--go_out={dest}", "--go_opt=paths=source_relative",
                    f"--go-grpc_out={dest}", "--go-grpc_opt=paths=source_relative", *map(str, sorted((source / "proto").rglob("*.proto")))], env=env, check=True)
    dashboard = source / "controller/dashboard/dist"
    dashboard.mkdir(exist_ok=True)
    (dashboard / "index.html").write_text("Controller built for local API use. Dashboard disabled.\n")
    binary = ROOT / "runs/bin/agentrl-controller"
    binary.parent.mkdir(exist_ok=True)
    subprocess.run(["go", "build", "-p", "8", "-o", str(binary), "."], cwd=source / "controller", env=env, check=True)
    print(binary)


if __name__ == "__main__":
    main()
