#!/usr/bin/env python3
"""Restore the THUDM upstream and the selected public OSWorld RL task pool."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def checkout(url, revision, target, sparse=None):
    fresh = not target.exists()
    if not target.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", url, str(target)], check=True)
    if git("-C", target, "remote", "get-url", "origin") != url:
        raise SystemExit(f"Unexpected source URL: {target}")
    if git("-C", target, "rev-parse", "HEAD") != revision:
        if not fresh:
            raise SystemExit(f"Revision changed at {target}; inspect before updating")
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision], check=True)
    if sparse:
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", *sparse], check=True)
    if fresh:
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
    return git("-C", target, "rev-parse", "HEAD")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--local-patch", action="store_true", help="Apply the recorded bring-up patch; preserve existing local work")
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/sources.lock.json").read_text())
    source = lock["upstream"]
    checkout(source["url"], source["commit"], ROOT / source["path"])
    if args.local_patch:
        repo = ROOT / source["path"]
        for name in ('local-bringup.patch', 'rl-integrity.patch', 'ais-distributed-training.patch',
                     'ais-desktop-lifecycle.patch'):
            patch = ROOT / 'patches' / name
            reverse = subprocess.run(['git', '-C', str(repo), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
            if reverse.returncode:
                subprocess.run(['git', '-C', str(repo), 'apply', '--check', str(patch)], check=True)
                subprocess.run(['git', '-C', str(repo), 'apply', str(patch)], check=True)
    if args.data:
        source = lock["training_data"]
        target = ROOT / "data/ScaleCUA"
        checkout("https://huggingface.co/datasets/" + source["repo_id"], source["revision"], target,
                 ["osworld/rl_tasks", "osworld/judge_functions/rl_tasks"])
        catalog = ROOT / "reports/training-data-catalog.json"
        if not catalog.exists():
            import urllib.request
            url = f"https://huggingface.co/api/datasets/{source['repo_id']}/revision/{source['revision']}?blobs=true"
            with urllib.request.urlopen(url, timeout=60) as stream:
                info = json.load(stream)
            files = [f for f in info["siblings"] if f["rfilename"] == "README.md" or
                     f["rfilename"].startswith(("osworld/rl_tasks/", "osworld/judge_functions/rl_tasks/"))]
            catalog.write_text(json.dumps({"repo_id": source["repo_id"], "revision": info["sha"], "files": files}, indent=2) + "\n")
        spec = json.loads(catalog.read_text())
        assert spec["revision"] == source["revision"]
        files = []
        for item in spec["files"]:
            path = target / item["rfilename"]
            content = path.read_bytes()
            assert len(content) == item["size"], str(path)
            if path.suffix == ".json":
                json.loads(content)
            files.append({"name": item["rfilename"], "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {"repo_id": source["repo_id"], "revision": source["revision"], "files": files,
                    "task_count": sum(f["name"].startswith("osworld/rl_tasks/") and f["name"].endswith(".json") for f in files),
                    "checks": ["pinned git revision", "Hub file sizes", "JSON parse", "local SHA256 recorded"]}
        (ROOT / "reports/training-data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"task_count": manifest["task_count"], "verified_files": len(files), "data_dir": str(target)}))
    print("THUDM sources ready.")


if __name__ == "__main__":
    main()
