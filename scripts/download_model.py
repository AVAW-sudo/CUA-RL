#!/usr/bin/env python3
"""Download the revision-pinned model and verify Hub sizes and safetensors headers."""
import datetime
import argparse
import json
from pathlib import Path
import struct
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--reference', action='store_true', help='Download the released RL reference to a separate directory')
    mode.add_argument('--scienceboard-sft', action='store_true', help='Download the separate official ScienceBoard SFT control')
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    report_name = 'model_manifest.json'
    if args.reference:
        spec = json.loads((ROOT / 'configs/sources.lock.json').read_text())['released_rl_model']
        cfg.update(model_id=spec['repo_id'], model_revision=spec['revision'],
                   model_dir=str(Path(cfg['model_dir']).with_name('THUDM-ScaleCUA-qwen3-vl-osworld-rl-reference')))
        report_name = 'released-rl-model-manifest.json'
    elif args.scienceboard_sft:
        spec = json.loads((ROOT / 'configs/sources.lock.json').read_text())['scienceboard_sft_model']
        cfg.update(model_id=spec['repo_id'], model_revision=spec['revision'],
                   model_dir=str(Path(cfg['model_dir']).with_name('THUDM-ScaleCUA-qwen3-vl-scienceboard-sft')))
        report_name = 'scienceboard-sft-model-manifest.json'
    api = HfApi()
    info = api.model_info(cfg["model_id"], revision=cfg["model_revision"], files_metadata=True)
    assert info.sha == cfg['model_revision']
    target = Path(snapshot_download(repo_id=cfg["model_id"], revision=cfg["model_revision"], local_dir=cfg["model_dir"], max_workers=4))
    files = []
    for entry in info.siblings:
        path = target / entry.rfilename
        assert path.is_file(), f"Missing {path}"
        size = path.stat().st_size
        assert entry.size is None or size == entry.size, f"Size mismatch: {path}"
        files.append({"name": entry.rfilename, "bytes": size})
    index = json.loads((target / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        with (target / shard).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < length < 100_000_000
            header = json.loads(stream.read(length))
            payload_bytes = (target / shard).stat().st_size - 8 - length
            for key, value in header.items():
                if key != "__metadata__":
                    begin, end = value["data_offsets"]
                    assert 0 <= begin <= end <= payload_bytes, (shard, key)
            headers[shard] = header
    for name, shard in index["weight_map"].items():
        assert name in headers[shard], (name, shard)
    report = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "repo_id": cfg["model_id"], "revision": info.sha, "local_dir": str(target), "checks": ["Hub file sizes", "safetensors header offsets", "weight index coverage"], "files": files}
    (ROOT / 'reports' / report_name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": str(target), "revision": info.sha, "bytes": sum(x["bytes"] for x in files), "status": "verified"}))


if __name__ == "__main__":
    main()
