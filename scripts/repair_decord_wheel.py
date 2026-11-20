#!/usr/bin/env python3
"""Repair the upstream py3 Decord wheel's contradictory internal cp36 tag.

Only WHEEL/RECORD metadata changes. The ctypes library and Python files retain
their exact bytes. The original filename already advertises py3-none.
"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILENAME = "decord-0.6.0-py3-none-manylinux2010_x86_64.whl"


def main():
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    output = Path(cfg["cache_dir"]) / "wheels" / FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/decord/0.6.0/json", timeout=30))
    source = next(entry for entry in metadata["urls"] if entry["filename"] == FILENAME)
    original = urllib.request.urlopen(source["url"], timeout=120).read()
    assert hashlib.sha256(original).hexdigest() == source["digests"]["sha256"]
    record_path = "decord-0.6.0.dist-info/RECORD"
    wheel_path = "decord-0.6.0.dist-info/WHEEL"
    records = []
    payload_hashes = {}
    with zipfile.ZipFile(io.BytesIO(original)) as src, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            if name == record_path:
                continue
            data = src.read(name)
            if name == wheel_path:
                old = b"Tag: cp36-cp36m-manylinux2010_x86_64"
                assert old in data
                data = data.replace(old, b"Tag: py3-none-manylinux2010_x86_64")
            else:
                payload_hashes[name] = hashlib.sha256(data).hexdigest()
            dst.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((name, "sha256=" + digest, str(len(data))))
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(records + [(record_path, "", "")])
        dst.writestr(record_path, stream.getvalue())
    with zipfile.ZipFile(output) as patched:
        assert all(hashlib.sha256(patched.read(name)).hexdigest() == digest for name, digest in payload_hashes.items())
    report = {"source_url": source["url"], "source_sha256": source["digests"]["sha256"], "patched_wheel": str(output), "patched_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "changed_files": [wheel_path, record_path], "payload_unchanged": True, "reason": "Match internal metadata to the published py3-none wheel filename for the ctypes binding"}
    (ROOT / "reports/decord-wheel-repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
