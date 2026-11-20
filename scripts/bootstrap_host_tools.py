#!/usr/bin/env python3
"""Install private build/QEMU tools by extraction, without changing system packages."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import urllib.request
import zipfile
import concurrent.futures
import shlex
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    if socket.gethostname() not in cfg["workers"]:
        raise SystemExit("Run on an authorized worker")
    base = Path(cfg["cache_dir"]) / "host-tools"
    base.mkdir(parents=True, exist_ok=True)
    resources = [
        ("https://go.dev/dl/go1.27.4.linux-amd64.tar.gz", "go", "7226736c2339719cda28fbf3a54d71aa63bcefe376e98cef00034268dab24840"),
        ("https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-x64.tar.xz", "node-v22.16.0-linux-x64", None),
        ("https://github.com/protocolbuffers/protobuf/releases/download/v27.5/protoc-27.5-linux-x86_64.zip", "protoc", None),
    ]
    report = []
    for url, folder, checksum in resources:
        archive = base / url.rsplit("/", 1)[-1]
        if not archive.exists():
            urllib.request.urlretrieve(url, str(archive) + ".partial")
            Path(str(archive) + ".partial").rename(archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if checksum:
            assert digest == checksum
        if not (base / folder).exists():
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as z:
                    z.extractall(base / folder)
                (base / folder / "bin/protoc").chmod(0o755)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(base, filter="data")
        report.append({"url": url, "sha256": digest})
    apt_base = Path(cfg["cache_dir"]) / "qemu-system"
    lists = Path(cfg["cache_dir"]) / "browser-system/lists"
    archives, prefix = apt_base / "archives", apt_base / "root"
    for p in [archives / "partial", prefix]:
        p.mkdir(parents=True, exist_ok=True)
    apt_config = apt_base / "apt.conf"
    apt_config.write_text('#clear APT::Update::Post-Invoke;\n#clear APT::Update::Post-Invoke-Success;\n#clear DPkg::Post-Invoke;\n')
    opts = ["-c", str(apt_config), "-o", f"Dir::State::Lists={lists}", "-o", f"Dir::Cache::archives={archives}",
            "-o", "Dir::State::status=/dev/null", "-o", "Debug::NoLocking=1", "-o", "APT::Get::List-Cleanup=0"]
    listing = subprocess.check_output(["apt-get", *opts, "--print-uris", "--download-only", "--no-install-recommends", "-y", "install",
                    "qemu-system-x86", "qemu-utils", "seabios", "ipxe-qemu", "redis-server"], text=True)
    def fetch(line):
        url, name, size, checksum = shlex.split(line)
        target = archives / name
        if not target.exists() or target.stat().st_size != int(size):
            urls = [url.replace("http://", "https://", 1)]
            if "/ubuntu/" in url:
                urls.append("https://mirrors.edge.kernel.org/ubuntu/" + url.split("/ubuntu/", 1)[1])
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(urls[attempt % len(urls)], timeout=20) as stream, open(str(target) + ".partial", "wb") as output:
                        shutil.copyfileobj(stream, output)
                    assert Path(str(target) + ".partial").stat().st_size == int(size), name
                    Path(str(target) + ".partial").rename(target)
                    break
                except Exception:
                    if attempt == 3:
                        raise
        algorithm, expected = checksum.split(":", 1)
        algorithm = "md5" if algorithm.lower() == "md5sum" else algorithm.lower()
        assert hashlib.new(algorithm, target.read_bytes()).hexdigest() == expected, name
        return name
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        downloaded = list(pool.map(fetch, [line for line in listing.splitlines() if line.startswith("'http")]))
    print(f"Verified {len(downloaded)} private APT package downloads", flush=True)
    for archive in sorted(archives.glob("*.deb")):
        subprocess.run(["dpkg-deb", "-x", str(archive), str(prefix)], check=True)
    report.append({"private_apt_prefix": str(prefix), "packages": [p.name for p in archives.glob("*.deb")]})
    (ROOT / "reports/host-tools.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Private Go, Node, protoc, QEMU and Redis tools extracted.")


if __name__ == "__main__":
    main()
