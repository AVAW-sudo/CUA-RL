"""Stage the project-owned OSWorld CPU runtime from shared storage once per node."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time


def publish_worker(site, project, destination, controller):
    """Bundle a validated worker environment and its frozen task and source trees."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ais-worker-pack-') as tmp:
        local = Path(tmp) / 'worker.tar.gz'
        with tarfile.open(local, 'w:gz', compresslevel=1, dereference=False) as tar:
            def keep(info):
                return None if '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc') else info
            for arcname, directory in [('site', site), ('SCALE-CUA', project / 'SCALE-CUA'),
                                       ('data/ScaleCUA', project / 'data/ScaleCUA')]:
                tar.add(directory, arcname, filter=keep)
            tar.add(controller, 'bin/agentrl-controller', recursive=False)
        with local.open('rb') as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                checksum.update(block)
        digest = checksum.hexdigest()
        with tempfile.TemporaryDirectory(dir=destination) as staging:
            staging = Path(staging)
            output = destination / (digest + '.tar.gz')
            if not output.exists():
                shutil.copyfile(local, staging / 'archive')
                (staging / 'archive').replace(output)
            manifest = {'sha256': digest, 'bytes': local.stat().st_size, 'python_abi': 'cp310'}
            (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            (staging / 'manifest.json').replace(destination / 'manifest.json')
    return manifest


def publish_frontier_archive(archive, destination):
    """Publish independently transferable pieces of a frozen site/ archive."""
    size = archive.stat().st_size
    piece_bytes = 256 * 1024**2
    destination.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    with archive.open('rb') as src:
        for data in iter(lambda: src.read(8 * 1024**2), b''):
            checksum.update(data)
    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        def publish(offset):
            length = min(piece_bytes, size - offset)
            pending = Path(tmp) / str(offset)
            part_hash = hashlib.sha256()
            with archive.open('rb') as src, pending.open('wb') as dst:
                src.seek(offset)
                remaining = length
                while remaining:
                    data = src.read(min(remaining, 8 * 1024**2))
                    if not data:
                        raise EOFError('Frontier archive changed during publication')
                    dst.write(data)
                    part_hash.update(data)
                    remaining -= len(data)
            digest = part_hash.hexdigest()
            pending.replace(destination / (digest + '.part'))
            return {'sha256': digest, 'bytes': length}
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(publish, range(0, size, piece_bytes)))
        manifest = {'sha256': checksum.hexdigest(), 'bytes': size, 'parts': parts, 'python_abi': 'cp310'}
        pending = Path(tmp) / 'manifest.json'
        pending.write_text(json.dumps(manifest, indent=2) + '\n')
        pending.replace(destination / 'manifest.json')
    return manifest


def fetch_pieces(manifest, parts, scratch):
    def fetch(item):
        digest = item['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid runtime piece SHA-256')
        local = scratch / (digest + '.part')
        checksum = hashlib.sha256()
        with (manifest.parent / local.name).open('rb') as src, local.open('wb') as dst:
            for data in iter(lambda: src.read(8 * 1024**2), b''):
                checksum.update(data)
                dst.write(data)
        if checksum.hexdigest() != digest or local.stat().st_size != item['bytes']:
            raise ValueError(f'Runtime piece failed integrity check: {digest}')
        return local
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, {p['sha256']: p for p in parts}.values()))
    indexed = {p.stem: p for p in paths}
    return [indexed[item['sha256']] for item in parts]


def stage_runtime(manifest, role):
    required = {
        'worker': ('site', 'SCALE-CUA', 'data/ScaleCUA'),
        'fixtures': ('tasks',),
        'sft-ocr': ('site/easyocr', 'models/model'),
        'frontier': ('site/vllm', 'site/torch', 'site/transformers'),
    }[role]
    started = time.monotonic()
    descriptor = json.loads(manifest.read_text())
    digest = descriptor['sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError(f'Invalid {role} archive SHA-256')
    archive = manifest.parent / (digest + '.tar.gz')
    cache = Path(tempfile.gettempdir()) / f'cua-{role}-cache-{os.getuid()}'
    cache.mkdir(mode=0o700, exist_ok=True)
    target = cache / digest
    with (cache / (digest + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = (target / 'ready').is_file()
        if not hit:
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                local = Path(tmp) / 'runtime.tar.gz'
                checksum = hashlib.sha256()
                pieces = (fetch_pieces(manifest, descriptor['parts'], Path(tmp))
                          if 'parts' in descriptor else [archive])
                with local.open('wb') as dst:
                    for piece in pieces:
                        with piece.open('rb') as src:
                            for chunk in iter(lambda: src.read(8 * 1024**2), b''):
                                checksum.update(chunk)
                                dst.write(chunk)
                for piece in set(pieces):
                    if piece.parent == Path(tmp):
                        piece.unlink()
                if checksum.hexdigest() != digest:
                    raise ValueError(f'{role} archive does not match its manifest')
                unpacked = Path(tmp) / 'runtime'
                unpacked.mkdir()
                with tarfile.open(local, 'r:gz') as tar:
                    tar.extractall(unpacked, filter='data')
                for name in required:
                    if not (unpacked / name).is_dir():
                        raise ValueError(f'Incomplete {role} archive: {name}')
                if role == 'sft-ocr' and any(not (unpacked / 'models/model' / n).is_file()
                                             for n in ('craft_mlt_25k.pth', 'english_g2.pth')):
                    raise ValueError('OCR weights missing from the evaluation runtime')
                if role == 'worker' and not os.access(unpacked / 'bin/agentrl-controller', os.X_OK):
                    raise ValueError('Missing controller executable in worker archive')
                (unpacked / 'ready').touch()
                unpacked.rename(target)
    return target, {'sha256': digest, 'cache_hit': hit, 'stage_seconds': round(time.monotonic() - started, 3)}


def stage_worker(manifest):
    return stage_runtime(manifest, 'worker')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True, type=Path)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--controller', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish_worker(args.site, args.project, args.destination, args.controller)), flush=True)
