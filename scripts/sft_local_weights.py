"""Create a private model view backed by independently digest-checked node-local shards."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import os
import shutil
import tempfile


def prepare_weights(checkpoint, revision, source, shards, view, lock):
    record = json.loads(lock.read_text())[checkpoint]
    expected = record['sha256']
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    if record['revision'] != revision or set(index['weight_map'].values()) != set(expected):
        raise ValueError('Checkpoint revision or weight index mismatch')
    view.mkdir()
    def transfer(name, directory, copy):
        src = (shards or source) / name
        digest = hashlib.sha256()
        with src.open('rb') as stream:
            if copy:
                with (directory / name).open('xb') as target:
                    for block in iter(lambda: stream.read(8 * 1024**2), b''):
                        digest.update(block)
                        target.write(block)
                (directory / name).chmod(0o400)
            else:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
        if digest.hexdigest() != expected[name]:
            raise ValueError(f'Invalid checkpoint shard {name}')

    if shards is None:
        cache = Path(tempfile.gettempdir()) / f'cua-sft-weights-{os.getuid()}'
        cache.mkdir(mode=0o700, exist_ok=True)
        directory = cache / revision
        with (cache / (revision + '.lock')).open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            def identities(folder):
                result = {}
                for name in expected:
                    stat = (folder / name).stat()
                    result[name] = [stat.st_size, stat.st_ctime_ns, stat.st_ino]
                return result
            if not directory.exists():
                if shutil.disk_usage(cache).free < sum((source / name).stat().st_size for name in expected):
                    raise RuntimeError('Insufficient local disk space for SFT checkpoint')
                with tempfile.TemporaryDirectory(dir=cache) as tmp:
                    stage = Path(tmp) / 'weights'
                    stage.mkdir()
                    with ThreadPoolExecutor(4) as pool:
                        list(pool.map(lambda name: transfer(name, stage, True), expected))
                    (stage / 'manifest.json').write_text(json.dumps({'sha256': expected, 'files': identities(stage)}))
                    stage.rename(directory)
            if json.loads((directory / 'manifest.json').read_text()) != {'sha256': expected, 'files': identities(directory)}:
                raise ValueError('Node-local SFT weight cache changed')
    else:
        directory = shards.resolve(strict=True)
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda name: transfer(name, directory, False), expected))
    for name in expected:
        (view / name).symlink_to(directory / name)
    for path in source.iterdir():
        if path.suffix in ('.json', '.jinja', '.model', '.txt') and path.is_file():
            (view / path.name).symlink_to(path.resolve())
    return expected
