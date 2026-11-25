"""Cache one verified, immutable evaluation base image per AIS node."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

# SHA-256 of the extracted files in the verified migration inventory.
DIGESTS = {'osworld': '6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313',
           'scienceboard': '8c60e0110f721d21c46d97e88f7028e3c8e83de4e4b9dbb93bad6e2631552951'}


def verified_image(benchmark, shared, expected_size):
    digest = DIGESTS[benchmark]
    cache = Path(tempfile.gettempdir()) / f'cua-sft-images-{os.getuid()}'
    cache.mkdir(mode=0o700, exist_ok=True)
    target = Path(os.environ.get('CUA_EVAL_VM_IMAGE', cache / (benchmark + '.qcow2'))).resolve()
    override = 'CUA_EVAL_VM_IMAGE' in os.environ
    marker = cache / (benchmark + '.json')
    with (cache / (benchmark + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        def identity():
            st = target.stat()
            return {'path': str(target), 'bytes': st.st_size, 'inode': st.st_ino, 'device': st.st_dev,
                    'ctime_ns': st.st_ctime_ns, 'mtime_ns': st.st_mtime_ns, 'sha256': digest}
        if not override and target.is_file() and marker.is_file() and json.loads(marker.read_text()) == identity():
            return target, digest
        source = target if target.is_file() or override else shared
        if source.stat().st_size != expected_size:
            raise ValueError(f'Unexpected base image size: {source}')
        saving = source != target
        if saving and shutil.disk_usage(cache).free < expected_size:
            raise RuntimeError(f'Insufficient node-local capacity to stage {source}')
        with tempfile.NamedTemporaryFile(dir=cache) as tmp, source.open('rb') as src:
            checksum = hashlib.sha256()
            for block in iter(lambda: src.read(8 * 1024**2), b''):
                checksum.update(block)
                if saving:
                    tmp.write(block)
            if checksum.hexdigest() != digest:
                raise ValueError(f'Base image failed SHA-256 verification: {source}')
            if saving:
                tmp.flush()
                os.chmod(tmp.name, 0o400)
                os.link(tmp.name, target)
        if not override:
            pending = marker.with_suffix('.tmp')
            pending.write_text(json.dumps(identity()) + '\n')
            pending.replace(marker)
    return target, digest
