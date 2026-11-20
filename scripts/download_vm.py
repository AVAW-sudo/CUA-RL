#!/usr/bin/env python3
"""Download a pinned public VM, extract one expected image, verify ZIP CRC and SHA256."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import zipfile

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['osworld', 'scienceboard'])
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/paths.json').read_text())
    if socket.gethostname() not in cfg['workers']:
        parser.error('Download/extraction is restricted to worker-0/1')
    spec = json.loads((ROOT / 'configs/sources.lock.json').read_text())[
        'smoke_vm' if args.benchmark == 'osworld' else 'scienceboard_vm']
    target = ROOT / spec['extracted_file']
    target.parent.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'reports' / f'{args.benchmark}-vm-manifest.json'
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            if report_path.exists():
                old = json.loads(report_path.read_text())
                if old.get('revision') == spec['revision'] and old.get('image_bytes') == target.stat().st_size:
                    print(json.dumps({'status': 'reusing previous verified image', 'report': str(report_path)}))
                    return
            raise RuntimeError(f'Existing image has no matching manifest; will not overwrite {target}')
        info = HfApi().dataset_info(spec['repo_id'], revision=spec['revision'], files_metadata=True)
        assert info.sha == spec['revision']
        entry = next(x for x in info.siblings if x.rfilename == spec['filename'])
        archive = Path(hf_hub_download(spec['repo_id'], spec['filename'], repo_type='dataset',
                                      revision=spec['revision'], local_dir=str(target.parent / 'download')))
        assert archive.stat().st_size == entry.size
        print(json.dumps({'status': 'extracting', 'archive': str(archive)}), flush=True)
        temporary = target.with_suffix('.qcow2.partial')
        digest = hashlib.sha256()
        with zipfile.ZipFile(archive) as zipped:
            expected_member = spec.get('archive_member', target.name)
            members = [m for m in zipped.infolist() if not m.is_dir() and m.filename == expected_member]
            if len(members) != 1:
                raise RuntimeError(f'Expected one {expected_member}; found {[m.filename for m in members]}')
            member = members[0]
            with zipped.open(member) as source, temporary.open('wb') as out:
                while block := source.read(16 * 1024 * 1024):
                    digest.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            assert temporary.stat().st_size == member.file_size
            temporary.replace(target)
        report = {'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'repo_id': spec['repo_id'], 'revision': spec['revision'], 'archive_bytes': entry.size,
                  'image': str(target), 'image_bytes': target.stat().st_size, 'image_sha256': digest.hexdigest(),
                  'archive_member': member.filename, 'zip_crc32': f'{member.CRC:08x}',
                  'checks': ['Hub archive size', 'ZIP member CRC32', 'image byte count', 'local image SHA256 recorded']}
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
