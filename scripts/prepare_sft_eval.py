"""Export a private pinned evaluator with the SFT evaluation fixes. Print its path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(os.environ.get('CUA_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def core_hashes(source):
    files = ('run_scalecua_os.py', 'run_scalecua_sci.py', 'lib_run_single.py', 'desktop_env/desktop_env.py',
             'desktop_env/scienceboard/evaluator.py', 'desktop_env/controllers/python.py', 'desktop_env/controllers/setup.py',
             'desktop_env/evaluators/getters/chrome.py', 'desktop_env/evaluators/metrics/docs.py',
             'evaluation_examples/test_nogdrive.json', 'evaluation_examples/test_scienceboard.json')
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}


def prepare(parent=None):
    parent = parent or Path(tempfile.gettempdir()) / f'cua-sft-eval-{os.getuid()}'
    repo = ROOT / 'SCALE-CUA'
    revision = json.loads((ROOT / 'configs/sources.lock.json').read_text())['upstream']['commit']
    patch = ROOT / 'patches/sft-evaluation.patch'
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / (revision + '-' + digest + '-v2')
    source = target / 'osworld_eval'
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            stage = Path(tmp) / 'stage'
            stage.mkdir()
            archive = Path(tmp) / 'source.tar'
            subprocess.run(['git', '-C', str(repo), 'archive', '-o', str(archive), revision, 'osworld_eval'], check=True)
            subprocess.run(['tar', '-xf', str(archive), '-C', str(stage)], check=True)
            subprocess.run(['git', 'apply', '--whitespace=nowarn', str(patch)], cwd=stage, check=True)
            code = stage / 'osworld_eval'
            # No large downloads are part of the Git archive. Each staged evaluator keeps its own cache.
            (code / 'logs').mkdir(exist_ok=True)
            metadata = {'revision': revision, 'patch_sha256': digest, 'core_sha256': core_hashes(code)}
            (code / 'sft-source.json').write_text(json.dumps(metadata, indent=2) + '\n')
            try:
                stage.rename(target)
            except OSError:
                if not source.is_dir():
                    raise
    identity = json.loads((source / 'sft-source.json').read_text())
    if identity != {'revision': revision, 'patch_sha256': digest, 'core_sha256': core_hashes(source)}:
        raise ValueError('Prepared source identity mismatch')
    if os.environ.get('CUA_SFT_PATCH_SHA256', digest) != digest:
        raise ValueError('Evaluation patch changed since the campaign was created')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path(tempfile.gettempdir()) / f'cua-sft-eval-{os.getuid()}')
    print(prepare(parser.parse_args().parent.resolve()))
