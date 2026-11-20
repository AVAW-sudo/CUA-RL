"""Apply a recorded TE 2.12 compatibility fix for the pinned FA 2.8.3 build tag."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'configs/paths.json').read_text())
site = Path(cfg['rl_env']) / 'lib/python3.12/site-packages'
target = site / 'transformer_engine/pytorch/attention/dot_product_attention/backends.py'
old = 'fa_utils.version = PkgVersion(get_pkg_version("flash-attn"))'
new = 'fa_utils.version = PkgVersion(PkgVersion(get_pkg_version("flash-attn")).public)'
before = target.read_text()
assert (site / 'transformer_engine-2.12.0.dist-info').is_dir()
if old in before:
    assert before.count(old) == 1
    after = before.replace(old, new)
    backup = ROOT / 'runs/runtime-patches/te-backends.original.py'
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(before)
    patchdir = ROOT / 'patches'
    patchdir.mkdir(exist_ok=True)
    (patchdir / 'transformer-engine-fa-version.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='a/' + str(target.relative_to(site)), tofile='b/' + str(target.relative_to(site)))))
    target.write_text(after)
elif new in before:
    after = before
    before = (ROOT / 'runs/runtime-patches/te-backends.original.py').read_text()
else:
    raise RuntimeError('Unexpected TE source; refusing patch')
report = {'file': str(target), 'reason': 'PEP 440 local CUDA/ABI build suffix incorrectly exceeds TE maximum public version 2.8.3; compare public API version while preserving installed package and binary identity',
          'sha256_before': hashlib.sha256(before.encode()).hexdigest(), 'sha256_after': hashlib.sha256(after.encode()).hexdigest()}
(ROOT / 'reports/transformer-engine-patch.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
