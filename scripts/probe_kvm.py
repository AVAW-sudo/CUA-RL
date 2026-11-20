#!/usr/bin/env python3
"""Probe KVM as the normal worker user; optionally expose a missing device node.

Does not alter cgroups, capabilities, host mounts, or Kubernetes permissions.
A node created by this script is removed if the ordinary-user probe fails.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBE = '''import fcntl, json, os
result = {"uid": os.getuid()}
try:
    device = os.open("/dev/kvm", os.O_RDWR | os.O_CLOEXEC)
    try:
        result["api_version"] = fcntl.ioctl(device, 0xAE00, 0)
        assert result["api_version"] == 12
        vm = fcntl.ioctl(device, 0xAE01, 0)
        try:
            vcpu = fcntl.ioctl(vm, 0xAE41, 0)
            os.close(vcpu)
            result["vm_and_vcpu_created"] = True
        finally:
            os.close(vm)
    finally:
        os.close(device)
    result["passed"] = True
except Exception as exc:
    result.update(passed=False, error=str(exc), errno=getattr(exc,"errno",None))
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-device', action='store_true')
    parser.add_argument('--uid', type=int, default=1000)
    parser.add_argument('--gid', type=int, default=100)
    args = parser.parse_args()
    host = socket.gethostname()
    if host not in {'research-a-worker-0', 'research-a-worker-1'}:
        parser.error('Only the two authorized workers may be probed')
    target = Path('/dev/kvm')
    created = False
    before = target.exists()
    major, minor = map(int, Path('/sys/class/misc/kvm/dev').read_text().strip().split(':'))
    assert (major, minor) == (10, 232)
    if args.install_device and not before:
        if os.geteuid() != 0:
            parser.error('--install-device needs the existing authorized container sudo access')
        os.mknod(target, stat.S_IFCHR | 0o660, os.makedev(major, minor))
        created = True
        os.chown(target, args.uid, args.gid)
        os.chmod(target, 0o660)
    if target.exists():
        value = target.stat()
        assert stat.S_ISCHR(value.st_mode) and value.st_rdev == os.makedev(major, minor)

    def drop_privileges():
        os.setgroups([])
        os.setgid(args.gid)
        os.setuid(args.uid)

    p = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True,
                       timeout=20, preexec_fn=drop_privileges if os.geteuid() == 0 else None)
    result = json.loads(p.stdout) if p.returncode == 0 else {'passed': False, 'error': p.stderr}
    if created and not result['passed']:
        target.unlink()
    result.update(hostname=host, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  device_preexisted=before, device_created=created,
                  created_device_retained=created and result['passed'],
                  scope='Container device node only; host/cgroup/security configuration unchanged')
    path = ROOT / 'reports' / ('kvm-' + host.removeprefix('research-a-') + '.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    if os.geteuid() == 0:
        os.chown(path, args.uid, args.gid)
    print(json.dumps(result))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
