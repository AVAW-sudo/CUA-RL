"""Full RAM/device/disk snapshots in private local overlays, with reset checks."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import urllib.request
from qemu_runtime import LocalVM, ROOT


class SnapshotVM(LocalVM):
    def __init__(self, address, cpus=4, memory_mb=8192, base_image=None):
        super().__init__(address, cpus, memory_mb, 'tcg', base_image)
        self.audit_directory = self.directory
        self.directory = Path('/tmp') / f'cua-rl-snapshots-{os.getuid()}' / self.id
        self.directory.mkdir(parents=True)
        self.clean = False

    def start(self):
        state = super().start()
        state['reset_backend'] = 'full_qemu_savevm_loadvm'
        (self.audit_directory / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        return state

    def monitor(self, command, timeout=180):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(timeout)
            stream.connect(str(self.directory / 'monitor.sock'))
            def receive():
                value = b''
                while b'(qemu)' not in value:
                    block = stream.recv(65536)
                    if not block: raise RuntimeError('QEMU monitor closed')
                    value += block
                return value.decode(errors='replace')
            receive()
            stream.sendall(command.encode() + b'\n')
            result = receive()
            if any(word in result.lower() for word in ['error', 'failed', 'does not exist']):
                raise RuntimeError(result)
            return result

    def guest(self, code):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(f'http://{self.address}:5000/execute',
            data=json.dumps({'command': ['python', '-c', code]}).encode(),
            headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=60) as response:
            result = json.load(response)
        if result.get('returncode') != 0:
            raise RuntimeError('Guest reset check failed: ' + str(result))
        return result.get('output', '').strip()

    def save_clean(self):
        self.guest("from pathlib import Path; import pyautogui; "
                   f"Path('/tmp/cua-baseline-identity').write_text({self.id!r}); "
                   "pyautogui.moveTo(400,300)")
        self.monitor('savevm cua_clean')
        self.clean = True

    def restore_clean(self):
        if not self.clean: raise RuntimeError('No clean snapshot')
        started = time.monotonic()
        self.monitor('loadvm cua_clean')
        self.monitor('cont')
        self.wait_ready(timeout=90)
        # loadvm restores the old guest wall clock. Fresh boots use current RTC;
        # align to the host epoch before any task setup or network interaction.
        # 'password' is the documented public OSWorld image's guest sudo password,
        # never a host/platform credential. Try passwordless sudo first.
        epoch = time.time()
        self.guest("import subprocess; "
            f"cmd=['sudo','-n','date','-u','-s','@{epoch:.6f}']; "
            "r=subprocess.run(cmd,capture_output=True,text=True); "
            "r=r if r.returncode==0 else subprocess.run(['sudo','-S','-p','','date','-u','-s',cmd[-1]],input='password\\n',capture_output=True,text=True); "
            "assert r.returncode==0, 'Guest clock correction failed: '+r.stderr")
        measured = json.loads(self.guest("from pathlib import Path; import json,pyautogui,time; "
             "print(json.dumps({'baseline':Path('/tmp/cua-baseline-identity').read_text(),"
             "'dirty':Path('/tmp/cua-reset-dirty').exists(),'position':list(pyautogui.position()),'epoch':time.time()}))"))
        if measured['baseline'] != self.id or measured['dirty'] or measured['position'] != [400,300]:
            raise RuntimeError('Snapshot disk/input state did not reset: ' + str(measured))
        measured['clock_skew_seconds'] = abs(time.time() - measured['epoch'])
        if measured['clock_skew_seconds'] > 60:
            raise RuntimeError('Restored guest wall clock is stale: ' + str(measured))
        measured['reset_seconds'] = time.monotonic() - started
        return measured


def implementation_hash():
    digest = hashlib.sha256()
    for name in ['snapshot_vm.py', 'qemu_runtime.py', 'local_osworld.py']:
        digest.update((ROOT / 'scripts' / name).read_bytes())
    return digest.hexdigest()
