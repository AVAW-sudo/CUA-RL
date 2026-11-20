"""Private QEMU backend with explicit acceleration and one overlay per VM."""
import fcntl
import json
import io
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(os.environ.get('CUA_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def command(binary):
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    prefix = Path(cfg["cache_dir"]) / "qemu-system/root"
    loader = prefix / "lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
    return [str(loader), "--library-path", f"{prefix}/usr/lib/x86_64-linux-gnu:{prefix}/lib/x86_64-linux-gnu", str(prefix / "usr/bin" / binary)]


def require_kvm():
    """Check actual device access; never silently fall back to software emulation."""
    try:
        device = os.open('/dev/kvm', os.O_RDWR | os.O_CLOEXEC)
        try:
            if fcntl.ioctl(device, 0xAE00, 0) != 12:
                raise RuntimeError('Unsupported KVM API version')
            vm = fcntl.ioctl(device, 0xAE01, 0)
            try:
                vcpu = fcntl.ioctl(vm, 0xAE41, 0)
                os.close(vcpu)
            finally:
                os.close(vm)
        finally:
            os.close(device)
    except OSError as exc:
        raise RuntimeError(f'KVM is unavailable to uid {os.getuid()}: {exc}; '
                           'platform device allocation is required') from exc


class LocalVM:
    def __init__(self, address="127.0.0.10", cpus=2, memory_mb=4096,
                 accelerator=None, base_image=None):
        cfg = json.loads((ROOT / "configs/paths.json").read_text())
        if socket.gethostname() not in cfg["workers"]:
            raise RuntimeError("QEMU is limited to worker-0/1")
        self.accelerator = accelerator or os.environ.get('CUA_VM_ACCEL', 'tcg')
        if self.accelerator not in {'kvm', 'tcg'}:
            raise ValueError('accelerator must be kvm or tcg')
        if self.accelerator == 'kvm':
            require_kvm()
        if not 1 <= cpus <= 16 or not 2048 <= memory_mb <= 32768:
            raise ValueError('VM resources outside the local validation limits')
        self.base_image = Path(base_image).resolve() if base_image else ROOT / 'data/osworld-vm/Ubuntu.qcow2'
        self.prefix = Path(cfg["cache_dir"]) / "qemu-system/root"
        self.address, self.cpus, self.memory_mb = address, cpus, memory_mb
        self.id = uuid.uuid4().hex
        self.directory = ROOT / "runs/vms" / self.id
        self.directory.mkdir(parents=True)
        self.process = None

    def start(self):
        if self.process is not None:
            raise RuntimeError('Create a new LocalVM for each fresh overlay')
        base = self.base_image
        if not base.is_file():
            raise FileNotFoundError(base)
        if base == ROOT / 'data/osworld-vm/Ubuntu.qcow2':
            assert base.stat().st_size == 24460197888, "VM base image must finish extraction first"
        self.disk = self.directory / "overlay.qcow2"
        subprocess.run([*command("qemu-img"), "create", "-f", "qcow2", "-F", "qcow2", "-b", str(base), str(self.disk)], check=True)
        network = "user,id=n0" + "".join(f",hostfwd=tcp:{self.address}:{p}-:{p}" for p in [5000, 9222, 8080, 8000])
        accel_flags = ['-accel', 'kvm', '-cpu', 'host'] if self.accelerator == 'kvm' else ['-accel', 'tcg,thread=multi', '-cpu', 'max']
        args = [*command("qemu-system-x86_64"), "-name", "cua-rl-" + self.id,
                "-machine", "q35", *accel_flags,
                "-smp", str(self.cpus), "-m", str(self.memory_mb), "-L", str(self.prefix / "usr/share/qemu"),
                "-bios", str(self.prefix / "usr/share/seabios/bios-256k.bin"),
                "-drive", f"file={self.disk},if=virtio,format=qcow2", "-device", f"virtio-vga,romfile={self.prefix}/usr/share/seabios/vgabios-virtio.bin",
                "-netdev", network, "-device", "virtio-net-pci,netdev=n0,romfile=", "-device", "qemu-xhci", "-device", "usb-tablet",
                "-display", "none", "-vnc", f"{self.address}:0", "-monitor", f"unix:{self.directory}/monitor.sock,server,nowait",
                "-serial", f"file:{self.directory}/serial.log", "-no-reboot"]
        with (self.directory / "qemu.log").open("ab", buffering=0) as log:
            env = dict(os.environ, QEMU_MODULE_DIR=str(self.prefix / "usr/lib/x86_64-linux-gnu/qemu"))
            self.process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state = {"id": self.id, "pid": self.process.pid, "owner_pid": os.getpid(), "hostname": socket.gethostname(), "address": self.address,
                 "run_id": os.environ.get('CUA_VM_RUN_ID'),
                 "backend": 'QEMU KVM' if self.accelerator == 'kvm' else 'QEMU TCG (software CPU emulation)',
                 "accelerator": self.accelerator, "base_image": str(base),
                 "cpus": self.cpus, "memory_mb": self.memory_mb,
                 "command": args, "directory": str(self.directory)}
        (self.directory / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        return state

    def wait_ready(self, timeout=600):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError((self.directory / "qemu.log").read_text())
            try:
                with opener.open(f"http://{self.address}:5000/screenshot", timeout=5) as response:
                    screenshot = response.read()
                if screenshot.startswith(b"\x89PNG"):
                    from PIL import Image, ImageStat
                    screen = Image.open(io.BytesIO(screenshot)).convert('RGB')
                    if max(ImageStat.Stat(screen.resize((64, 36))).stddev) < 5:
                        time.sleep(3)
                        continue
                    (self.directory / "ready.png").write_bytes(screenshot)
                    return time.monotonic() - start
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError(f"VM not ready after {timeout}s; {self.directory}")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


if __name__ == "__main__":
    vm = LocalVM()
    state = vm.start()
    (ROOT / "runs/qemu-smoke-service.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
    try:
        seconds = vm.wait_ready()
        print(json.dumps({"ready": True, "seconds": seconds}), flush=True)
    except Exception:
        vm.stop()
        raise
