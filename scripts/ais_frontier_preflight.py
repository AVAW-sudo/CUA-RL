"""Cold-compile and verify the vLLM vision rotary kernel on the assigned A100."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def main():
    # The job image may override Triton's own assembler before Python starts.
    spec = importlib.util.find_spec('triton')
    if spec is None or not spec.origin:
        raise RuntimeError('The policy environment is missing Triton')
    bundled = Path(spec.origin).resolve().parent / 'backends/nvidia/bin/ptxas'
    if not os.access(bundled, os.X_OK):
        raise RuntimeError(f'The policy runtime is missing its bundled ptxas: {bundled}')
    os.environ['TRITON_PTXAS_PATH'] = str(bundled)

    # Exercise compilation even when the node has already cached this kernel.
    with tempfile.TemporaryDirectory(prefix='cua-triton-check-') as cold_cache:
        os.environ['TRITON_CACHE_DIR'] = cold_cache
        import torch
        import triton
        from triton import knobs
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        # Both the Frontier and training Triton versions select this tool for A100.
        tool = knobs.nvidia.ptxas
        if Path(tool.path).resolve() != bundled:
            raise RuntimeError(f'Triton selected {tool.path}, expected {bundled}')
        device = torch.cuda.current_device()
        values = torch.arange(256, device=device, dtype=torch.float32)
        total = values.sum().item()
        if total != 32640:
            raise RuntimeError(f'Unexpected GPU sum {total}')
        x = values.reshape(1, 4, 2, 32)
        sin = torch.ones((4, 16), device=device)
        rotated = apply_rotary_emb(x, torch.zeros_like(sin), sin)
        left, right = x.chunk(2, dim=-1)
        torch.testing.assert_close(rotated, torch.cat((-right, left), dim=-1), rtol=0, atol=0)

    maps = Path('/proc/self/maps').read_text().splitlines()
    drivers = sorted({line.split()[-1] for line in maps if '/libcuda.so' in line})
    print(json.dumps({
        'device': torch.cuda.get_device_name(device), 'cuda_build': torch.version.cuda,
        'driver_libraries': drivers, 'total': total, 'triton_version': triton.__version__,
        'ptxas_path': tool.path, 'ptxas_version': tool.version, 'vision_rotary_cold_compile': 'passed',
    }))


if __name__ == '__main__':
    main()
