"""Explicit identities for SFT, released RL, and a locally trained checkpoint."""
import json
from pathlib import Path

CHECKPOINTS = ['osworld-sft', 'scienceboard-sft', 'osworld-rl-reference', 'reproduced-rl']


def checkpoint_spec(root, name):
    cfg = json.loads((root / 'configs/paths.json').read_text())
    lock = json.loads((root / 'configs/sources.lock.json').read_text())
    if name == 'reproduced-rl':
        return json.loads((root / 'runs/reproduced-rl-model.json').read_text())
    key, folder, served, port, state, manifest = {
        'osworld-sft': ('sft_model', 'THUDM-ScaleCUA-qwen3-vl-osworld-sft', 'scalecua-qwen3-vl-8b-sft', 18080, 'inference-service.json', 'model_manifest.json'),
        'scienceboard-sft': ('scienceboard_sft_model', 'THUDM-ScaleCUA-qwen3-vl-scienceboard-sft', 'scalecua-qwen3-vl-8b-scienceboard-sft', 18081, 'scienceboard-inference-service.json', 'scienceboard-sft-model-manifest.json'),
        'osworld-rl-reference': ('released_rl_model', 'THUDM-ScaleCUA-qwen3-vl-osworld-rl-reference', 'scalecua-qwen3-vl-8b-rl-reference', 18100, 'released-rl-inference-service.json', 'released-rl-model-manifest.json'),
    }[name]
    source = lock[key]
    return {'model_id': source['repo_id'], 'model_revision': source['revision'],
            'model_dir': str(Path(cfg['model_dir']).with_name(folder)),
            'served_model_name': served, 'port': port, 'state_file': state, 'manifest_file': manifest}
