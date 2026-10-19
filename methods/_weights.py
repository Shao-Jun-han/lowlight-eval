"""复用权重读取，不复制任何官方网络源码。"""

from pathlib import Path

import torch


def read_state_dict(path):
    """只读张量权重，支持 safetensors、常见 checkpoint 容器与 DataParallel 前缀。"""
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(path), device="cpu")
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("state_dict", "params_ema", "params", "model"):
            if isinstance(weights, dict) and key in weights and isinstance(weights[key], dict):
                weights = weights[key]
                break
    return {key.removeprefix("module."): value for key, value in weights.items()}
