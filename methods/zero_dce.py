"""Zero-DCE 官方 DCE-Net 的推理适配器，不改写网络和迭代公式。"""

import importlib.util as _importlib

import numpy as _np
import torch as _torch

from dataset import validate_rgb as _validate_rgb
from methods._weights import read_state_dict as _read_state_dict

__all__ = ["load_model", "enhance"]


def load_model(weight_path, device):
    """从配置提供的官方目录加载 model.py，并严格匹配官方预训练权重。"""
    if weight_path is None:
        raise ValueError("Zero-DCE 必须使用有效预训练权重。")
    spec = _importlib.find_spec("model")
    if spec is None:
        raise ImportError("找不到 Zero-DCE 的 model.py，请检查 repo_path。")
    # 用独立名称读取，避免把官方通用名称 model 留在全局模块缓存。
    local_spec = _importlib.spec_from_file_location("_official_zero_dce_model", spec.origin)
    official = _importlib.module_from_spec(local_spec)
    local_spec.loader.exec_module(official)
    model = official.enhance_net_nopool()
    model.load_state_dict(_read_state_dict(weight_path), strict=True)
    return model.to(device).eval().requires_grad_(False)


@_torch.no_grad()
def enhance(model, img):
    """RGB float32 [0,1] 输入，返回第 8 次迭代的同尺寸增强结果。"""
    _validate_rgb(img, "Zero-DCE 输入")
    tensor = _torch.from_numpy(_np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(next(model.parameters()).device)
    _, output, _ = model(tensor)
    return output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(_np.float32)
