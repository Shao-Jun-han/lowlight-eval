"""HVI-CIDNet 官方模型的 LOLv1 推理适配器。"""

import numpy as _np
import torch as _torch
import torch.nn.functional as _F

from dataset import validate_rgb as _validate_rgb
from methods._weights import read_state_dict as _read_state_dict

__all__ = ["load_model", "enhance"]


def load_model(weight_path, device):
    """严格加载 .pth/.safetensors；None 仅供显式授权的随机权重流程测试。"""
    from net.CIDNet import CIDNet

    model = CIDNet()
    if weight_path is not None:
        if str(weight_path).lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(str(weight_path), device="cpu")
        else:
            state_dict = _read_state_dict(weight_path)
        model.load_state_dict(state_dict, strict=True)
    # 对齐官方 eval.py --lol：饱和度门控开启，alpha_s 使用官方默认 1.3。
    model.trans.gated = True
    model.trans.gated2 = False
    return model.to(device).eval().requires_grad_(False)


@_torch.no_grad()
def enhance(model, img):
    """只向右/下反射填充到 8 的倍数，推理后裁回原尺寸，不使用 GT。"""
    _validate_rgb(img, "HVI-CIDNet 输入")
    height, width = img.shape[:2]
    if min(height, width) < 8:
        raise ValueError("HVI-CIDNet 输入的高和宽至少为 8。")
    tensor = _torch.from_numpy(_np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(next(model.parameters()).device)
    tensor = _F.pad(tensor, (0, (-width) % 8, 0, (-height) % 8), mode="reflect")
    output = model(tensor)[:, :, :height, :width]
    return output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(_np.float32)
