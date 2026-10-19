"""严格读取一对一的 low/high 图像，避免把错配数据送入评测。"""

from pathlib import Path

import numpy as np
from PIL import Image

_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_rgb(path):
    """把图像读为 H×W×3、RGB、float32、[0, 1]；不缩放或裁剪。"""
    with Image.open(path) as image:
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            raise ValueError(f"不支持 {image.mode} 图像：{path}；请先明确位深转换规则。")
        return np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)


def validate_rgb(image, label="图像"):
    """检查适配器约定，不静默修正 BGR、位深、范围或尺寸错误。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} 必须为 H×W×3 numpy 数组。")
    if image.dtype != np.float32:
        raise ValueError(f"{label} 必须为 float32，实际是 {image.dtype}。")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f"{label} 必须为有限数值，且范围为 [0, 1]。")


def save_rgb(image, path):
    """把有效 RGB 浮点图四舍五入为 8 位 PNG，并自动创建输出目录。"""
    validate_rgb(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def _index_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{directory}。请修改 config.yaml 中的数据路径。")
    indexed = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _EXTENSIONS:
            if path.stem in indexed:
                raise ValueError(f"同一目录存在重复图像编号 {path.stem}：{directory}")
            indexed[path.stem] = path
    return indexed


def list_pairs(low_dir, high_dir, expected_pairs=15):
    """按文件名主干配对；漏图、重复编号、数量不符或大小不符时立即报错。"""
    low = _index_images(low_dir)
    high = _index_images(high_dir)
    if low.keys() != high.keys():
        raise ValueError(
            f"low/high 不匹配：缺少 high={sorted(low.keys() - high.keys())}；"
            f"缺少 low={sorted(high.keys() - low.keys())}"
        )
    if not low:
        raise ValueError("数据集为空。")
    if expected_pairs is not None and len(low) != expected_pairs:
        raise ValueError(f"期望 {expected_pairs} 对测试图，实际 {len(low)} 对。请检查是否误用训练集。")
    pairs = []
    for name in sorted(low):
        with Image.open(low[name]) as a, Image.open(high[name]) as b:
            if a.size != b.size:
                raise ValueError(f"{name} 的 low/high 尺寸不一致：{a.size} / {b.size}")
        pairs.append((name, low[name], high[name]))
    return pairs
