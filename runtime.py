"""配置、设备和动态导入的薄工具层；评测流程不依赖任何模型结构。"""

import hashlib
import importlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def resolve_path(base, value):
    """解析配置内的相对路径，并支持用户目录 ~。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def load_config(path):
    """读取 YAML，把所有路径变成绝对路径，并检查必要字段。"""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = path.parent
    for key in ("low_dir", "high_dir"):
        config["dataset"][key] = str(resolve_path(base, config["dataset"][key]))
    for key in ("output_dir", "cache_dir"):
        config[key] = str(resolve_path(base, config[key]))
    names = set()
    for method in config["methods"]:
        name = method["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"方法名须唯一且只含字母、数字、下划线、连字符：{name}")
        names.add(name)
        for key in ("repo_path", "weight_path"):
            if method.get(key) is not None:
                method[key] = str(resolve_path(base, method[key]))
    config["config_path"] = str(path)
    return config


def setup(config):
    """固定种子和 CPU 线程数，选择 MPS/CPU；不静默切换运行设备。"""
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.hub.set_dir(config["cache_dir"])
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in {"mps", "cpu"}:
        raise ValueError("本项目 device 支持 auto / mps / cpu。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("配置要求 MPS，但当前进程不可用。可改为 device: cpu。")
    return torch.device(requested)


def select_methods(config, requested=None):
    """按配置顺序选择方法，未知名称或空列表均报错。"""
    available = {item["name"] for item in config["methods"]}
    if requested and set(requested) - available:
        raise ValueError(f"未知方法：{sorted(set(requested) - available)}")
    selected = [item for item in config["methods"] if
                (item["name"] in requested if requested else item.get("enabled", True))]
    if not selected:
        raise ValueError("没有选择任何方法。")
    return selected


def load_adapter(method, device, allow_random=False):
    """准备第三方导入路径后，只调用 adapter 的标准 load_model 接口。"""
    if method.get("repo_path"):
        repo = Path(method["repo_path"])
        if not repo.is_dir():
            raise FileNotFoundError(f"官方代码目录不存在：{repo}；请先运行 scripts/prepare_models.py。")
        sys.path.insert(0, str(repo))
    weight = method.get("weight_path")
    if weight is None and not allow_random:
        raise ValueError(f"{method['name']} 未指定权重；随机初始化只允许显式 --allow-random。")
    if weight is not None and not Path(weight).is_file():
        raise FileNotFoundError(f"权重文件不存在：{weight}。不会静默使用随机权重。")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # 官方源码目录保持干净，不写入 __pycache__。
    try:
        adapter = importlib.import_module(method["module"])
        return adapter, adapter.load_model(weight, device)
    finally:
        sys.dont_write_bytecode = previous


def synchronize(device):
    """等待异步 MPS 算子完成，使计时包含真实推理时间。"""
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def file_sha256(path):
    """分块计算文件指纹，用于追溯数据和模型权重。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """输出可阅读的 UTF-8 JSON；调用方负责把非有限数值转成字符串。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
