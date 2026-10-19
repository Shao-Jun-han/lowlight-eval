"""打印运行环境，并可严格列出配置中的测试图像对。"""

import argparse
import platform
import sys

import torch

from dataset import list_pairs
from runtime import load_config, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()} ({platform.machine()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}; available: {torch.cuda.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}; available: {torch.backends.mps.is_available()}")
    print(f"Selected device: {device}")
    if args.check_data:
        data = config["dataset"]
        pairs = list_pairs(data["low_dir"], data["high_dir"], data["expected_pairs"])
        for index, (name, low, high) in enumerate(pairs, 1):
            print(f"{index:02d}. {name}: {low} | {high}")
        print(f"配对检查通过：{len(pairs)} 对。")


if __name__ == "__main__":
    main()
