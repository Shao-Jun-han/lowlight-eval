"""用任意已配置方法执行单张推理，便于先确认模型加载与颜色。"""

import argparse
from time import perf_counter

import torch

from dataset import read_rgb, save_rgb, validate_rgb
from runtime import load_adapter, load_config, select_methods, setup, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-random", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup(config)
    method = select_methods(config, [args.method])[0]
    adapter, model = load_adapter(method, device, args.allow_random)
    image = read_rgb(args.input)
    with torch.no_grad():
        for _ in range(config["timing"]["warmup"]):
            adapter.enhance(model, image)
        synchronize(device)
        start = perf_counter()
        enhanced = adapter.enhance(model, image)
        synchronize(device)
        elapsed = perf_counter() - start
    validate_rgb(enhanced, "增强结果")
    if enhanced.shape != image.shape:
        raise ValueError("增强结果尺寸与输入不一致。")
    save_rgb(enhanced, args.output)
    mode = "RANDOM INIT / 非正式结果" if method.get("weight_path") is None else "PRETRAINED"
    print(f"{method['name']}: {elapsed:.4f} s; {device}; {mode}; saved: {args.output}")


if __name__ == "__main__":
    main()
