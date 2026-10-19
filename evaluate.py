"""batch_size=1 的统一推理评测，检查失败时保留已有结果并停止后续方法。"""

import argparse
import csv
import gc
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dataset import list_pairs, read_rgb, save_rgb, validate_rgb
from metrics import Metrics
from runtime import (file_sha256, load_adapter, load_config, select_methods,
                     setup, synchronize, write_json)

_FIELDS = ["dataset", "method", "weights", "row_type", "image", "psnr", "ssim", "lpips", "seconds"]
_SCORES = ["psnr", "ssim", "lpips", "seconds"]


def _csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _format(value):
    return "inf" if math.isinf(value) else f"{value:.6f}"


def _save_tables(run_dir, rows, summaries):
    _csv_write(run_dir / "results.csv", rows)
    _csv_write(run_dir / "summary.csv", summaries)
    lines = ["| Method | Weights | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Seconds/image ↓ |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        scores = " | ".join(_format(row[key]) for key in _SCORES)
        lines.append(f"| {row['method']} | {row['weights']} | {scores} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanity_check(method, rows, config):
    """保留任务包区间判据；只报告疑点，不据此修改图像或调整分数。"""
    if not config["sanity"]["enabled"]:
        return {"status": "disabled", "failures": []}
    if method.get("weight_path") is None:
        return {"status": "not_applicable_random_init", "failures": []}
    expected = method.get("sanity")
    if not expected:
        return {"status": "no_reference", "failures": []}
    failures = []
    for key in ("psnr", "ssim"):
        if key not in expected:
            continue
        actual = float(np.mean([row[key] for row in rows]))
        lower, upper = expected[key]
        if not lower <= actual <= upper:
            failures.append(f"平均 {key}={actual:.6f}，超出 [{lower}, {upper}]")
    maximum = max(row["seconds"] for row in rows)
    if "max_seconds" in expected and maximum > expected["max_seconds"]:
        failures.append(f"最慢单张 {maximum:.3f}s，超过 {expected['max_seconds']}s")
    total = sum(row["seconds"] for row in rows)
    limit = config["timing"].get("max_total_seconds")
    if limit is not None and total > limit:
        failures.append(f"累计推理 {total:.3f}s，超过 {limit}s")
    return {"status": "failed" if failures else "passed", "failures": failures,
            "reference": expected, "max_seconds": maximum, "inference_seconds": total}


def _repo_revision(path):
    if path is None:
        return None
    result = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def evaluate(config, selected, allow_random=False):
    """执行配置指定的方法；返回运行目录和是否全部通过启用的 sanity 检查。"""
    device = setup(config)
    data = config["dataset"]
    pairs = list_pairs(data["low_dir"], data["high_dir"], data["expected_pairs"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(config["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata = {
        "run_id_utc": run_id, "status": "running", "config": config,
        "device": str(device), "python": sys.version, "platform": platform.platform(),
        "versions": {key: version(key) for key in ("torch", "torchvision", "numpy", "scikit-image", "lpips")},
        "protocol": {"color": "RGB", "range": [0, 1], "dtype": "float32",
                     "crop_border": 0, "gt_mean_correction": False, "batch_size": 1,
                     "ssim": "RGB, Gaussian sigma=1.5, win=11, population covariance",
                     "timing": "enhance including device transfers; excludes load/metrics/save; warmup excluded",
                     "quantize_8bit": config["metrics"].get("quantize_8bit", False)},
        "pairs": [{"image": name, "low": str(low), "high": str(high),
                   "low_sha256": file_sha256(low), "high_sha256": file_sha256(high)}
                  for name, low, high in pairs],
        "requested_methods": [method["name"] for method in selected], "methods": [],
    }
    rows, summaries = [], []
    write_json(run_dir / "run.json", metadata)
    print(f"运行目录：{run_dir}", flush=True)
    try:
        metrics = Metrics(config["metrics"]["lpips_net"], config["metrics"]["lpips_device"])
        for method in selected:
            name = method["name"]
            weight_mode = "random_init" if method.get("weight_path") is None else "pretrained"
            if weight_mode == "random_init":
                print(f"注意：{name} 使用随机初始化，仅用于流程验证。", flush=True)
            adapter, model = load_adapter(method, device, allow_random)
            first = read_rgb(pairs[0][1])
            with torch.no_grad():
                for _ in range(config["timing"]["warmup"]):
                    adapter.enhance(model, first)
                synchronize(device)
                per_method = []
                for image_id, low, high in pairs:
                    image, target = read_rgb(low), read_rgb(high)
                    synchronize(device)
                    start = perf_counter()
                    prediction = adapter.enhance(model, image)
                    synchronize(device)
                    elapsed = perf_counter() - start
                    validate_rgb(prediction, f"{name} 输出")
                    if prediction.shape != image.shape:
                        raise ValueError(f"{name} 改变了图像尺寸：{prediction.shape} / {image.shape}")
                    if config["metrics"].get("quantize_8bit", False):
                        prediction = np.rint(prediction * 255).astype(np.float32) / np.float32(255)
                    scores = metrics(prediction, target)
                    row = {"dataset": data["name"], "method": name, "weights": weight_mode,
                           "row_type": "image", "image": image_id, **scores, "seconds": elapsed}
                    per_method.append(row)
                    rows.append(row)
                    save_rgb(prediction, run_dir / "images" / name / f"{image_id}.png")
                    np.save(run_dir / "images" / name / f"{image_id}.npy", prediction)
                    print(f"{name}/{image_id}: PSNR={scores['psnr']:.4f}, SSIM={scores['ssim']:.4f}, "
                          f"LPIPS={scores['lpips']:.4f}, {elapsed:.4f}s", flush=True)
                mean_row = {"dataset": data["name"], "method": name, "weights": weight_mode,
                            "row_type": "mean", "image": "__mean__",
                            **{key: float(np.mean([row[key] for row in per_method])) for key in _SCORES}}
                rows.append(mean_row)
                summaries.append(mean_row)
                sanity = _sanity_check(method, per_method, config)
                metadata["methods"].append({"name": name, "weights": weight_mode,
                    "weight_path": method.get("weight_path"),
                    "weight_sha256": file_sha256(method["weight_path"]) if method.get("weight_path") else None,
                    "repo_revision": _repo_revision(method.get("repo_path")), "sanity": sanity})
                _save_tables(run_dir, rows, summaries)
            del model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            if sanity["status"] == "failed":
                metadata["status"] = "sanity_failed"
                metadata["guidance"] = "检查 RGB/BGR、数值范围、测试划分，再核对权重与评测口径（包括 GT 均值校正）。"
                for failure in sanity["failures"]:
                    print(f"Sanity Check 未通过：{failure}", flush=True)
                print(metadata["guidance"] + " 已停止后续方法。", flush=True)
                return run_dir, False
        metadata["status"] = "completed" if config["sanity"]["enabled"] and all(
            item["sanity"]["status"] == "passed" for item in metadata["methods"]
        ) else "completed_unvalidated"
        return run_dir, True
    except Exception as error:
        metadata["status"] = "error"
        metadata["error"] = f"{type(error).__name__}: {error}"
        _save_tables(run_dir, rows, summaries)
        raise
    finally:
        metadata["wall_seconds"] = perf_counter() - started
        write_json(run_dir / "run.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--allow-random", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    selected = select_methods(config, args.methods)
    run_dir, passed = evaluate(config, selected, args.allow_random)
    print(f"结果已保存：{run_dir}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
