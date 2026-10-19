"""创建明确标注的合成测试集；只能验证软件流程，不能用于论文性能结论。"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import save_rgb
from runtime import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-hvi", action="store_true")
    args = parser.parse_args()
    data_root = ROOT / "artifacts" / "smoke_data"
    # 两张非 8 倍数尺寸的图，同时覆盖反射填充和裁回原尺寸的适配逻辑。
    for index, (height, width) in enumerate(((65, 73), (79, 97))):
        y, x = np.mgrid[0:height, 0:width]
        target = np.stack([x / width, y / height, ((x // 9 + y // 9) % 2) * 0.7 + 0.1], axis=-1)
        target = (0.05 + 0.85 * target).astype(np.float32)
        low = (target * (0.12 + 0.06 * index)).astype(np.float32)
        save_rgb(low, data_root / "low" / f"synthetic_{index}.png")
        save_rgb(target, data_root / "high" / f"synthetic_{index}.png")
    config = load_config(ROOT / "config.yaml")
    config.pop("config_path")
    config["dataset"] = {"name": "SYNTHETIC_SMOKE_TEST", "low_dir": str(data_root / "low"),
                         "high_dir": str(data_root / "high"), "expected_pairs": 2}
    config["output_dir"] = str(ROOT / "artifacts" / "smoke_runs")
    config["sanity"]["enabled"] = False
    if args.random_hvi:
        config["methods"][1]["weight_path"] = None
    target = ROOT / "artifacts" / ("smoke_random.yaml" if args.random_hvi else "smoke.yaml")
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"合成流程测试配置：{target}")
    print("此数据不是 LOLv1；产物不得用作真实测试集评测或验收。")


if __name__ == "__main__":
    main()
