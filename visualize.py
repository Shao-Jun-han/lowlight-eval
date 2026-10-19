"""从一份运行记录生成对比拼图，不重新推理，也不改变原始图像亮度。"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def make_comparison(run_dir, output=None, image_ids=None, panel_width=400):
    """每行一张样本、每列一种方法；标注指标、随机权重及 sanity 状态。"""
    run_dir = Path(run_dir).expanduser().resolve()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    methods = metadata["methods"]
    if not methods:
        raise ValueError("该运行没有已完成的方法，无法生成对比图。")
    pairs = {item["image"]: item for item in metadata["pairs"]}
    chosen = image_ids or list(pairs)[:3]
    if not chosen or set(chosen) - pairs.keys():
        raise ValueError("所选图像编号不在本次运行中。")
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
        scores = {(row["method"], row["image"]): row for row in csv.DictReader(handle)
                  if row["row_type"] == "image"}
    gap, top, header, footer = 16, 78, 44, 40
    panel_height = round(panel_width * 0.75)
    row_height = header + panel_height + footer + gap
    columns = len(methods) + 2
    canvas = Image.new("RGB", (gap + columns * (panel_width + gap), top + row_height * len(chosen)), "#eef1f6")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = _font(26), _font(20), _font(15)
    title = f"{metadata['config']['dataset']['name']} | {metadata['device'].upper()} | {metadata['status']}"
    draw.text((gap, 12), title, fill="#102037", font=title_font)
    draw.text((gap, 48), "RGB | No GT brightness correction | Scores use recorded evaluation protocol",
              fill="#44516a", font=small_font)
    for row_index, image_id in enumerate(chosen):
        pair = pairs[image_id]
        items = [("Input", Path(pair["low"]), None)]
        for method in methods:
            label = method["name"] + (" [RANDOM]" if method["weights"] == "random_init" else "")
            items.append((label, run_dir / "images" / method["name"] / f"{image_id}.png",
                          scores[(method["name"], image_id)]))
        items.append(("Ground truth", Path(pair["high"]), None))
        y = top + row_index * row_height
        for column, (label, path, score) in enumerate(items):
            x = gap + column * (panel_width + gap)
            draw.rounded_rectangle((x, y, x + panel_width, y + row_height - gap), radius=9, fill="white")
            draw.text((x + 10, y + 10), label, fill="#102037", font=label_font)
            with Image.open(path) as source:
                fitted = ImageOps.contain(source.convert("RGB"), (panel_width, panel_height), Image.Resampling.LANCZOS)
                canvas.paste(fitted, (x + (panel_width - fitted.width) // 2,
                                    y + header + (panel_height - fitted.height) // 2))
            caption = f"ID {image_id}" if score is None else (
                f"P {float(score['psnr']):.2f} | S {float(score['ssim']):.3f} | L {float(score['lpips']):.3f}")
            draw.text((x + 10, y + header + panel_height + 10), caption, fill="#44516a", font=small_font)
    output = Path(output).expanduser().resolve() if output else run_dir / "comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"对比拼图：{output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--panel-width", type=int, default=400)
    args = parser.parse_args()
    if args.panel_width < 200:
        parser.error("--panel-width 至少为 200")
    make_comparison(args.run, args.output, args.images, args.panel_width)
