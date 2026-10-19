"""所有方法共用的 RGB 全图指标；默认不裁边、不做 GT 均值校正。"""

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset import validate_rgb


def psnr_ssim(prediction, target):
    """在 [0,1] RGB 上算 PSNR 和 11×11 高斯 SSIM，返回 Python float。"""
    validate_rgb(prediction, "预测图")
    validate_rgb(target, "GT")
    if prediction.shape != target.shape:
        raise ValueError(f"预测图与 GT 尺寸不同：{prediction.shape} / {target.shape}")
    if min(prediction.shape[:2]) < 11:
        raise ValueError("SSIM 使用 11×11 窗口，输入高和宽至少为 11。")
    a, b = prediction.astype(np.float64), target.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = math.inf if mse == 0 else -10 * math.log10(mse)
    ssim = structural_similarity(
        a, b, data_range=1.0, channel_axis=2, gaussian_weights=True,
        sigma=1.5, win_size=11, use_sample_covariance=False,
    )
    return psnr, float(ssim)


class Metrics:
    """只创建一次预训练 LPIPS 网络，逐张计算，不以随机网络冒充正式指标。"""

    def __init__(self, net="alex", device="cpu"):
        import lpips

        self.device = torch.device(device)
        self.network = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
        self.network.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prediction, target):
        """返回 PSNR / SSIM / LPIPS；LPIPS 输入线性映射到 [-1,1]。"""
        psnr, ssim = psnr_ssim(prediction, target)
        if min(prediction.shape[:2]) < 64:
            raise ValueError("为保证 LPIPS 骨干可运行，评测图像高和宽至少为 64。")
        tensors = [torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
                   .unsqueeze(0).to(self.device) * 2 - 1 for image in (prediction, target)]
        distance = float(self.network(*tensors).item())
        if not math.isfinite(distance):
            raise ValueError("LPIPS 输出非有限值。")
        return {"psnr": psnr, "ssim": ssim, "lpips": distance}


def self_test(include_lpips=False):
    """验证恒等图像与已知误差；可额外验证真实预训练 LPIPS。"""
    rng = np.random.default_rng(42)
    image = rng.random((64, 64, 3), dtype=np.float32)
    psnr, ssim = psnr_ssim(image, image)
    assert math.isinf(psnr) and abs(ssim - 1) < 1e-12
    black = np.zeros_like(image)
    gray = np.full_like(image, 0.1)
    known, _ = psnr_ssim(black, gray)
    assert abs(known - 20) < 1e-5
    print(f"指标自测通过：恒等 PSNR={psnr}，SSIM={ssim}；已知误差 PSNR={known:.6f} dB")
    if include_lpips:
        values = Metrics()(image, image)
        assert abs(values["lpips"]) < 1e-6
        print(f"预训练 AlexNet LPIPS 恒等自测通过：{values['lpips']}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".cache" / "torch"))
    parser = argparse.ArgumentParser(description="指标自测")
    parser.add_argument("--lpips", action="store_true", help="包含预训练 LPIPS，首次需下载骨干权重")
    self_test(parser.parse_args().lpips)
