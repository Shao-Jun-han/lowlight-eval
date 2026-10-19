"""检查会使论文评测失真的输入、配对、数值与验收边界。"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import list_pairs, read_rgb, save_rgb, validate_rgb
from evaluate import _sanity_check
from metrics import psnr_ssim
from runtime import load_adapter


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).random((64, 72, 3), dtype=np.float32)

    def test_identity_and_known_psnr(self):
        psnr, ssim = psnr_ssim(self.image, self.image)
        self.assertTrue(math.isinf(psnr))
        self.assertEqual(ssim, 1.0)
        black = np.zeros_like(self.image)
        gray = np.full_like(self.image, 0.1)
        self.assertAlmostEqual(psnr_ssim(black, gray)[0], 20.0, places=5)

    def test_rgb_round_trip_preserves_red_and_blue(self):
        red = np.zeros_like(self.image)
        red[:, :, 0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/red.png"
            save_rgb(red, path)
            np.testing.assert_array_equal(read_rgb(path), red)

    def test_wrong_dtype_range_and_nan_are_rejected(self):
        for image in (self.image.astype(np.uint8), self.image * 255,
                      np.full_like(self.image, np.nan)):
            with self.subTest(dtype=image.dtype):
                with self.assertRaises(ValueError):
                    validate_rgb(image)

    def test_unmatched_names_and_training_split_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_rgb(self.image, root / "low/a.png")
            save_rgb(self.image, root / "high/b.png")
            with self.assertRaises(ValueError):
                list_pairs(root / "low", root / "high")
            (root / "high/b.png").rename(root / "high/a.png")
            with self.assertRaises(ValueError):
                list_pairs(root / "low", root / "high", expected_pairs=15)
            self.assertEqual(len(list_pairs(root / "low", root / "high", 1)), 1)

    def test_duplicate_stems_and_different_sizes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_rgb(self.image, root / "low/a.png")
            save_rgb(self.image, root / "low/a.bmp")
            save_rgb(self.image, root / "high/a.png")
            with self.assertRaises(ValueError):
                list_pairs(root / "low", root / "high", 1)
            (root / "low/a.bmp").unlink()
            save_rgb(self.image[:30], root / "high/a.png")
            with self.assertRaises(ValueError):
                list_pairs(root / "low", root / "high", 1)

    def test_missing_weight_never_falls_back_to_random(self):
        with self.assertRaises(FileNotFoundError):
            load_adapter({"name": "hvi", "module": "methods.hvi_cidnet",
                          "weight_path": "/nonexistent/checkpoint.pth"}, "cpu", allow_random=True)
        with self.assertRaises(ValueError):
            load_adapter({"name": "hvi", "module": "methods.hvi_cidnet", "weight_path": None}, "cpu")

    def test_out_of_range_sanity_and_random_results_cannot_pass(self):
        method = {"weight_path": "test.pth", "sanity": {"psnr": [13, 17], "ssim": [0.5, 0.65], "max_seconds": 1}}
        config = {"sanity": {"enabled": True}, "timing": {"max_total_seconds": 300}}
        rows = [{"psnr": 40, "ssim": 0.99, "seconds": 0.1}]
        self.assertEqual(_sanity_check(method, rows, config)["status"], "failed")
        method["weight_path"] = None
        self.assertEqual(_sanity_check(method, rows, config)["status"], "not_applicable_random_init")


if __name__ == "__main__":
    unittest.main()
