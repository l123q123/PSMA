# 测试 DaD 在 PASM 静态 Patch 预算下的选点规则。

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.dad import (
    DaDDetections,
    FrozenDaDDetector,
    _image_tensor,
    _normalized_to_pixels,
    select_global_dad_points,
    select_patch_budgeted_points,
)
from pasm_v0.loma_matcher import (
    FrozenLoMaMatcher,
    loma_b128_config,
    normalized_keypoints_to_pixels,
    pixel_keypoints_to_normalized,
)
from pasm_v0.sampling import PatchLayout


class DaDSelectionTests(unittest.TestCase):
    """验证 DaD 的全局和 Patch 预算选点不变量。"""

    def setUp(self) -> None:
        """创建四个 4×4 Patch 组成的最小图像网格。"""
        self.config = PasmV0Config(
            image_size=8,
            patch_rows=2,
            patch_cols=2,
            total_points=4,
            round_min_points=(1, 0, 0),
            max_points_per_patch=4,
        )
        self.layout = PatchLayout(self.config)

    def test_patch_budgeted_selection_prefers_nms_then_dense_fill(self) -> None:
        """每个 Patch 应先保留 NMS 峰，再以稠密分数补足自身预算。"""
        dense_scores = np.zeros((8, 8), dtype=np.float32)
        dense_scores[0, 0] = 0.95
        dense_scores[1, 6] = 0.90
        detections = DaDDetections(
            dense_scores=dense_scores,
            nms_points=np.array([[2.25, 2.5], [7.2, 7.3]], dtype=np.float32),
            nms_scores=np.array([0.10, 0.05], dtype=np.float32),
        )
        budgets = np.array([2, 1, 0, 1], dtype=np.int64)

        points, patch_ids, scores = select_patch_budgeted_points(detections, self.layout, budgets)

        self.assertEqual(points.shape, (4, 2))
        self.assertEqual(scores.shape, (4,))
        np.testing.assert_array_equal(np.bincount(patch_ids, minlength=4), budgets)
        np.testing.assert_array_equal(self.layout.patch_ids(points), patch_ids)
        self.assertTrue(any(np.allclose(point, [2.25, 2.5]) for point in points))
        self.assertTrue(any(np.allclose(point, [0.5, 0.5]) for point in points))
        self.assertTrue(any(np.allclose(point, [6.5, 1.5]) for point in points))
        self.assertTrue(any(np.allclose(point, [7.2, 7.3]) for point in points))
        self.assertEqual(len(np.unique(points, axis=0)), len(points))

    def test_global_selection_prefers_nms_then_dense_fill_without_duplicates(self) -> None:
        """全局 DaD 基线也应优先保留原生 NMS 峰，再以稠密分数补足。"""
        dense_scores = np.zeros((8, 8), dtype=np.float32)
        dense_scores[1, 3] = 0.9
        dense_scores[7, 0] = 0.8
        dense_scores[2, 2] = 0.7
        detections = DaDDetections(
            dense_scores=dense_scores,
            nms_points=np.array([[6.2, 6.3]], dtype=np.float32),
            nms_scores=np.array([0.01], dtype=np.float32),
        )

        points, scores = select_global_dad_points(detections, image_size=8, point_count=2)

        np.testing.assert_allclose(points, np.array([[6.2, 6.3], [3.5, 1.5]], dtype=np.float32))
        np.testing.assert_allclose(scores, np.array([0.01, 0.9], dtype=np.float32))
        self.assertEqual(len(np.unique(points, axis=0)), len(points))

    def test_frozen_dad_rejects_missing_local_weights(self) -> None:
        """冻结 DaD 必须在加载模型前拒绝不存在的本地权重。"""
        with self.assertRaises(FileNotFoundError):
            FrozenDaDDetector("missing_dad_weights.pth", image_size=518, device="cpu")

    def test_frozen_loma_rejects_missing_local_dino_weights(self) -> None:
        """完整 LoMa 应在构造前明确要求本地 DINOv2 权重。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loma_path = root / "loma.pth"
            dad_path = root / "dad.pth"
            loma_path.touch()
            dad_path.touch()
            with self.assertRaises(FileNotFoundError):
                FrozenLoMaMatcher(loma_path, dad_path, root / "missing_dino.pth", device="cpu")

    def test_loma_b128_config_matches_the_local_128_dimensional_checkpoint(self) -> None:
        """本地 loma_B128 权重必须使用 DeDoDe-B 的 128→256 配置。"""
        config = loma_b128_config()

        self.assertEqual(config.input_dim, 128)
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.descriptor, "dedode_b")
        self.assertEqual(config.n_layers, 9)

    def test_loma_pixel_and_normalized_coordinates_round_trip(self) -> None:
        """LoMa 的像素和归一化关键点坐标应能在 518 像素图像内往返转换。"""
        points = torch.tensor([[0.5, 0.5], [259.0, 259.0], [517.5, 517.5]], dtype=torch.float32)

        normalized = pixel_keypoints_to_normalized(points, image_size=518)
        restored = normalized_keypoints_to_pixels(normalized, image_size=518)

        np.testing.assert_allclose(restored.numpy(), points.numpy(), atol=1e-5)
        np.testing.assert_allclose(normalized[1].numpy(), [0.0, 0.0], atol=1e-6)

    def test_dad_image_and_coordinate_helpers_prepare_model_input(self) -> None:
        """DaD 适配器应将 CHW 图像加上批维并将归一化点转为像素坐标。"""
        image = _image_tensor(np.zeros((3, 8, 8), dtype=np.float32), image_size=8, device=torch.device("cpu"))
        pixels = _normalized_to_pixels(torch.tensor([[0.0, 0.0]]), image_size=8)

        self.assertEqual(tuple(image.shape), (1, 3, 8, 8))
        np.testing.assert_allclose(pixels, np.array([[4.0, 4.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
