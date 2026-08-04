# 测试 DaD 在 PASM 静态 Patch 预算下的选点规则。

import unittest

import numpy as np

from pasm_v0.config import PasmV0Config
from pasm_v0.dad import DaDDetections, select_global_dad_points, select_patch_budgeted_points
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


if __name__ == "__main__":
    unittest.main()
