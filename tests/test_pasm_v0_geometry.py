# 测试 PASM v0.1 的直接几何投影监督。

import unittest

import numpy as np
import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.geometry import (
    PairGeometry,
    projected_match_targets,
    projected_patch_support_targets,
    visible_projected_points,
)
from pasm_v0.sampling import PatchLayout


class PasmV0GeometryTests(unittest.TestCase):
    """验证直接投影、遮挡过滤和 Patch 支持目标。"""

    def setUp(self) -> None:
        """创建与投影坐标一致的小尺寸 Patch 网格。"""
        self.config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
        )

    def test_identity_geometry_labels_mutual_nearest_sampled_points(self) -> None:
        """相同相机下的双向最近点应成为可靠正样本。"""
        geometry = self._identity_geometry()
        points_a = np.array([[4.0, 4.0], [20.0, 20.0]], dtype=np.float32)
        points_b = np.array([[4.5, 4.0], [20.0, 20.5], [30.0, 30.0]], dtype=np.float32)

        targets_a, valid_a, targets_b, valid_b = projected_match_targets(
            points_a, points_b, geometry, threshold_px=1.0, depth_tolerance=0.01
        )

        torch.testing.assert_close(targets_a, torch.tensor([0, 1]))
        torch.testing.assert_close(valid_a, torch.tensor([True, True]))
        torch.testing.assert_close(targets_b, torch.tensor([0, 1, -1]))
        torch.testing.assert_close(valid_b, torch.tensor([True, True, False]))

    def test_occluded_projection_does_not_create_positive_target(self) -> None:
        """投影深度与目标深度不一致时不能产生正样本。"""
        geometry = self._identity_geometry(depth_b=1.0)

        targets_a, valid_a, _, _ = projected_match_targets(
            np.array([[8.0, 8.0]], dtype=np.float32),
            np.array([[8.0, 8.0]], dtype=np.float32),
            geometry,
            threshold_px=1.0,
            depth_tolerance=0.01,
        )

        torch.testing.assert_close(targets_a, torch.tensor([-1]))
        torch.testing.assert_close(valid_a, torch.tensor([False]))

    def test_projection_support_counts_visible_patch_coverage(self) -> None:
        """固定投影网格应为两图生成归一化的 Patch 支持。"""
        support_a, support_b = projected_patch_support_targets(
            self._identity_geometry(), PatchLayout(self.config), stride=4, depth_tolerance=0.01
        )

        self.assertEqual(support_a.shape, (16,))
        self.assertEqual(support_b.shape, (16,))
        self.assertAlmostEqual(float(support_a.sum()), 1.0)
        self.assertAlmostEqual(float(support_b.sum()), 1.0)
        np.testing.assert_allclose(support_a, support_b)

    def test_visible_projection_returns_source_indices_and_coordinates(self) -> None:
        """公开投影接口应同时返回可见源点索引与目标浮点坐标。"""
        indices, projected = visible_projected_points(
            np.array([[4.0, 4.0], [40.0, 4.0]], dtype=np.float32),
            self._identity_geometry(),
            "a_to_b",
            depth_tolerance=0.01,
        )

        np.testing.assert_array_equal(indices, np.array([0], dtype=np.int64))
        np.testing.assert_allclose(projected, np.array([[4.0, 4.0]], dtype=np.float32))

    def _identity_geometry(self, depth_b: float = 2.0) -> PairGeometry:
        """构造深度一致的单位相机几何信息。"""
        depth_a = np.full((32, 32), 2.0, dtype=np.float32)
        target_depth = np.full((32, 32), depth_b, dtype=np.float32)
        return PairGeometry(
            depth_a=depth_a,
            depth_b=target_depth,
            intrinsics_a=np.eye(3, dtype=np.float64),
            intrinsics_b=np.eye(3, dtype=np.float64),
            cam2world_a=np.eye(4, dtype=np.float64),
            cam2world_b=np.eye(4, dtype=np.float64),
            original_size_a=(32, 32),
            original_size_b=(32, 32),
            image_size=32,
        )


if __name__ == "__main__":
    unittest.main()
