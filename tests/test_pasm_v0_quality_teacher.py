# 测试软质量教师的候选格、Patch 内选点和几何质量指标。

import unittest

import numpy as np

from pasm_v0.config import PasmV0Config
from pasm_v0.geometry import PairGeometry
from pasm_v0.quality_teacher import (
    cosine_mutual_matches,
    geometry_coverage_rate,
    mutual_geometry_inlier_rate,
    patch_candidate_points,
    sampled_descriptors,
    select_patch_teacher_points,
    teacher_scores,
)
from pasm_v0.sampling import PatchLayout


class PasmV0QualityTeacherTests(unittest.TestCase):
    """验证软质量教师的独立基础逻辑。"""

    def setUp(self) -> None:
        """创建用于精确断言的小型管理 Patch 网格。"""
        self.config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
        )
        self.layout = PatchLayout(self.config)

    def test_patch_candidate_points_stay_in_declared_patch(self) -> None:
        """每个局部候选点都必须位于其声明的管理 Patch 内。"""
        points, patch_ids = patch_candidate_points(self.layout, candidates_per_axis=4)

        self.assertEqual(points.shape, (256, 2))
        np.testing.assert_array_equal(self.layout.patch_ids(points), patch_ids)

    def test_patch_selection_keeps_one_highest_score_per_patch(self) -> None:
        """每个 Patch 应独立保留自身分数最高的一个候选点。"""
        points, patch_ids = patch_candidate_points(self.layout, candidates_per_axis=4)
        scores = np.arange(len(points), dtype=np.float32)

        selected, selected_ids, selected_scores = select_patch_teacher_points(
            points, patch_ids, scores, self.layout, points_per_patch=1
        )

        self.assertEqual(selected.shape, (16, 2))
        np.testing.assert_array_equal(selected_ids, np.arange(16, dtype=np.int64))
        np.testing.assert_array_equal(selected_scores, np.arange(15, 256, 16, dtype=np.float32))

    def test_teacher_scores_prefer_descriptor_with_clearer_negative_margin(self) -> None:
        """唯一正确描述子应比重复描述子获得更高的教师分数。"""
        points = np.array([[4.0, 4.0], [20.0, 4.0]], dtype=np.float32)
        target_candidates = np.array([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0]], dtype=np.float32)
        descriptor_a = self._descriptor_grid()
        descriptor_b = self._descriptor_grid()
        descriptor_a[:, 4, 20] = np.array([0.0, 1.0], dtype=np.float32)
        descriptor_b[:, 4, 20] = np.array([0.0, 1.0], dtype=np.float32)

        scores = teacher_scores(
            points,
            target_candidates,
            descriptor_a,
            descriptor_b,
            self._identity_geometry(),
            "a_to_b",
            depth_tolerance=0.01,
            temperature=0.1,
            exclusion_radius=3.0,
            device="cpu",
        )

        self.assertGreater(float(scores[1]), float(scores[0]))
        self.assertGreater(float(scores[1]), 0.9)

    def test_cosine_mutual_matches_return_only_bidirectional_nearest_neighbors(self) -> None:
        """余弦相似度的互为最近邻应剔除共享同一目标的重复源点。"""
        descriptors_a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        descriptors_b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        source, target, similarities = cosine_mutual_matches(descriptors_a, descriptors_b)

        np.testing.assert_array_equal(source, np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(target, np.array([0, 1], dtype=np.int64))
        np.testing.assert_allclose(similarities, np.ones(2, dtype=np.float32))

    def test_sampled_descriptors_read_normalized_features_at_float_coordinates(self) -> None:
        """公开描述子读取接口应返回与坐标对应的归一化局部特征。"""
        grid = self._descriptor_grid()
        grid[:, 4, 20] = np.array([0.0, 1.0], dtype=np.float32)

        descriptors = sampled_descriptors(
            grid, np.array([[4.0, 4.0], [20.0, 4.0]], dtype=np.float32), 32, "cpu"
        )

        np.testing.assert_allclose(descriptors, np.eye(2, dtype=np.float32), atol=1e-6)

    def test_geometry_metrics_measure_visible_coverage_and_match_inliers(self) -> None:
        """投影覆盖率和匹配内点率应分别按可见点和全部匹配边统计。"""
        geometry = self._identity_geometry()
        points_a = np.array([[4.0, 4.0], [20.0, 20.0]], dtype=np.float32)
        points_b = np.array([[4.0, 4.0], [26.0, 20.0]], dtype=np.float32)
        source = np.array([0, 1], dtype=np.int64)
        target = np.array([0, 1], dtype=np.int64)

        coverage = geometry_coverage_rate(points_a, points_b, geometry, "a_to_b", 3.0, 0.01)
        inliers = mutual_geometry_inlier_rate(points_a, points_b, source, target, geometry, 3.0, 0.01)

        self.assertAlmostEqual(coverage, 0.5)
        self.assertAlmostEqual(inliers, 0.5)

    def _descriptor_grid(self) -> np.ndarray:
        """创建默认为相同第一维描述子的二维稠密特征图。"""
        grid = np.zeros((2, 32, 32), dtype=np.float32)
        grid[0] = 1.0
        return grid

    def _identity_geometry(self) -> PairGeometry:
        """创建深度一致的单位相机几何信息。"""
        depth = np.full((32, 32), 2.0, dtype=np.float32)
        return PairGeometry(
            depth_a=depth,
            depth_b=depth.copy(),
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
