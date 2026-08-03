# 测试 PASM v0 的 Patch 预算与 Patch 内重采样规则。

import unittest
from unittest.mock import patch

import numpy as np

from pasm_v0.config import PasmV0Config
from pasm_v0.sampling import (
    PatchLayout,
    allocate_patch_budget,
    resample_in_patches,
    sample_quality_points,
)


class PasmV0SamplingTests(unittest.TestCase):
    """验证 v0 的固定预算采样规则。"""

    def test_default_experiment_uses_1024_points_and_six_pixel_labels(self) -> None:
        """默认实验应扩展点预算并放宽当前采样点的投影对齐阈值。"""
        config = PasmV0Config()

        self.assertEqual(config.total_points, 1024)
        self.assertEqual(config.round_min_points, (4, 1, 0))
        self.assertEqual(config.max_points_per_patch, 20)
        self.assertEqual(config.geometry_match_threshold_px, 6.0)
        self.assertEqual(config.anchor_threshold, 0.05)
        self.assertEqual(config.budget_score_power, 2.0)
        self.assertEqual(config.jitter_ratio, 0.35)

    def setUp(self) -> None:
        """创建小尺寸网格以便精确验证采样行为。"""
        self.config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
        )
        self.layout = PatchLayout(self.config)

    def test_initial_sampling_gives_two_points_to_every_patch(self) -> None:
        """首轮必须为每个 Patch 提供两个探索点。"""
        points, patch_ids = self.layout.initial_points(seed=7)

        self.assertEqual(points.shape, (32, 2))
        np.testing.assert_array_equal(
            np.bincount(patch_ids, minlength=16), np.full(16, 2)
        )

    def test_uniform_sampling_near_upper_bound_stays_in_declared_patch(self) -> None:
        """接近 Patch 上边界的随机点在转为 float32 后也不得跨越边界。"""
        generator = _NearUpperBoundGenerator()
        with patch("pasm_v0.sampling.np.random.default_rng", return_value=generator):
            points, patch_ids = self.layout.initial_points(seed=0)

        self.assertTrue(np.all(points < self.config.image_size))
        np.testing.assert_array_equal(self.layout.patch_ids(points), patch_ids)

    def test_first_resampling_keeps_one_point_for_every_patch(self) -> None:
        """第一轮重采样前不允许任何 Patch 被淘汰。"""
        values = np.linspace(0.0, 1.0, 16, dtype=np.float32)

        budget = allocate_patch_budget(values, self.config, round_index=1)

        self.assertEqual(int(budget.sum()), 32)
        self.assertTrue(np.all(budget >= 1))
        self.assertTrue(np.all(budget <= 4))

    def test_later_budget_can_eliminate_low_value_patches(self) -> None:
        """允许淘汰后，低价值 Patch 可以获得零预算。"""
        values = np.zeros(16, dtype=np.float32)
        values[:8] = 1.0

        budget = allocate_patch_budget(values, self.config, round_index=2)

        self.assertEqual(int(budget.sum()), 32)
        np.testing.assert_array_equal(budget[:8], np.full(8, 4))
        np.testing.assert_array_equal(budget[8:], np.zeros(8, dtype=np.int64))

    def test_budget_score_power_concentrates_remaining_budget(self) -> None:
        """更高的预算分数幂必须向高分 Patch 集中更多自由预算。"""
        values = np.linspace(0.1, 1.0, 16, dtype=np.float64)
        common = {
            "image_size": 32,
            "patch_rows": 4,
            "patch_cols": 4,
            "total_points": 32,
            "round_min_points": (2, 0, 0),
            "max_points_per_patch": 4,
        }
        linear = allocate_patch_budget(values, PasmV0Config(**common, budget_score_power=1.0), 1)
        sharpened = allocate_patch_budget(values, PasmV0Config(**common, budget_score_power=2.0), 1)

        self.assertEqual(int(linear.sum()), 32)
        self.assertEqual(int(sharpened.sum()), 32)
        self.assertGreater(int(sharpened[-4:].sum()), int(linear[-4:].sum()))

    def test_budget_score_power_must_be_positive(self) -> None:
        """预算分数幂为零或负数时必须拒绝配置。"""
        with self.assertRaises(ValueError):
            PasmV0Config(budget_score_power=0.0)

    def test_reserved_anchors_raise_patch_budget_lower_bound(self) -> None:
        """被保留锚点的 Patch 预算不能低于锚点数量。"""
        reserved = np.zeros(16, dtype=np.int64)
        reserved[3] = 3

        budget = allocate_patch_budget(np.ones(16, dtype=np.float32), self.config, 2, reserved=reserved)

        self.assertEqual(int(budget.sum()), 32)
        self.assertGreaterEqual(int(budget[3]), 3)

    def test_resampling_keeps_every_point_inside_its_patch(self) -> None:
        """重采样后的坐标必须留在被分配的 Patch 内。"""
        points, patch_ids = self.layout.initial_points(seed=4)
        quality = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
        budget = np.full(16, 2, dtype=np.int64)

        sampled, sampled_ids = resample_in_patches(
            self.layout, points, patch_ids, quality, budget, self.config, seed=9
        )

        self.assertEqual(sampled.shape, (32, 2))
        np.testing.assert_array_equal(np.bincount(sampled_ids, minlength=16), budget)
        np.testing.assert_array_equal(self.layout.patch_ids(sampled), sampled_ids)

    def test_resampling_retains_anchor_coordinates_exactly(self) -> None:
        """下一轮点集必须包含未扰动的锚点原始坐标。"""
        points, patch_ids = self.layout.initial_points(seed=4)
        quality = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
        budget = np.full(16, 2, dtype=np.int64)
        anchor = np.array([[9.0, 9.0]], dtype=np.float32)

        sampled, sampled_ids = resample_in_patches(
            self.layout,
            points,
            patch_ids,
            quality,
            budget,
            self.config,
            anchor_points=anchor,
            anchor_patch_ids=np.array([5], dtype=np.int64),
            anchor_scores=np.array([0.9], dtype=np.float32),
            seed=9,
        )

        locations = np.flatnonzero(np.all(sampled == anchor[0], axis=1))
        self.assertEqual(len(locations), 1)
        self.assertEqual(int(sampled_ids[locations[0]]), 5)

    def test_quality_sampling_uses_unique_cells_inside_each_patch(self) -> None:
        """质量采样的非锚点在同一 Patch 内不能重复质量格。"""
        config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
            quality_cells_per_axis=2,
        )
        layout = PatchLayout(config)
        logits = np.arange(config.patch_count * 4, dtype=np.float32).reshape(config.patch_count, 4)
        budget = np.full(config.patch_count, 2, dtype=np.int64)

        points, patch_ids, cell_ids = sample_quality_points(layout, logits, budget, config, seed=5)

        self.assertEqual(points.shape, (32, 2))
        np.testing.assert_array_equal(layout.patch_ids(points), patch_ids)
        for patch_id in range(config.patch_count):
            selected = cell_ids[patch_ids == patch_id]
            self.assertEqual(len(selected), len(np.unique(selected)))

    def test_quality_sampling_retains_anchor_coordinates_exactly(self) -> None:
        """质量采样必须原样保留锚点并从同一 Patch 补足预算。"""
        config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
            quality_cells_per_axis=2,
        )
        layout = PatchLayout(config)
        anchor = np.array([[9.0, 9.0]], dtype=np.float32)
        budget = np.full(config.patch_count, 2, dtype=np.int64)

        points, patch_ids, _ = sample_quality_points(
            layout,
            np.zeros((config.patch_count, 4), dtype=np.float32),
            budget,
            config,
            seed=7,
            anchor_points=anchor,
            anchor_patch_ids=np.array([5], dtype=np.int64),
        )

        locations = np.flatnonzero(np.all(points == anchor[0], axis=1))
        self.assertEqual(len(locations), 1)
        self.assertEqual(int(patch_ids[locations[0]]), 5)

    def test_quality_sampling_allows_patch_budget_fully_occupied_by_anchors(self) -> None:
        """锚点恰好占满 Patch 预算时不应再请求新的质量格。"""
        config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
            quality_cells_per_axis=2,
        )
        layout = PatchLayout(config)
        anchor = np.array([[9.0, 9.0], [10.0, 10.0]], dtype=np.float32)
        budget = np.full(config.patch_count, 2, dtype=np.int64)

        points, patch_ids, _ = sample_quality_points(
            layout,
            np.zeros((config.patch_count, 4), dtype=np.float32),
            budget,
            config,
            anchor_points=anchor,
            anchor_patch_ids=np.array([5, 5], dtype=np.int64),
        )

        selected = points[patch_ids == 5]
        np.testing.assert_array_equal(selected, anchor)


class _NearUpperBoundGenerator:
    """为边界舍入测试生成无限接近一的随机数。"""

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        """返回所有分量均无限接近一的指定形状数组。"""
        return np.full(shape, np.nextafter(1.0, 0.0), dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
