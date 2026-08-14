# 测试 PASM v0 的 Patch 预算与 Patch 内重采样规则。

import unittest
from unittest.mock import patch

import numpy as np

from pasm_v0.config import PasmV0Config, dino_token_static_config, legacy_iterative_37_config
from pasm_v0.sampling import PatchLayout, allocate_patch_budget, resample_in_patches


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

    def test_dino_token_static_config_uses_one_round_and_37_square_patches(self) -> None:
        """DINO 原始 token 网格静态版本允许零预算 Patch。"""
        config = dino_token_static_config()
        budget = allocate_patch_budget(
            np.linspace(0.0, 1.0, config.patch_count, dtype=np.float32),
            config,
            round_index=0,
        )
        self.assertEqual((config.patch_rows, config.patch_cols), (37, 37))
        self.assertEqual(config.iterations, 1)
        self.assertEqual(config.round_min_points, (0,))
        self.assertEqual(config.max_points_per_patch, 4)
        self.assertEqual(config.budget_score_power, 1.0)
        self.assertEqual(int(budget.sum()), 1024)
        self.assertTrue(np.all(budget >= 0))
        self.assertTrue(np.all(budget <= 4))
        self.assertGreater(int(np.count_nonzero(budget == 0)), 0)

    def test_legacy_iterative_37_initial_sampling_uses_1024_unique_patches(self) -> None:
        """37×37 旧三轮配置首轮应从不同 Patch 中生成 1024 个点。"""
        config = legacy_iterative_37_config()
        layout = PatchLayout(config)
        points, patch_ids = layout.initial_points(seed=7)
        self.assertEqual(points.shape, (1024, 2))
        self.assertEqual(config.geometry_support_stride, 2)
        self.assertEqual(len(np.unique(patch_ids)), 1024)
        np.testing.assert_array_equal(layout.patch_ids(points), patch_ids)

    def setUp(self) -> None:
        """创建小尺寸网格以便精确验证采样行为。"""
        self.config = PasmV0Config(
            image_size=32, patch_rows=4, patch_cols=4, total_points=32,
            round_min_points=(2, 1, 0), max_points_per_patch=4,
        )
        self.layout = PatchLayout(self.config)

    def test_initial_sampling_gives_two_points_to_every_patch(self) -> None:
        """首轮必须为每个 Patch 提供两个探索点。"""
        points, patch_ids = self.layout.initial_points(seed=7)
        self.assertEqual(points.shape, (32, 2))
        np.testing.assert_array_equal(np.bincount(patch_ids, minlength=16), np.full(16, 2))

    def test_uniform_sampling_near_upper_bound_stays_in_declared_patch(self) -> None:
        """接近 Patch 上边界的随机点在转为 float32 后也不得跨越边界。"""
        generator = _NearUpperBoundGenerator()
        with patch("pasm_v0.sampling.np.random.default_rng", return_value=generator):
            points, patch_ids = self.layout.initial_points(seed=0)
        self.assertTrue(np.all(points < self.config.image_size))
        np.testing.assert_array_equal(self.layout.patch_ids(points), patch_ids)

    def test_first_resampling_keeps_one_point_for_every_patch(self) -> None:
        """第一轮重采样前不允许任何 Patch 被淘汰。"""
        budget = allocate_patch_budget(np.linspace(0.0, 1.0, 16), self.config, round_index=1)
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
        values = np.linspace(0.1, 1.0, 16)
        common = dict(image_size=32, patch_rows=4, patch_cols=4, total_points=32,
                      round_min_points=(2, 0, 0), max_points_per_patch=4)
        linear = allocate_patch_budget(values, PasmV0Config(**common, budget_score_power=1.0), 1)
        sharpened = allocate_patch_budget(values, PasmV0Config(**common, budget_score_power=2.0), 1)
        self.assertGreater(int(sharpened[-4:].sum()), int(linear[-4:].sum()))

    def test_budget_score_power_must_be_positive(self) -> None:
        """预算分数幂为零或负数时必须拒绝配置。"""
        with self.assertRaises(ValueError):
            PasmV0Config(budget_score_power=0.0)

    def test_reserved_anchors_raise_patch_budget_lower_bound(self) -> None:
        """被保留锚点的 Patch 预算不能低于锚点数量。"""
        reserved = np.zeros(16, dtype=np.int64)
        reserved[3] = 3
        budget = allocate_patch_budget(np.ones(16), self.config, 2, reserved=reserved)
        self.assertGreaterEqual(int(budget[3]), 3)

    def test_resampling_keeps_every_point_inside_its_patch(self) -> None:
        """重采样后的坐标必须留在被分配的 Patch 内。"""
        points, patch_ids = self.layout.initial_points(seed=4)
        sampled, sampled_ids = resample_in_patches(
            self.layout, points, patch_ids, np.linspace(0.0, 1.0, len(points)),
            np.full(16, 2), self.config, seed=9,
        )
        self.assertEqual(sampled.shape, (32, 2))
        np.testing.assert_array_equal(np.bincount(sampled_ids, minlength=16), np.full(16, 2))
        np.testing.assert_array_equal(self.layout.patch_ids(sampled), sampled_ids)

    def test_resampling_retains_anchor_coordinates_exactly(self) -> None:
        """下一轮点集必须包含未扰动的锚点原始坐标。"""
        points, patch_ids = self.layout.initial_points(seed=4)
        anchor = np.array([[9.0, 9.0]], dtype=np.float32)
        sampled, sampled_ids = resample_in_patches(
            self.layout, points, patch_ids, np.linspace(0.0, 1.0, len(points)),
            np.full(16, 2), self.config, anchor_points=anchor,
            anchor_patch_ids=np.array([5]), anchor_scores=np.array([0.9]), seed=9,
        )
        locations = np.flatnonzero(np.all(sampled == anchor[0], axis=1))
        self.assertEqual(len(locations), 1)
        self.assertEqual(int(sampled_ids[locations[0]]), 5)


class _NearUpperBoundGenerator:
    """为边界舍入测试生成无限接近一的随机数。"""

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        """返回所有分量均无限接近一的指定形状数组。"""
        return np.full(shape, np.nextafter(1.0, 0.0), dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
