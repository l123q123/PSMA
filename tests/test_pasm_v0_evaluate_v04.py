# 测试 PASM v0.4 A/B 评测的预算分布统计。

import math
import unittest

import numpy as np

from pasm_v0.config import PasmV0Config, dino_token_static_config
from pasm_v0.evaluate_v03 import _static_budget_round
from pasm_v0.evaluate_v04 import (
    budget_distribution_for_config,
    accepted_checkpoint_target_kind,
    aggregate_rows,
    budget_distribution,
)


class PasmV04EvaluateTests(unittest.TestCase):
    """验证 A/B 评测输出能描述预算是否过度集中。"""

    def test_budget_distribution_reports_active_entropy_and_cap_hits(self) -> None:
        """预算统计必须正确计算活跃 Patch、熵和上限命中数。"""
        stats = budget_distribution(np.array([0, 1, 1, 4], dtype=np.int64), max_points_per_patch=4)

        self.assertEqual(stats["active_patches"], 3)
        self.assertEqual(stats["cap_patches"], 1)
        self.assertAlmostEqual(stats["budget_entropy"], -2.0 * (1.0 / 6.0) * math.log(1.0 / 6.0) - (4.0 / 6.0) * math.log(4.0 / 6.0))

    def test_budget_distribution_accepts_static_16_patch_cap(self) -> None:
        """16×16 静态模型的每 Patch 上限是 20，不应被旧的 4 点检查拒绝。"""
        stats = budget_distribution_for_config(
            np.array([0, 4, 20], dtype=np.int64), dino_token_static_config(16)
        )

        self.assertEqual(stats["active_patches"], 2)
        self.assertEqual(stats["cap_patches"], 1)

    def test_static_budget_round_uses_the_final_configured_round(self) -> None:
        """批量静态评测必须适配单轮和旧三轮 checkpoint。"""
        self.assertEqual(_static_budget_round(dino_token_static_config()), 0)
        self.assertEqual(_static_budget_round(PasmV0Config()), 2)

    def test_aggregate_rows_keeps_two_geometry_comparison_labels(self) -> None:
        """同为 geometry 的深浅模型对比不能被固定标签过滤掉。"""
        common = {
            "mutual_matches": 10,
            "inliers_3px": 6,
            "inliers_6px": 7,
            "inlier_rate_3px": 0.6,
            "inlier_rate_6px": 0.7,
            "active_patches_a": 100,
            "active_patches_b": 101,
            "budget_entropy_a": 4.0,
            "budget_entropy_b": 4.1,
        }
        rows = [
            {"target_kind": "geometry_deep", **common},
            {"target_kind": "geometry_single", **common},
        ]

        self.assertEqual([row["target_kind"] for row in aggregate_rows(rows)], [
            "geometry_deep", "geometry_single"
        ])

    def test_geometry_comparison_accepts_calibrated_semantic_checkpoint(self) -> None:
        """几何 A/B 评测必须接受新的 calibrated 语义 checkpoint。"""
        self.assertTrue(accepted_checkpoint_target_kind("geometry", "geometry"))
        self.assertTrue(accepted_checkpoint_target_kind("geometry_calibrated", "geometry"))
        self.assertFalse(accepted_checkpoint_target_kind("quality", "geometry"))


if __name__ == "__main__":
    unittest.main()
