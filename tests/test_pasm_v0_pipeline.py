# 测试 PASM v0 的三轮点级联合重采样流程与几何训练目标。

import unittest

import numpy as np
import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.geometry import PairGeometry
from pasm_v0.matching import PairLocalQualityEncoder, PairSemanticEncoder
from pasm_v0.pipeline import PasmV0, RoundRecord
from pasm_v0.sampling import PatchLayout
from pasm_v0.train import _result_loss


class PasmV0PipelineTests(unittest.TestCase):
    """验证 v0 的三轮流程和几何监督接口。"""

    def setUp(self) -> None:
        """创建可快速运行的小尺寸 v0 配置。"""
        self.config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
            quality_cells_per_axis=2,
        )

    def _runner(self) -> PasmV0:
        """创建与小型局部描述子图维度一致的 v0.2 流程。"""
        return PasmV0(
            PairSemanticEncoder(token_dim=8, hidden_dim=8, heads=2),
            PairLocalQualityEncoder(local_dim=4, semantic_dim=8, heads=2),
            self.config,
        )

    def test_pipeline_keeps_budget_and_patch_ownership_for_three_rounds(self) -> None:
        """三轮流程必须始终保持总点数并在最后一轮允许零预算。"""
        torch.manual_seed(3)
        runner = self._runner()

        result = runner.run(
            tokens_a=torch.randn(1, 16, 8),
            tokens_b=torch.randn(1, 16, 8),
            descriptor_grid_a=np.random.default_rng(1).normal(size=(4, 8, 8)).astype(np.float32),
            descriptor_grid_b=np.random.default_rng(2).normal(size=(4, 8, 8)).astype(np.float32),
            seed=5,
        )

        self.assertEqual(len(result.rounds), 3)
        self.assertEqual(result.points_a.shape, (32, 2))
        self.assertEqual(result.points_b.shape, (32, 2))
        self.assertEqual(tuple(result.edge_weights.shape), (32, 32))
        self.assertTrue(np.all(result.rounds[0].budget_a == 2))
        self.assertTrue(np.all(result.rounds[1].budget_a >= 1))
        self.assertTrue(np.all(result.rounds[2].budget_a >= 0))
        self.assertEqual(int(result.rounds[2].budget_a.sum()), 32)
        np.testing.assert_array_equal(PatchLayout(self.config).patch_ids(result.points_a), result.patch_ids_a)
        torch.testing.assert_close(result.rounds[0].quality_logits_a, result.rounds[1].quality_logits_a)
        torch.testing.assert_close(result.rounds[0].quality_logits_a, result.rounds[2].quality_logits_a)

    def test_projected_geometry_targets_drive_point_and_patch_losses(self) -> None:
        """直接投影标签应同时驱动点级和 Patch 价值损失。"""
        points_a = np.array([[1.0, 1.0], [8.0, 8.0], [15.0, 15.0]], dtype=np.float32)
        points_b = np.array([[1.0, 1.0], [8.0, 8.0], [25.0, 25.0]], dtype=np.float32)
        weights = torch.full((3, 3), 1.0 / 3.0, requires_grad=True)
        values = torch.full((16,), 0.5, requires_grad=True)
        patch_ids_a = PatchLayout(self.config).patch_ids(points_a)
        patch_ids_b = PatchLayout(self.config).patch_ids(points_b)
        record = RoundRecord(
            points_a=points_a,
            points_b=points_b,
            patch_ids_a=patch_ids_a,
            patch_ids_b=patch_ids_b,
            budget_a=np.bincount(patch_ids_a, minlength=16),
            budget_b=np.bincount(patch_ids_b, minlength=16),
            edge_weights=weights,
            patch_values_a=values,
            patch_values_b=values,
        )
        geometry = PairGeometry(
            depth_a=np.full((32, 32), 2.0, dtype=np.float32),
            depth_b=np.full((32, 32), 2.0, dtype=np.float32),
            intrinsics_a=np.eye(3, dtype=np.float64),
            intrinsics_b=np.eye(3, dtype=np.float64),
            cam2world_a=np.eye(4, dtype=np.float64),
            cam2world_b=np.eye(4, dtype=np.float64),
            original_size_a=(32, 32),
            original_size_b=(32, 32),
            image_size=32,
        )
        loss = _result_loss([record], geometry, PatchLayout(self.config), self.config)

        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(values.grad)

    def test_pipeline_retains_selected_anchors_in_the_next_round(self) -> None:
        """每轮选出的锚点坐标必须原样进入下一轮点集。"""
        config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
            anchor_threshold=0.0,
        )
        torch.manual_seed(9)
        runner = PasmV0(
            PairSemanticEncoder(token_dim=8, hidden_dim=8, heads=2),
            PairLocalQualityEncoder(local_dim=4, semantic_dim=8, heads=2),
            config,
        )
        result = runner.run(
            tokens_a=torch.randn(1, 16, 8),
            tokens_b=torch.randn(1, 16, 8),
            descriptor_grid_a=np.random.default_rng(3).normal(size=(4, 8, 8)).astype(np.float32),
            descriptor_grid_b=np.random.default_rng(4).normal(size=(4, 8, 8)).astype(np.float32),
            seed=11,
        )

        for round_index, record in enumerate(result.rounds[:-1]):
            next_record = result.rounds[round_index + 1]
            self.assertGreater(len(record.anchor_indices_a), 0)
            self.assertEqual(len(record.anchor_indices_a), len(record.anchor_indices_b))
            for index_a in record.anchor_indices_a:
                self.assertTrue(np.any(np.all(next_record.points_a == record.points_a[index_a], axis=1)))
            for index_b in record.anchor_indices_b:
                self.assertTrue(np.any(np.all(next_record.points_b == record.points_b[index_b], axis=1)))


if __name__ == "__main__":
    unittest.main()
