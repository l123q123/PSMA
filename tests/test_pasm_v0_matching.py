# 测试 PASM v0 的双图语义条件化与全局点级匹配。

import unittest

import numpy as np
import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.matching import (
    PairLocalQualityEncoder,
    PairSemanticEncoder,
    global_edge_weights,
    patch_feedback,
    select_anchor_matches,
)
from pasm_v0.sampling import PatchLayout


class PasmV0MatchingTests(unittest.TestCase):
    """验证全局点级边分数的基本性质。"""

    def test_identity_descriptors_make_diagonal_edges_strongest(self) -> None:
        """相同的局部与语义描述子应产生对角最高边。"""
        descriptors = torch.eye(3)
        semantics = torch.eye(3)

        weights = global_edge_weights(
            descriptors,
            descriptors,
            semantics,
            semantics,
            torch.arange(3),
            torch.arange(3),
            temperature=0.1,
        )

        torch.testing.assert_close(weights.argmax(dim=1), torch.arange(3))

    def test_patch_feedback_aggregates_point_level_edges(self) -> None:
        """Patch 反馈必须由其内部点的边质量平均得到。"""
        weights = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]], dtype=torch.float32
        )

        feedback = patch_feedback(weights, torch.tensor([0, 0, 1]), patch_count=2)

        self.assertGreater(float(feedback[0]), float(feedback[1]))

    def test_anchor_selection_keeps_mutual_edges_under_both_patch_caps(self) -> None:
        """锚点选择必须同时遵守两张图所属 Patch 的容量。"""
        weights = torch.tensor([[0.90, 0.01], [0.02, 0.80]], dtype=torch.float32)

        anchors = select_anchor_matches(
            weights,
            np.array([0, 0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            threshold=0.1,
            max_points_per_patch=1,
        )

        np.testing.assert_array_equal(anchors.indices_a, np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(anchors.indices_b, np.array([0], dtype=np.int64))
        np.testing.assert_allclose(anchors.scores, np.array([0.9], dtype=np.float32))

    def test_anchor_selection_reserves_next_round_minimum_budget(self) -> None:
        """锚点选择必须为未命中 Patch 留出下一轮最低探索预算。"""
        weights = torch.diag(torch.tensor(np.tile([0.9, 0.8, 0.7, 0.6], 8)))
        patch_ids = np.repeat(np.arange(8), 4).astype(np.int64)

        anchors = select_anchor_matches(
            weights,
            patch_ids,
            patch_ids,
            threshold=0.1,
            max_points_per_patch=4,
            patch_count=16,
            minimum_per_patch=1,
            total_points=32,
        )
        reserved = np.bincount(patch_ids[anchors.indices_a], minlength=16)

        self.assertEqual(len(anchors.indices_a), 24)
        self.assertLessEqual(int(np.maximum(1, reserved).sum()), 32)

    def test_semantic_encoder_returns_value_for_every_patch(self) -> None:
        """双向语义编码器应为两图的全部 Patch 输出条件化价值。"""
        torch.manual_seed(5)
        encoder = PairSemanticEncoder(token_dim=6, hidden_dim=8, heads=2)

        semantic_a, semantic_b, value_a, value_b = encoder(
            torch.randn(1, 4, 6), torch.randn(1, 4, 6)
        )

        self.assertEqual(tuple(semantic_a.shape), (1, 4, 8))
        self.assertEqual(tuple(semantic_b.shape), (1, 4, 8))
        self.assertEqual(tuple(value_a.shape), (1, 4))
        self.assertEqual(tuple(value_b.shape), (1, 4))
        self.assertTrue(torch.all((value_a >= 0.0) & (value_a <= 1.0)))

    def test_local_quality_encoder_returns_dense_joint_descriptors_and_logits(self) -> None:
        """局部质量网络必须从双图语义生成可反传的密集输出。"""
        config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            quality_cells_per_axis=2,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
        )
        torch.manual_seed(7)
        encoder = PairLocalQualityEncoder(local_dim=4, semantic_dim=8, heads=2)
        grid_a = torch.randn(4, 32, 32)
        grid_b = torch.randn(4, 32, 32)
        semantic_a = torch.randn(1, 16, 8)
        semantic_b = torch.randn(1, 16, 8)

        descriptors_a, descriptors_b, logits_a, logits_b = encoder(
            grid_a, grid_b, semantic_a, semantic_b, PatchLayout(config)
        )
        loss = descriptors_a.sum() + descriptors_b.sum() + logits_a.sum() + logits_b.sum()
        loss.backward()

        self.assertEqual(tuple(descriptors_a.shape), (8, 8, 8))
        self.assertEqual(tuple(descriptors_b.shape), (8, 8, 8))
        self.assertEqual(tuple(logits_a.shape), (1, 8, 8))
        self.assertEqual(tuple(logits_b.shape), (1, 8, 8))
        self.assertTrue(torch.isfinite(logits_a).all())
        self.assertIsNotNone(encoder.quality_head.weight.grad)
        self.assertIsNotNone(encoder.local_projection.weight.grad)
        self.assertIsNotNone(encoder.cross_attention.in_proj_weight.grad)


if __name__ == "__main__":
    unittest.main()
