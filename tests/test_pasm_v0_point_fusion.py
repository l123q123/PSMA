import unittest

import torch

from pasm_v0.point_fusion import (
    ContrastiveDensePointFusion,
    DensePointFusion,
    create_point_fusion,
    local_logit_zscore,
    point_candidate_logits,
    select_dense_topk,
)


class DensePointFusionTests(unittest.TestCase):
    def test_local_logit_zscore_is_finite_and_orders_local_contrast(self) -> None:
        logits = torch.zeros(1, 1, 7, 7)
        logits[0, 0, 3, 3] = 4.0
        logits[0, 0, 3, 2] = -2.0

        zscore = local_logit_zscore(logits, window_size=5)

        self.assertTrue(torch.isfinite(zscore).all())
        self.assertGreater(float(zscore[0, 0, 3, 3]), 0.0)
        self.assertLess(float(zscore[0, 0, 3, 2]), 0.0)
        torch.testing.assert_close(local_logit_zscore(torch.ones_like(logits)), torch.zeros_like(logits))

    def test_contrastive_fusion_is_near_identity_at_initialization(self) -> None:
        model = ContrastiveDensePointFusion()
        dad = torch.randn(1, 32, 8, 8)
        base = torch.randn(1, 1, 16, 16)
        tokens = torch.randn(1, 16 * 16, 128)

        output = model(dad, base, tokens)

        torch.testing.assert_close(output, base, atol=0.05, rtol=0.0)

    def test_contrastive_fusion_amplifies_high_local_dad_logits_and_reduces_low_ones(self) -> None:
        model = ContrastiveDensePointFusion()
        with torch.no_grad():
            model.gain_head.weight.zero_()
            model.gain_head.bias.zero_()
            model.penalty_head.weight.zero_()
            model.penalty_head.bias.fill_(-20.0)
        dad = torch.zeros(1, 32, 8, 8)
        base = torch.zeros(1, 1, 9, 9)
        base[0, 0, 4, 4] = 4.0
        base[0, 0, 4, 3] = -2.0
        tokens = torch.zeros(1, 16 * 16, 128)

        output = model(dad, base, tokens)

        self.assertGreater(float(output[0, 0, 4, 4]), float(base[0, 0, 4, 4]))
        self.assertLess(float(output[0, 0, 4, 3]), float(base[0, 0, 4, 3]))

    def test_fusion_factory_preserves_residual_default_and_rejects_unknown_mode(self) -> None:
        self.assertIsInstance(create_point_fusion("residual"), DensePointFusion)
        self.assertIsInstance(create_point_fusion("contrastive"), ContrastiveDensePointFusion)
        with self.assertRaises(ValueError):
            create_point_fusion("unknown")

    def test_residual_zero_initialization_preserves_dad_base_logits(self) -> None:
        model = DensePointFusion(semantic_dim=128, dad_channels=32, alpha_init=0.0)
        dad = torch.randn(1, 32, 16, 16)
        base = torch.randn(1, 1, 16, 16)
        tokens = torch.randn(1, 16 * 16, 128)

        output = model(dad, base, tokens)

        self.assertEqual(tuple(output.shape), tuple(base.shape))
        torch.testing.assert_close(output, base)

    def test_fusion_accepts_both_semantic_token_grids(self) -> None:
        model = DensePointFusion(semantic_dim=128, dad_channels=32)
        dad = torch.randn(1, 32, 16, 16)
        base = torch.randn(1, 1, 16, 16)

        for grid in (16, 37):
            with self.subTest(grid=grid):
                output = model(dad, base, torch.randn(1, grid * grid, 128))
                self.assertEqual(tuple(output.shape), (1, 1, 16, 16))

    def test_fusion_convolutions_use_reflect_padding_at_image_boundaries(self) -> None:
        model = DensePointFusion()
        convolutions = [layer for layer in model.fusion if isinstance(layer, torch.nn.Conv2d)]

        self.assertEqual([layer.padding_mode for layer in convolutions[:-1]], ["reflect", "reflect"])

    def test_candidate_logits_are_bilinearly_sampled_at_pixel_coordinates(self) -> None:
        logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        points = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32)

        sampled = point_candidate_logits(logits, points, image_size=4)

        self.assertEqual(tuple(sampled.shape), (2,))
        self.assertAlmostEqual(float(sampled[0]), 0.0, places=5)
        self.assertAlmostEqual(float(sampled[1]), 15.0, places=5)

    def test_dense_topk_returns_requested_normalized_points(self) -> None:
        logits = torch.zeros(1, 1, 16, 16)
        logits[0, 0, 3, 5] = 5.0

        points, scores = select_dense_topk(logits, num_points=4)

        self.assertEqual(tuple(points.shape), (4, 2))
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.isfinite(points).all())


if __name__ == "__main__":
    unittest.main()
