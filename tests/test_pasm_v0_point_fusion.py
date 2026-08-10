import unittest

import torch

from pasm_v0.point_fusion import DensePointFusion, point_candidate_logits, select_dense_topk


class DensePointFusionTests(unittest.TestCase):
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
