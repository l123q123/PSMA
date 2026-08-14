import unittest
from unittest.mock import patch

import torch

from pasm_v0.point_fusion import (
    CandidateSuppressionFusion,
    ContrastiveDensePointFusion,
    DensePointFusion,
    PatchGatedDaDFusion,
    SemanticCoverageDaDFusion,
    create_point_fusion,
    local_logit_zscore,
    point_candidate_logits,
    select_candidate_topk,
    select_dense_topk,
)


class DensePointFusionTests(unittest.TestCase):
    def test_semantic_coverage_fusion_starts_as_dense_original_dad_logits(self) -> None:
        model = SemanticCoverageDaDFusion()
        dad = torch.randn(1, 32, 8, 8)
        base = torch.randn(1, 1, 16, 16)
        tokens = torch.randn(1, 16 * 16, 128)

        fused = model(dad, base, tokens)

        self.assertEqual(tuple(fused.shape), (1, 1, 16, 16))
        torch.testing.assert_close(fused, base)

    def test_patch_gate_initializes_to_neutral_half_probability(self) -> None:
        model = PatchGatedDaDFusion()
        base = torch.zeros(1, 1, 16, 16)
        tokens = torch.randn(1, 2 * 2, 128)
        candidates = torch.tensor([[1.0, 1.0]])

        _, gates = model(base, tokens, candidates, image_size=16)

        torch.testing.assert_close(gates, torch.full_like(gates, 0.5))

    def test_patch_gate_preserves_original_dad_order_inside_a_trusted_patch(self) -> None:
        model = PatchGatedDaDFusion()
        with torch.no_grad():
            model.semantic_gate_head.weight.zero_()
            model.semantic_gate_head.bias.fill_(20.0)
        base = torch.zeros(1, 1, 16, 16)
        base[0, 0, 1, 1] = 3.0
        base[0, 0, 1, 2] = 1.0
        tokens = torch.zeros(1, 2 * 2, 128)
        candidates = torch.tensor([[1.0, 1.0], [2.0, 1.0]])

        final_logits, gates = model(base, tokens, candidates, image_size=16)

        torch.testing.assert_close(final_logits[0] - final_logits[1], torch.tensor(2.0))
        self.assertGreater(float(gates.min().detach()), 0.99)

    def test_candidate_suppression_only_reduces_fixed_dad_candidate_scores(self) -> None:
        model = CandidateSuppressionFusion()
        dad = torch.randn(1, 32, 8, 8)
        base = torch.randn(1, 1, 16, 16)
        tokens = torch.randn(1, 16 * 16, 128)
        candidates = torch.tensor([[1.0, 2.0], [12.0, 11.0]], dtype=torch.float32)

        final_logits, penalties = model(dad, base, tokens, candidates, image_size=16)
        sampled_base = point_candidate_logits(base, candidates, image_size=16)

        self.assertEqual(tuple(final_logits.shape), (2,))
        self.assertEqual(tuple(penalties.shape), (2,))
        self.assertTrue(torch.all(penalties >= 0.0))
        self.assertTrue(torch.all(final_logits <= sampled_base))

    def test_candidate_suppression_preserves_high_semantic_quality_more_than_low_quality(self) -> None:
        model = CandidateSuppressionFusion()
        with torch.no_grad():
            model.semantic_quality_head.weight.zero_()
            model.candidate_penalty_head[-1].weight.zero_()
            model.candidate_penalty_head[-1].bias.zero_()
        dad = torch.zeros(1, 32, 8, 8)
        base = torch.zeros(1, 1, 16, 16)
        tokens = torch.zeros(1, 16 * 16, 128)
        candidates = torch.tensor([[8.0, 8.0]], dtype=torch.float32)

        with torch.no_grad():
            model.semantic_quality_head.bias.fill_(6.0)
        _, high_quality_penalty = model(dad, base, tokens, candidates, image_size=16)
        with torch.no_grad():
            model.semantic_quality_head.bias.fill_(-6.0)
        _, low_quality_penalty = model(dad, base, tokens, candidates, image_size=16)

        self.assertLess(float(high_quality_penalty[0].detach()), float(low_quality_penalty[0].detach()))

    def test_candidate_suppression_initializes_with_shared_non_saturated_penalty(self) -> None:
        model = CandidateSuppressionFusion()
        dad = torch.randn(1, 32, 8, 8)
        base = torch.tensor([[[[3.0, 1.0], [2.0, 0.0]]]])
        tokens = torch.randn(1, 16 * 16, 128)
        candidates = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        final_logits, penalties = model(dad, base, tokens, candidates, image_size=2)
        sampled_base = point_candidate_logits(base, candidates, image_size=2)

        self.assertGreater(float(penalties.min().detach()), 0.1)
        self.assertLess(float(penalties.max().detach()), 1.0)
        torch.testing.assert_close(penalties[0], penalties[1])
        torch.testing.assert_close(final_logits[0] - final_logits[1], sampled_base[0] - sampled_base[1])
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

        self.assertGreater(float(output[0, 0, 4, 4].detach()), float(base[0, 0, 4, 4]))
        self.assertLess(float(output[0, 0, 4, 3].detach()), float(base[0, 0, 4, 3]))

    def test_fusion_factory_preserves_residual_default_and_rejects_unknown_mode(self) -> None:
        self.assertIsInstance(create_point_fusion("residual"), DensePointFusion)
        self.assertIsInstance(create_point_fusion("contrastive"), ContrastiveDensePointFusion)
        self.assertIsInstance(create_point_fusion("candidate_suppression"), CandidateSuppressionFusion)
        self.assertIsInstance(create_point_fusion("patch_gate"), PatchGatedDaDFusion)
        self.assertIsInstance(create_point_fusion("semantic_coverage"), SemanticCoverageDaDFusion)
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

    def test_dense_topk_can_enable_dad_native_coverage_reweighting(self) -> None:
        """覆盖消融必须把 DaD 原生 density reweighting 传给最终选点器。"""
        logits = torch.zeros(1, 1, 16, 16)
        points = torch.zeros((1, 4, 2))
        scores = torch.ones((1, 4))
        with patch("loma.detector.dad.sample_keypoints", return_value=(points, scores)) as sampler:
            select_dense_topk(logits, num_points=4, increase_coverage=True)

        self.assertTrue(sampler.call_args.kwargs["increase_coverage"])

    def test_candidate_topk_uses_only_fixed_candidates_and_excludes_border_band(self) -> None:
        candidates = torch.tensor(
            [[0.5, 8.0], [4.0, 4.0], [8.0, 8.0], [14.0, 8.0]], dtype=torch.float32
        )
        logits = torch.tensor([10.0, 1.0, 2.0, 9.0])

        points, scores = select_candidate_topk(
            candidates, logits, num_points=2, image_size=16, border_margin_px=2
        )

        torch.testing.assert_close(points, torch.tensor([[8.0, 8.0], [4.0, 4.0]]))
        torch.testing.assert_close(scores, torch.tensor([2.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
