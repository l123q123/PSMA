import unittest
from argparse import Namespace

import torch

from pasm_v0.point_fusion import DensePointFusion
from pasm_v0.train_v05 import (
    PointFusionTrainConfig,
    make_trainable_tensor,
    point_fusion_loss,
    train_options_from_args,
)


class PointFusionTrainingTests(unittest.TestCase):
    def test_training_config_rejects_invalid_teacher_parameters(self) -> None:
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(teacher_candidate_count=0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(teacher_sigma_px=0.0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(fusion_mode="unknown")
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(ranking_mode="unknown")
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(ranking_temperature=0.0)

    def test_point_fusion_loss_backpropagates_through_candidate_logits(self) -> None:
        logits_a = torch.tensor([1.0, -1.0], requires_grad=True)
        logits_b = torch.tensor([0.5, -0.5], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        total, point, rank = point_fusion_loss(logits_a, labels, logits_b, labels, ranking_weight=0.5)
        total.backward()

        self.assertGreater(float(point.detach()), 0.0)
        self.assertGreaterEqual(float(rank.detach()), 0.0)
        self.assertIsNotNone(logits_a.grad)
        self.assertIsNotNone(logits_b.grad)

    def test_soft_pairwise_ranking_remains_positive_after_hinge_margin_is_satisfied(self) -> None:
        logits_a = torch.tensor([3.0, -3.0], requires_grad=True)
        logits_b = torch.tensor([4.0, -4.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        _, _, hinge_rank = point_fusion_loss(logits_a, labels, logits_b, labels, ranking_weight=1.0)
        total, _, soft_pairwise_rank = point_fusion_loss(
            logits_a,
            labels,
            logits_b,
            labels,
            ranking_weight=1.0,
            ranking_mode="soft_pairwise",
            ranking_temperature=1.0,
        )
        total.backward()

        self.assertEqual(float(hinge_rank.detach()), 0.0)
        self.assertGreater(float(soft_pairwise_rank.detach()), 0.0)
        self.assertGreater(float(logits_a.grad.abs().sum()), 0.0)
        self.assertGreater(float(logits_b.grad.abs().sum()), 0.0)

    def test_make_trainable_tensor_allows_frozen_semantic_tokens_in_fusion_backward(self) -> None:
        with torch.inference_mode():
            frozen_tokens = torch.randn(1, 16 * 16, 128)
        semantic_tokens = make_trainable_tensor(frozen_tokens)
        fusion = DensePointFusion(alpha_init=1.0)

        output = fusion(
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 1, 16, 16),
            semantic_tokens,
        )
        output.sum().backward()

        self.assertFalse(semantic_tokens.is_inference())
        self.assertIsNotNone(fusion.semantic_projection.weight.grad)

    def test_cli_options_map_file_arguments_to_train_model_parameter_names(self) -> None:
        options = train_options_from_args(
            Namespace(manifest="pairs.json", output="fusion.pt", epochs=1)
        )

        self.assertEqual(options["manifest_path"], "pairs.json")
        self.assertEqual(options["output_path"], "fusion.pt")
        self.assertNotIn("manifest", options)
        self.assertNotIn("output", options)


if __name__ == "__main__":
    unittest.main()
