import unittest
from argparse import Namespace
from pathlib import Path
import tempfile

import torch

from pasm_v0.point_fusion import DensePointFusion, create_point_fusion
from pasm_v0.config import PasmV0Config
from pasm_v0.train_v05 import (
    PointFusionTrainConfig,
    candidate_supervision_labels,
    candidate_suppression_loss,
    dense_geometric_invisibility_loss,
    dense_local_dad_preservation_loss,
    dense_semantic_coverage_loss,
    geometric_invisibility_loss,
    make_trainable_tensor,
    local_dad_preservation_loss,
    patch_gate_loss,
    patch_gate_targets,
    patch_support_targets,
    point_fusion_loss,
    _load_fusion_checkpoint,
    _save_checkpoint,
    semantic_coverage_loss,
    soft_patch_occupancy,
    train_options_from_args,
)


class PointFusionTrainingTests(unittest.TestCase):
    def test_dense_local_preservation_keeps_base_order_inside_supported_patch(self) -> None:
        base = torch.zeros((1, 1, 16, 16))
        base[0, 0, 2, 2] = 3.0
        base[0, 0, 2, 3] = 1.0
        fused = base.clone().requires_grad_(True)
        swapped = base.clone().requires_grad_(True)
        with torch.no_grad():
            swapped[0, 0, 2, 2] = 1.0
            swapped[0, 0, 2, 3] = 3.0
        support = torch.tensor([1.0, 0.0, 0.0, 0.0])
        valid = torch.tensor([True, False, False, False])

        equal_loss = dense_local_dad_preservation_loss(
            fused, base, support, valid, image_size=16, patch_grid=2, temperature=1.0,
        )
        swapped_loss = dense_local_dad_preservation_loss(
            swapped, base, support, valid, image_size=16, patch_grid=2, temperature=1.0,
        )
        swapped_loss.backward()

        torch.testing.assert_close(equal_loss, torch.tensor(0.0))
        self.assertGreater(float(swapped_loss.detach()), 0.0)
        self.assertGreater(float(swapped.grad.abs().sum()), 0.0)

    def test_dense_coverage_loss_rewards_supported_patches_with_dense_topk_mass(self) -> None:
        support = torch.tensor([1.0, 1.0, 0.0, 0.0])
        valid = torch.tensor([True, True, False, False])
        spread = torch.full((1, 1, 16, 16), -6.0, requires_grad=True)
        clustered = torch.full((1, 1, 16, 16), -6.0, requires_grad=True)
        with torch.no_grad():
            spread[0, 0, 2, 2] = 6.0
            spread[0, 0, 2, 10] = 6.0
            clustered[0, 0, 2, 2] = 6.0
            clustered[0, 0, 3, 2] = 5.0

        spread_loss = dense_semantic_coverage_loss(
            spread, support, valid, image_size=16, patch_grid=2, topk=2, temperature=0.5,
        )
        clustered_loss = dense_semantic_coverage_loss(
            clustered, support, valid, image_size=16, patch_grid=2, topk=2, temperature=0.5,
        )
        spread_loss.backward()

        self.assertLess(float(spread_loss.detach()), float(clustered_loss.detach()))
        self.assertGreater(float(spread.grad.abs().sum()), 0.0)

    def test_dense_invisibility_loss_penalizes_topk_mass_in_fully_invisible_patch(self) -> None:
        fully_invisible = torch.tensor([True, False, False, False])
        visible_preferred = torch.full((1, 1, 16, 16), -6.0, requires_grad=True)
        invisible_preferred = torch.full((1, 1, 16, 16), -6.0, requires_grad=True)
        with torch.no_grad():
            visible_preferred[0, 0, 2, 10] = 6.0
            invisible_preferred[0, 0, 2, 2] = 6.0

        good_loss = dense_geometric_invisibility_loss(
            visible_preferred, fully_invisible, image_size=16, patch_grid=2, topk=1, temperature=0.5,
        )
        bad_loss = dense_geometric_invisibility_loss(
            invisible_preferred, fully_invisible, image_size=16, patch_grid=2, topk=1, temperature=0.5,
        )
        bad_loss.backward()

        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        self.assertGreater(float(invisible_preferred.grad[0, 0, 2, 2]), 0.0)

    def test_geometric_invisibility_loss_discourages_soft_topk_mass_on_fully_invisible_patches(self) -> None:
        candidates = torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 1.0], [10.0, 2.0]])
        visible = torch.tensor([False, False, True, True])
        visible_preferred = torch.tensor([-3.0, -4.0, 3.0, 2.0], requires_grad=True)
        invisible_preferred = torch.tensor([3.0, 2.0, -3.0, -4.0], requires_grad=True)

        good_loss = geometric_invisibility_loss(
            visible_preferred, candidates, visible, image_size=16, patch_grid=2, topk=2, temperature=0.5,
        )
        bad_loss = geometric_invisibility_loss(
            invisible_preferred, candidates, visible, image_size=16, patch_grid=2, topk=2, temperature=0.5,
        )
        bad_loss.backward()

        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        self.assertGreater(float(invisible_preferred.grad[0]), 0.0)

    def test_geometric_invisibility_loss_is_zero_when_all_candidates_are_visible(self) -> None:
        logits = torch.tensor([0.0, 2.0], requires_grad=True)
        candidates = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

        loss = geometric_invisibility_loss(
            logits, candidates, torch.tensor([True, True]), image_size=16, patch_grid=2, topk=1, temperature=0.5,
        )
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_geometric_invisibility_loss_exempts_mixed_patches(self) -> None:
        logits = torch.tensor([2.0, 3.0], requires_grad=True)
        candidates = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

        loss = geometric_invisibility_loss(
            logits, candidates, torch.tensor([True, False]), image_size=16, patch_grid=2, topk=1, temperature=0.5,
        )
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_geometric_invisibility_loss_keeps_gradient_when_every_candidate_is_invisible(self) -> None:
        logits = torch.tensor([0.0, 3.0], requires_grad=True)
        candidates = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

        loss = geometric_invisibility_loss(
            logits, candidates, torch.tensor([False, False]), image_size=16, patch_grid=2, topk=1, temperature=0.5,
        )
        loss.backward()

        self.assertGreater(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[1]), 0.0)

    def test_semantic_coverage_loss_rewards_supported_patches_with_selected_candidates(self) -> None:
        candidates = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [9.0, 1.0], [10.0, 2.0]],
            dtype=torch.float32,
        )
        support = torch.tensor([1.0, 1.0, 0.0, 0.0])
        valid = torch.tensor([True, True, False, False])
        spread = torch.tensor([4.0, -5.0, 4.0, -5.0], requires_grad=True)
        clustered = torch.tensor([5.0, 4.0, -5.0, -6.0], requires_grad=True)

        spread_loss = semantic_coverage_loss(
            spread, candidates, support, valid, image_size=16, patch_grid=2, topk=2, temperature=0.5,
        )
        clustered_loss = semantic_coverage_loss(
            clustered, candidates, support, valid, image_size=16, patch_grid=2, topk=2, temperature=0.5,
        )
        spread_loss.backward()

        self.assertLess(float(spread_loss.detach()), float(clustered_loss.detach()))
        self.assertGreater(float(spread.grad.abs().sum()), 0.0)

    def test_soft_patch_occupancy_and_coverage_loss_ignore_invalid_patches(self) -> None:
        candidates = torch.tensor(
            [[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [9.0, 9.0]], dtype=torch.float32,
        )
        logits = torch.tensor([3.0, 2.0, 100.0, 100.0])
        occupancy = soft_patch_occupancy(logits, candidates, image_size=16, patch_grid=2, topk=2, temperature=1.0)
        support = torch.tensor([1.0, 1.0, 0.0, 0.0])
        valid = torch.tensor([True, True, False, False])

        loss = semantic_coverage_loss(
            logits, candidates, support, valid, image_size=16, patch_grid=2, topk=2, temperature=1.0,
        )
        changed = semantic_coverage_loss(
            logits,
            candidates,
            torch.tensor([1.0, 1.0, 999.0, 999.0]),
            valid,
            image_size=16,
            patch_grid=2,
            topk=2,
            temperature=1.0,
        )

        self.assertTrue(torch.all((occupancy >= 0.0) & (occupancy <= 1.0)))
        torch.testing.assert_close(loss, changed)

    def test_patch_support_targets_and_local_dad_preservation_are_finite(self) -> None:
        candidates = torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 1.0]], dtype=torch.float32)
        support, valid = patch_support_targets(
            labels=torch.tensor([0.3, 0.9, 0.0]),
            visible=torch.tensor([True, True, True]),
            candidates_xy=candidates,
            image_size=16,
            patch_grid=2,
        )
        base = torch.tensor([3.0, 1.0, 0.0])
        equal = local_dad_preservation_loss(
            base, base, candidates, support, valid, image_size=16, patch_grid=2, temperature=1.0,
        )
        swapped = local_dad_preservation_loss(
            torch.tensor([1.0, 3.0, 0.0]), base, candidates, support, valid, image_size=16, patch_grid=2, temperature=1.0,
        )
        empty = local_dad_preservation_loss(
            base, base, candidates, torch.zeros_like(support), torch.zeros_like(valid), image_size=16, patch_grid=2, temperature=1.0,
        )

        torch.testing.assert_close(support, torch.tensor([0.9, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(valid, torch.tensor([True, True, False, False]))
        torch.testing.assert_close(equal, torch.tensor(0.0))
        self.assertGreater(float(swapped.detach()), 0.0)
        self.assertTrue(torch.isfinite(empty))

    def test_patch_gate_targets_use_visible_maximum_and_loss_ignores_unknown_patches(self) -> None:
        targets = patch_gate_targets(
            labels=torch.tensor([0.2, 0.9, 0.0, 1.0]),
            visible=torch.tensor([True, True, True, False]),
            candidates_xy=torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 1.0], [14.0, 14.0]]),
            image_size=16,
            patch_grid=2,
        )
        torch.testing.assert_close(targets, torch.tensor([0.9, 0.0, -1.0, -1.0]))

        gates = torch.tensor([0.8, 0.2, 0.5, 0.5], requires_grad=True)
        loss = patch_gate_loss(gates, targets, gates, targets)
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(gates.grad[2]), 0.0)
        self.assertEqual(float(gates.grad[3]), 0.0)

    def test_training_config_rejects_invalid_teacher_parameters(self) -> None:
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(teacher_candidate_count=0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(teacher_sigma_px=0.0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(fusion_mode="unknown")
        self.assertEqual(PointFusionTrainConfig(fusion_mode="candidate_suppression").fusion_mode, "candidate_suppression")
        self.assertEqual(PointFusionTrainConfig(fusion_mode="patch_gate").fusion_mode, "patch_gate")
        self.assertEqual(PointFusionTrainConfig(fusion_mode="semantic_coverage").fusion_mode, "semantic_coverage")
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(ranking_mode="unknown")
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(ranking_temperature=0.0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(border_margin_px=-1)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(gate_penalty_scale=0.0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(gate_quality_gamma=0.0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(coverage_weight=-0.1)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(local_preservation_weight=-0.1)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(coverage_temperature=0.0)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(coverage_topk=0)
        self.assertEqual(PointFusionTrainConfig(invisibility_weight=0.5).invisibility_weight, 0.5)
        with self.assertRaises(ValueError):
            PointFusionTrainConfig(invisibility_weight=-0.1)

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

    def test_candidate_suppression_loss_keeps_positive_candidates_and_pushes_down_negatives(self) -> None:
        final_a = torch.tensor([2.0, -1.0], requires_grad=True)
        final_b = torch.tensor([1.5, -0.5], requires_grad=True)
        penalties_a = torch.tensor([0.1, 2.0], requires_grad=True)
        penalties_b = torch.tensor([0.1, 2.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        total, keep, suppress, rank = candidate_suppression_loss(
            final_a, penalties_a, labels, final_b, penalties_b, labels,
            suppression_weight=1.0, suppression_margin=1.0, ranking_weight=0.5,
        )
        total.backward()

        self.assertGreater(float(keep.detach()), 0.0)
        self.assertGreater(float(suppress.detach()), 0.0)
        self.assertGreaterEqual(float(rank.detach()), 0.0)
        self.assertIsNotNone(penalties_a.grad)
        self.assertIsNotNone(final_b.grad)

    def test_candidate_suppression_loss_ignores_unknown_teacher_candidates(self) -> None:
        labels = torch.tensor([1.0, -1.0, 0.0])
        final = torch.tensor([2.0, 0.0, -1.0])
        penalties = torch.tensor([0.1, 0.5, 1.5])
        _, keep, suppress, rank = candidate_suppression_loss(
            final, penalties, labels, final, penalties, labels,
            suppression_weight=1.0, suppression_margin=1.0, ranking_weight=0.5,
        )
        changed_penalties = penalties.clone()
        changed_penalties[1] = 100.0
        changed_final = final.clone()
        changed_final[1] = -100.0
        _, changed_keep, changed_suppress, changed_rank = candidate_suppression_loss(
            changed_final, changed_penalties, labels, changed_final, changed_penalties, labels,
            suppression_weight=1.0, suppression_margin=1.0, ranking_weight=0.5,
        )

        torch.testing.assert_close(keep, changed_keep)
        torch.testing.assert_close(suppress, changed_suppress)
        torch.testing.assert_close(rank, changed_rank)

    def test_candidate_supervision_labels_ignore_candidates_without_reliable_visibility(self) -> None:
        labels = torch.tensor([1.0, 0.0, 0.7])
        visible = torch.tensor([True, True, False])

        supervised = candidate_supervision_labels(labels, visible)

        torch.testing.assert_close(supervised, torch.tensor([1.0, 0.0, -1.0]))

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
            Namespace(manifest="pairs.json", output="fusion.pt", epochs=1, resume="resume.pt")
        )

        self.assertEqual(options["manifest_path"], "pairs.json")
        self.assertEqual(options["output_path"], "fusion.pt")
        self.assertNotIn("manifest", options)
        self.assertNotIn("output", options)
        self.assertEqual(options["resume_path"], "resume.pt")
        self.assertNotIn("resume", options)

    def test_v05_resume_restores_fusion_optimizer_history_and_epoch(self) -> None:
        fusion = create_point_fusion("semantic_coverage")
        optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
        settings = PointFusionTrainConfig(fusion_mode="semantic_coverage")
        history = [
            {"epoch": 1, "total_loss": 1.7},
            {"epoch": 2, "total_loss": 1.5},
        ]
        source_config = PasmV0Config(
            image_size=518,
            patch_rows=16,
            patch_cols=16,
            iterations=1,
            round_min_points=(0,),
            max_points_per_patch=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            _save_checkpoint(path, fusion, optimizer, source_config, "semantic.pt", settings, history, 1.5)
            restored = create_point_fusion("semantic_coverage")
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
            epoch, restored_history, best_loss = _load_fusion_checkpoint(
                path, restored, restored_optimizer, torch.device("cpu"), settings,
            )

        self.assertEqual(epoch, 2)
        self.assertEqual(restored_history, history)
        self.assertEqual(best_loss, 1.5)
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
        for expected, actual in zip(fusion.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
