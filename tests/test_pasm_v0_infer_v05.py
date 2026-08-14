import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from pasm_v0.dad import DaDDetections
from pasm_v0.infer_v05 import V05Runner, _candidate_score_map, _select_suppressed_candidates
from pasm_v0.point_fusion import SemanticCoverageDaDFusion


class CandidateSuppressionInferenceTests(unittest.TestCase):
    def test_adaptive_threshold_mode_does_not_fill_rejected_dense_peaks(self) -> None:
        detection = DaDDetections(
            dense_scores=np.zeros((16, 16), dtype=np.float32),
            nms_points=np.array([[2.0, 2.0], [12.0, 12.0]], dtype=np.float32),
            nms_scores=np.array([2.0, 1.0], dtype=np.float32),
            dense_logits=np.zeros((16, 16), dtype=np.float32),
            feature_map=np.zeros((8, 8, 32), dtype=np.float32),
        )

        class ConstantThreshold(nn.Module):
            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return torch.ones(1, device=tokens.device)

        selector = Mock(return_value=(
            torch.tensor([[-0.6875, -0.6875], [0.5625, 0.5625]]),
            torch.tensor([0.9, 0.8]),
        ))
        runner = V05Runner(
            config=type("Config", (), {"image_size": 16, "patch_rows": 16, "patch_cols": 16, "total_points": 2})(),
            dino=lambda images: images,
            semantic_encoder=lambda tokens_a, tokens_b: (
                torch.zeros(1, 16 * 16, 128), torch.zeros(1, 16 * 16, 128), None, None,
            ),
            detector=type("Detector", (), {"detect": lambda _, __: detection})(),
            fusion=SemanticCoverageDaDFusion(),
            device=torch.device("cpu"),
            point_selector=selector,
            fusion_mode="semantic_coverage",
            adaptive_threshold=ConstantThreshold(),
        )
        image = np.zeros((3, 16, 16), dtype=np.float32)
        with patch("pasm_v0.infer_v05.pool_dino_tokens", return_value=torch.zeros(2, 16 * 16, 128)):
            points_a, _, scores_a, _, diagnostics = runner.select(image, image)

        self.assertEqual(points_a.shape, (0, 2))
        self.assertEqual(scores_a.shape, (0,))
        self.assertEqual(diagnostics["selected_count_a"], 0)
        self.assertEqual(diagnostics["max_points"], 2)

    def test_semantic_coverage_mode_uses_dense_nms_selection_and_dense_probability_map(self) -> None:
        detection = DaDDetections(
            dense_scores=np.zeros((16, 16), dtype=np.float32),
            nms_points=np.array([[2.0, 2.0], [12.0, 12.0]], dtype=np.float32),
            nms_scores=np.array([2.0, 1.0], dtype=np.float32),
            dense_logits=np.zeros((16, 16), dtype=np.float32),
            feature_map=np.zeros((8, 8, 32), dtype=np.float32),
        )

        selector = Mock(return_value=(torch.tensor([[0.0, 0.0]]), torch.tensor([1.0])))
        runner = V05Runner(
            config=type("Config", (), {"image_size": 16, "patch_rows": 16, "patch_cols": 16, "total_points": 1})(),
            dino=lambda images: images,
            semantic_encoder=lambda tokens_a, tokens_b: (
                torch.zeros(1, 16 * 16, 128),
                torch.zeros(1, 16 * 16, 128),
                None,
                None,
            ),
            detector=type("Detector", (), {"detect": lambda _, __: detection})(),
            fusion=SemanticCoverageDaDFusion(),
            device=torch.device("cpu"),
            point_selector=selector,
            fusion_mode="semantic_coverage",
            candidate_pool_size=2,
            border_margin_px=0,
        )
        image = np.zeros((3, 16, 16), dtype=np.float32)
        tokens = torch.zeros(2, 16 * 16, 128)
        with (
            patch("pasm_v0.infer_v05.pool_dino_tokens", return_value=tokens),
        ):
            points_a, _, scores_a, _, diagnostics = runner.select(image, image)

        self.assertEqual(points_a.shape, (1, 2))
        self.assertEqual(scores_a.shape, (1,))
        self.assertEqual(diagnostics["fused_score_map_a"].shape, (16, 16))
        self.assertGreater(int(np.count_nonzero(diagnostics["fused_score_map_a"])), 2)
        selector.assert_called()
    def test_candidate_score_map_is_finite_for_fixed_candidate_logits(self) -> None:
        base = torch.zeros(1, 1, 16, 16)
        candidates = torch.tensor([[4.0, 4.0], [8.0, 8.0]])
        logits = torch.tensor([1.0, 2.0], requires_grad=True)

        score_map = _candidate_score_map(base, candidates, logits)

        self.assertEqual(score_map.shape, (16, 16))
        self.assertTrue(torch.isfinite(torch.from_numpy(score_map)).all())


if __name__ == "__main__":
    unittest.main()
