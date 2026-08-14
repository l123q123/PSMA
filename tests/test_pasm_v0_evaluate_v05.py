import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pasm_v0.dad import DaDDetections
from pasm_v0.evaluate_v05 import evaluate_pairs
from pasm_v0.infer_v05 import V05Runner, load_v05_runner


class _FakeDino(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros((images.shape[0], 4 * 4, 64), device=images.device)


class _FakeSemanticEncoder(torch.nn.Module):
    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor):
        semantic_a = torch.zeros((1, tokens_a.shape[1], 128), device=tokens_a.device)
        semantic_b = torch.zeros((1, tokens_b.shape[1], 128), device=tokens_b.device)
        values = torch.zeros((1, tokens_a.shape[1]), device=tokens_a.device)
        return semantic_a, semantic_b, values, values


class _FakeDetector:
    def detect(self, image: np.ndarray) -> DaDDetections:
        return DaDDetections(
            dense_scores=np.full((16, 16), 1.0 / 256.0, dtype=np.float32),
            nms_points=np.empty((0, 2), dtype=np.float32),
            nms_scores=np.empty((0,), dtype=np.float32),
            dense_logits=np.zeros((16, 16), dtype=np.float32),
            feature_map=np.zeros((8, 8, 32), dtype=np.float32),
        )


class _FakeFusion(torch.nn.Module):
    def forward(
        self, dad_features: torch.Tensor, base_logits: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> torch.Tensor:
        return base_logits


def _selector(logits: torch.Tensor, num_points: int):
    return torch.zeros((num_points, 2), dtype=logits.dtype), torch.ones((num_points,), dtype=logits.dtype)


class V05RunnerTests(unittest.TestCase):
    def test_runner_selects_1024_points_without_geometry(self) -> None:
        runner = V05Runner(
            config=SimpleNamespace(image_size=16, patch_rows=4, patch_cols=4, total_points=1024),
            dino=_FakeDino(),
            semantic_encoder=_FakeSemanticEncoder(),
            detector=_FakeDetector(),
            fusion=_FakeFusion(),
            device=torch.device("cpu"),
            point_selector=_selector,
        )
        image = np.zeros((3, 16, 16), dtype=np.float32)

        points_a, points_b, scores_a, scores_b, diagnostics = runner.select(image, image)

        self.assertEqual(points_a.shape, (1024, 2))
        self.assertEqual(points_b.shape, (1024, 2))
        self.assertEqual(scores_a.shape, (1024,))
        self.assertEqual(scores_b.shape, (1024,))
        self.assertFalse(diagnostics["used_geometry"])
        self.assertNotIn("geometry", diagnostics)

    def test_loader_rejects_checkpoint_without_v05_format(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            torch.save({"format": "not_v05"}, checkpoint)

            with self.assertRaisesRegex(ValueError, "v0.5"):
                load_v05_runner(
                    checkpoint, "missing_dino.pth", "missing_dad.pth", "missing_semantic.pt", "cpu"
                )


class _FakeV05Runner:
    config = SimpleNamespace(image_size=16, total_points=1024, geometry_depth_tolerance=0.05)

    def __init__(self) -> None:
        self.calls = 0

    def select(self, image_a: np.ndarray, image_b: np.ndarray):
        self.calls += 1
        points = np.zeros((1024, 2), dtype=np.float32)
        scores = np.ones((1024,), dtype=np.float32)
        diagnostics = {
            "used_geometry": False,
            "fused_score_map_a": np.full((16, 16), 1.0 / 256.0, dtype=np.float32),
            "fused_score_map_b": np.full((16, 16), 1.0 / 256.0, dtype=np.float32),
            "base_score_map_a": np.full((16, 16), 1.0 / 256.0, dtype=np.float32),
            "base_score_map_b": np.full((16, 16), 1.0 / 256.0, dtype=np.float32),
        }
        return points, points.copy(), scores, scores.copy(), diagnostics


class _FakeMatcher:
    def match(self, image_a, points_a, image_b, points_b):
        return np.array([0]), np.array([0]), np.array([0.9], dtype=np.float32)


class V05EvaluationTests(unittest.TestCase):
    def test_evaluator_accepts_custom_pair_manifest(self) -> None:
        pairs = [SimpleNamespace(pair_id=f"pair_{index}", image_a=Path("a.jpg"), image_b=Path("b.jpg")) for index in range(2)]
        runner = _FakeV05Runner()
        image = np.zeros((3, 16, 16), dtype=np.float32)

        with TemporaryDirectory() as directory, \
             patch("pasm_v0.evaluate_v05.load_manifest", return_value=pairs), \
             patch("pasm_v0.evaluate_v05.validate_heldout_against_training"), \
             patch("pasm_v0.evaluate_v05.load_v05_runner", return_value=runner), \
             patch("pasm_v0.evaluate_v05.FrozenLoMaMatcher", return_value=_FakeMatcher()), \
             patch("pasm_v0.evaluate_v05.load_resized_rgb", return_value=image), \
             patch("pasm_v0.evaluate_v05.load_pair_geometry", return_value=object()), \
             patch("pasm_v0.evaluate_v05._metric_row", side_effect=lambda index, pair_id, method, *_: {
                 "pair_index": index, "pair_id": pair_id, "method": method, "mutual_matches": 1,
                 "inliers_3px": 1, "inliers_6px": 1, "inlier_rate_3px": 1.0, "inlier_rate_6px": 1.0,
                 "coverage_3px": 1.0, "coverage_6px": 1.0, "high_confidence_matches": 1,
                 "high_confidence_inliers_3px": 1, "high_confidence_inliers_6px": 1,
                 "high_confidence_inlier_rate_3px": 1.0, "high_confidence_inlier_rate_6px": 1.0,
             }):
            output = evaluate_pairs(
                "images", "training.json", "dino.pth", "loma.pth", "dad.pth", "fusion.pt",
                "semantic.pt", directory, pair_manifest="custom.json", device="cpu", max_pairs=9,
            )

            with (output / "summary.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            self.assertEqual(runner.calls, 2)

    def test_evaluator_writes_nine_rows_and_budget_free_visualizations(self) -> None:
        pairs = [SimpleNamespace(pair_id=f"pair_{index}", image_a=Path("a.jpg"), image_b=Path("b.jpg")) for index in range(9)]
        runner = _FakeV05Runner()
        image = np.zeros((3, 16, 16), dtype=np.float32)

        with TemporaryDirectory() as directory, \
             patch("pasm_v0.evaluate_v05.load_heldout_pairs", return_value=pairs), \
             patch("pasm_v0.evaluate_v05.validate_heldout_against_training"), \
             patch("pasm_v0.evaluate_v05.load_v05_runner", return_value=runner), \
             patch("pasm_v0.evaluate_v05.FrozenLoMaMatcher", return_value=_FakeMatcher()), \
             patch("pasm_v0.evaluate_v05.load_resized_rgb", return_value=image), \
             patch("pasm_v0.evaluate_v05.load_pair_geometry", return_value=object()), \
             patch("pasm_v0.evaluate_v05._metric_row", side_effect=lambda index, pair_id, method, *_: {
                 "pair_index": index, "pair_id": pair_id, "method": method, "mutual_matches": 1,
                 "inliers_3px": 1, "inliers_6px": 1, "inlier_rate_3px": 1.0, "inlier_rate_6px": 1.0,
                 "coverage_3px": 1.0, "coverage_6px": 1.0, "high_confidence_matches": 1,
                 "high_confidence_inliers_3px": 1, "high_confidence_inliers_6px": 1,
                 "high_confidence_inlier_rate_3px": 1.0, "high_confidence_inlier_rate_6px": 1.0,
             }):
            output = evaluate_pairs(
                "images", "training.json", "dino.pth", "loma.pth", "dad.pth", "fusion.pt",
                "semantic.pt", directory, device="cpu", max_pairs=9,
            )

            with (output / "summary.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 9)
            with (output / "aggregate.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            self.assertEqual(runner.calls, 9)
            for index in range(1, 10):
                pair_output = output / f"{index:02d}_pair_{index - 1}"
                for filename in ("points.png", "matches.png", "fused_scores_a.png", "fused_scores_b.png"):
                    self.assertTrue((pair_output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
