import unittest
from argparse import Namespace

import torch

from pasm_v0.adaptive_threshold import (
    AdaptiveScoreThreshold,
    adaptive_threshold_loss,
    select_adaptive_dense_topk,
    threshold_supervision_labels,
)
from pasm_v0.train_adaptive_threshold import train_options_from_args


class AdaptiveThresholdTests(unittest.TestCase):
    def test_threshold_labels_keep_confirmed_invisible_candidates_as_negatives(self) -> None:
        labels = threshold_supervision_labels(
            torch.tensor([0.8, 0.0, 0.4]),
            torch.tensor([True, False, False]),
            torch.tensor([False, True, False]),
        )

        torch.testing.assert_close(labels, torch.tensor([0.8, 0.0, -1.0]))

    def test_cli_options_map_manifest_and_output_to_training_arguments(self) -> None:
        options = train_options_from_args(Namespace(manifest="pairs.json", output="threshold.pt", epochs=1))

        self.assertEqual(options["manifest_path"], "pairs.json")
        self.assertEqual(options["output_path"], "threshold.pt")
        self.assertNotIn("manifest", options)
        self.assertNotIn("output", options)

    def test_threshold_loss_learns_to_keep_high_score_reliable_candidates(self) -> None:
        dense_logits = torch.zeros((1, 1, 8, 8))
        dense_logits[0, 0, 2, 2] = 4.0
        dense_logits[0, 0, 2, 5] = -4.0
        semantic = torch.ones((1, 4, 128))
        candidates = torch.tensor([[2.0, 2.0], [5.0, 2.0]])
        labels = torch.tensor([1.0, 0.0])
        head = AdaptiveScoreThreshold()

        loss, threshold = adaptive_threshold_loss(
            dense_logits, semantic, candidates, labels, head, image_size=8, temperature=1.0, max_candidates=2,
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(threshold.shape, (1,))
        self.assertGreater(float(head.output.weight.grad.abs().sum()), 0.0)

    def test_adaptive_selection_can_keep_different_point_counts_per_view(self) -> None:
        dense_logits = torch.zeros((1, 1, 8, 8))
        dense_logits[0, 0, 2, 2] = 4.0
        dense_logits[0, 0, 2, 5] = 0.0
        points = torch.tensor([[-0.375, -0.375], [0.375, -0.375]])
        scores = torch.tensor([0.9, 0.4])

        class MeanThreshold(torch.nn.Module):
            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return tokens.mean(dim=(1, 2))

        selector = lambda _, __: (points, scores)
        head = MeanThreshold()
        many_points, _, many_threshold = select_adaptive_dense_topk(
            dense_logits, torch.full((1, 4, 128), -1.0), head, 2, selector,
        )
        few_points, _, few_threshold = select_adaptive_dense_topk(
            dense_logits, torch.full((1, 4, 128), 1.0), head, 2, selector,
        )

        self.assertEqual(len(many_points), 2)
        self.assertEqual(len(few_points), 1)
        self.assertLess(float(many_threshold), float(few_threshold))



if __name__ == "__main__":
    unittest.main()
