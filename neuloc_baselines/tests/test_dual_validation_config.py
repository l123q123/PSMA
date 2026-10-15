from __future__ import annotations

import unittest

from train import build_parser


class DualValidationConfigTest(unittest.TestCase):
    def test_parser_accepts_seen_and_unseen_validation_manifests(self):
        args = build_parser().parse_args([
            "--manifest", "train.json",
            "--val-manifest", "seen.json",
            "--unseen-val-manifest", "unseen.json",
            "--unseen-val-max-episodes", "128",
        ])
        self.assertEqual(args.unseen_val_manifest, "unseen.json")
        self.assertEqual(args.unseen_val_max_episodes, 128)


if __name__ == "__main__":
    unittest.main()
