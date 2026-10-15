import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from train import (
    STAGE_ORDER,
    apply_full_lvsm_ray_preset,
    _progress_postfix_train,
    _progress_postfix_val,
    build_parser,
    stage_indices,
    stagewise_validation_score,
    train_loader_kwargs,
    validation_score,
)


class ValidationConfigTest(unittest.TestCase):
    def test_parser_accepts_validation_manifest_and_frequency(self):
        args = build_parser().parse_args([
            "--manifest", "train.json",
            "--val-manifest", "val.json",
            "--val-every", "2",
            "--val-max-episodes", "128",
        ])
        self.assertEqual(args.val_manifest, "val.json")
        self.assertEqual(args.val_every, 2)
        self.assertEqual(args.val_max_episodes, 128)

    def test_full_preset_preserves_zero_support_views_for_token_budget_manifest(self):
        args = build_parser().parse_args([
            "--manifest", "train.json", "--full-lvsm-ray",
            "--official-lvsm-checkpoint", "official.pt",
            "--ray-checkpoint", "ray.pt", "--support-views", "0",
        ])
        self.assertEqual(apply_full_lvsm_ray_preset(args).support_views, 0)

    def test_loader_kwargs_enable_persistent_prefetch_workers(self):
        kwargs = train_loader_kwargs(num_workers=4)
        self.assertEqual(kwargs["num_workers"], 4)
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 2)

    def test_parser_accepts_validation_batch_size(self):
        args = build_parser().parse_args([
            "--manifest", "train.json",
            "--val-batch-size", "3",
        ])
        self.assertEqual(args.val_batch_size, 3)

    def test_validation_score_prefers_lower_pose_error(self):
        better = {"rotation_deg": 4.0, "center_error": 0.2}
        worse = {"rotation_deg": 5.0, "center_error": 0.2}
        self.assertLess(validation_score(better), validation_score(worse))

    def test_progress_postfix_reports_current_and_running_average(self):
        self.assertEqual(_progress_postfix_train(0.25, 0.75), {"loss": "0.2500", "avg_loss": "0.7500"})
        self.assertEqual(_progress_postfix_val({"rotation_deg": 12.34567, "center_error": 1.23456}), {
            "rot": "12.346deg",
            "center": "1.2346",
        })


    def test_stage_indices_keep_canonical_stage_order_and_limit_per_stage(self):
        class Dataset:
            episodes = [
                {"stage": "fine"}, {"stage": "global"}, {"stage": "dense"},
                {"stage": "semi"}, {"stage": "global"}, {"stage": "fine"},
            ]

        grouped = stage_indices(Dataset(), max_episodes=1)
        self.assertEqual(tuple(grouped), STAGE_ORDER)
        self.assertEqual(grouped, {
            "global": [1], "semi": [3], "dense": [2], "fine": [0],
        })

    def test_stagewise_score_gives_equal_weight_to_each_present_stage(self):
        scores = {
            "global": {"rotation_deg": 10.0, "center_error": 2.0},
            "fine": {"rotation_deg": 2.0, "center_error": 0.0},
        }
        self.assertEqual(stagewise_validation_score(scores), 7.0)

if __name__ == "__main__":
    unittest.main()
