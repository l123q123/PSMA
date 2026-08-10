import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pasm_v0.evaluate_pose import (
    evaluate_saved_matches,
    estimate_relative_pose,
    pose_auc,
    rotation_error_degrees,
    translation_error_degrees,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    camera = points @ rotation.T + translation
    normalized = camera[:, :2] / camera[:, 2:3]
    return normalized * np.array((intrinsics[0, 0], intrinsics[1, 1])) + intrinsics[:2, 2]


class RelativePoseMetricTests(unittest.TestCase):
    def test_pose_recovery_from_noise_free_calibrated_correspondences(self) -> None:
        rng = np.random.default_rng(7)
        intrinsics = np.array(((500.0, 0.0, 256.0), (0.0, 510.0, 256.0), (0.0, 0.0, 1.0)))
        angle = np.deg2rad(8.0)
        rotation = np.array(((np.cos(angle), 0.0, np.sin(angle)), (0.0, 1.0, 0.0), (-np.sin(angle), 0.0, np.cos(angle))))
        translation = np.array((0.35, 0.02, 0.05))
        world = np.column_stack((rng.uniform(-1.0, 1.0, 80), rng.uniform(-0.7, 0.7, 80), rng.uniform(4.0, 7.0, 80)))
        points_a = _project(world, np.eye(3), np.zeros(3), intrinsics)
        points_b = _project(world, rotation, translation, intrinsics)

        estimate = estimate_relative_pose(points_a, points_b, intrinsics, intrinsics)

        self.assertTrue(estimate.success)
        self.assertGreaterEqual(estimate.ransac_inliers, 5)
        self.assertLess(rotation_error_degrees(estimate.rotation, rotation), 0.1)
        self.assertLess(translation_error_degrees(estimate.translation, translation), 0.1)

    def test_pose_angle_metrics_handle_identical_rotation_and_essential_sign_ambiguity(self) -> None:
        identity = np.eye(3)

        self.assertAlmostEqual(rotation_error_degrees(identity, identity), 0.0, places=6)
        self.assertAlmostEqual(translation_error_degrees(np.array((1.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0))), 0.0, places=6)

    def test_pose_auc_keeps_explicit_failed_pairs_in_denominator(self) -> None:
        values = pose_auc(np.array((0.0, 180.0)), thresholds=(5.0, 10.0))

        self.assertAlmostEqual(values[5.0], 0.5, places=6)
        self.assertAlmostEqual(values[10.0], 0.5, places=6)

    def test_pose_recovery_rejects_opencv_result_with_fewer_than_five_pose_inliers(self) -> None:
        intrinsics = np.eye(3)
        points = np.array(((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (0.2, 0.2)))
        import cv2
        with patch.object(cv2, "findEssentialMat", return_value=(np.eye(3), np.ones((5, 1), dtype=np.uint8))), \
             patch.object(cv2, "recoverPose", return_value=(2, np.eye(3), np.array(((1.0,), (0.0,), (0.0,))), np.array(((1,), (1,), (0,), (0,), (0,)), dtype=np.uint8))):
            estimate = estimate_relative_pose(points, points, intrinsics, intrinsics)

        self.assertFalse(estimate.success)
        self.assertEqual(estimate.ransac_inliers, 0)

    def test_saved_match_evaluator_writes_per_pair_and_aggregate_pose_csv(self) -> None:
        intrinsics = np.array(((500.0, 0.0, 256.0), (0.0, 500.0, 256.0), (0.0, 0.0, 1.0)))
        rotation = np.eye(3)
        translation = np.array((0.3, 0.0, 0.0))
        rng = np.random.default_rng(8)
        world = np.column_stack((rng.uniform(-1.0, 1.0, 20), rng.uniform(-1.0, 1.0, 20), rng.uniform(4.0, 6.0, 20)))
        points_a = _project(world, np.eye(3), np.zeros(3), intrinsics)
        points_b = _project(world, rotation, translation, intrinsics)
        target_from_source = np.eye(4)
        target_from_source[:3, :3] = rotation
        target_from_source[:3, 3] = translation
        pair = SimpleNamespace(pair_id="pair_a")
        geometry = SimpleNamespace(
            intrinsics_a=intrinsics,
            intrinsics_b=intrinsics,
            cam2world_a=np.eye(4),
            cam2world_b=np.linalg.inv(target_from_source),
        )

        with TemporaryDirectory() as directory:
            match_root = Path(directory) / "matches"
            result_dir = match_root / "01_pair_a" / "dad_global"
            result_dir.mkdir(parents=True)
            np.savez_compressed(result_dir / "result.npz", keypoints_a=points_a, keypoints_b=points_b)
            output = Path(directory) / "pose"
            with patch("pasm_v0.evaluate_pose.load_heldout_pairs", return_value=[pair]), \
                 patch("pasm_v0.evaluate_pose.load_pair_geometry", return_value=geometry):
                evaluate_saved_matches(
                    "images", match_root, output, image_size=512, max_pairs=1,
                    result_subdir="dad_global",
                )

            with (output / "pose_summary.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            with (output / "pose_aggregate.csv").open(encoding="utf-8", newline="") as file:
                aggregate = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["success"], "True")
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(aggregate[0]["pairs"], "1")


if __name__ == "__main__":
    unittest.main()
