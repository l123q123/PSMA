import unittest

import numpy as np

from pasm_v0.geometry import PairGeometry
from pasm_v0.point_teacher import build_candidate_teacher


class CandidatePointTeacherTests(unittest.TestCase):
    def test_exact_visible_candidate_receives_unit_soft_label(self) -> None:
        labels, visible, distances = build_candidate_teacher(
            np.array([[10.0, 10.0]], dtype=np.float32),
            np.array([[10.0, 10.0]], dtype=np.float32),
            self._identity_geometry(),
            "a_to_b",
            sigma_px=3.0,
            search_radius_px=12.0,
            depth_tolerance=0.01,
        )

        np.testing.assert_allclose(labels, np.array([1.0], dtype=np.float32))
        np.testing.assert_array_equal(visible, np.array([True]))
        np.testing.assert_allclose(distances, np.array([0.0], dtype=np.float32))

    def test_visible_candidate_outside_search_radius_receives_zero_label(self) -> None:
        labels, visible, distances = build_candidate_teacher(
            np.array([[10.0, 10.0]], dtype=np.float32),
            np.array([[30.0, 10.0]], dtype=np.float32),
            self._identity_geometry(),
            "a_to_b",
            sigma_px=3.0,
            search_radius_px=6.0,
            depth_tolerance=0.01,
        )

        np.testing.assert_allclose(labels, np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(visible, np.array([True]))
        self.assertTrue(np.isinf(distances[0]))

    def test_occluded_source_point_receives_zero_label(self) -> None:
        labels, visible, distances = build_candidate_teacher(
            np.array([[10.0, 10.0]], dtype=np.float32),
            np.array([[10.0, 10.0]], dtype=np.float32),
            self._identity_geometry(depth_b=1.0),
            "a_to_b",
            sigma_px=3.0,
            search_radius_px=12.0,
            depth_tolerance=0.01,
        )

        np.testing.assert_allclose(labels, np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(visible, np.array([False]))
        self.assertTrue(np.isinf(distances[0]))

    def _identity_geometry(self, depth_b: float = 2.0) -> PairGeometry:
        depth_a = np.full((32, 32), 2.0, dtype=np.float32)
        return PairGeometry(
            depth_a=depth_a,
            depth_b=np.full((32, 32), depth_b, dtype=np.float32),
            intrinsics_a=np.eye(3, dtype=np.float64),
            intrinsics_b=np.eye(3, dtype=np.float64),
            cam2world_a=np.eye(4, dtype=np.float64),
            cam2world_b=np.eye(4, dtype=np.float64),
            original_size_a=(32, 32),
            original_size_b=(32, 32),
            image_size=32,
        )


if __name__ == "__main__":
    unittest.main()
