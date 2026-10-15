"""The CaR solver and ray decoder must use identical [u, v] patch centers."""

from __future__ import annotations

import unittest

import torch

from neuloc_lvsm.model import _camera_rays, _patch_centers
from train import _query_camera_rays


class PatchCoordinateConventionTest(unittest.TestCase):
    def test_solver_camera_rays_match_decoder_camera_rays(self) -> None:
        intrinsics = torch.tensor([[[12.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]])
        pixels = _patch_centers(16, 16, 8, intrinsics.device, intrinsics.dtype)
        decoder_rays = _camera_rays(pixels.unsqueeze(0), intrinsics)
        solver_rays = _query_camera_rays(intrinsics, 16, 8, count=4)
        self.assertTrue(torch.allclose(decoder_rays, solver_rays))


if __name__ == "__main__":
    unittest.main()
