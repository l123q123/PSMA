from __future__ import annotations

import unittest

import torch

from neuloc_lvsm.model import LVSMNeuLoc


class RayOutputMomentsTest(unittest.TestCase):
    def test_raw_moments_are_exposed_without_new_parameters(self) -> None:
        torch.manual_seed(0)
        model = LVSMNeuLoc(
            image_size=16, patch_size=8, dim=16, latent_count=4,
            encoder_layers=1, decoder_layers=1, heads=4,
        ).eval()
        state_keys_before = set(model.state_dict())

        images = torch.rand(1, 2, 3, 16, 16)
        intrinsics = torch.eye(3).repeat(1, 2, 1, 1)
        intrinsics[..., 0, 0] = 12.0
        intrinsics[..., 1, 1] = 12.0
        intrinsics[..., 0, 2] = 8.0
        intrinsics[..., 1, 2] = 8.0
        poses = torch.eye(4).repeat(1, 2, 1, 1)
        query = torch.rand(1, 3, 16, 16)

        output = model(images, intrinsics, poses, query, intrinsics[:, 0])
        expected = output.moments_raw - (
            output.moments_raw * output.directions
        ).sum(-1, keepdim=True) * output.directions
        self.assertEqual(output.moments_raw.shape, output.moments.shape)
        self.assertTrue(torch.allclose(output.moments, expected))
        self.assertEqual(set(model.state_dict()), state_keys_before)
