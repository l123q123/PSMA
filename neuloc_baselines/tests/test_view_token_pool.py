from __future__ import annotations

import unittest

import torch

from neuloc_lvsm.model import ViewTokenPool


class ViewTokenPoolTest(unittest.TestCase):
    def test_identity_when_target_matches_patch_count(self) -> None:
        tokens = torch.randn(2, 3, 16, 8)
        pool = ViewTokenPool(dim=8, patch_count=16, target_tokens=16)
        output = pool(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.equal(output, tokens))

    def test_pools_each_view_without_mixing_views(self) -> None:
        tokens = torch.randn(2, 3, 16, 8, requires_grad=True)
        pool = ViewTokenPool(dim=8, patch_count=16, target_tokens=4)
        output = pool(tokens)
        self.assertEqual(output.shape, (2, 3, 4, 8))
        output.square().mean().backward()
        self.assertIsNotNone(tokens.grad)

    def test_model_uses_view_pool_and_token_budget(self) -> None:
        from neuloc_lvsm.model import LVSMNeuLoc

        model = LVSMNeuLoc(
            image_size=16,
            patch_size=8,
            dim=16,
            latent_count=4,
            encoder_layers=1,
            decoder_layers=1,
            heads=4,
            pool_token_counts=(2, 4),
        )
        images = torch.rand(1, 3, 3, 16, 16)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(1, 3, 1, 1)
        poses = torch.eye(4).unsqueeze(0).repeat(1, 3, 1, 1)
        query = torch.rand(1, 3, 16, 16)
        query_k = torch.eye(3).unsqueeze(0)
        result = model(
            images,
            intrinsics,
            poses,
            query,
            query_k,
            view_token_count=2,
            support_token_budget=8,
        )
        self.assertEqual(result.latent.shape, (1, 4, 16))
        self.assertEqual(result.directions.shape, (1, 4, 3))
        self.assertEqual(result.moments.shape, (1, 4, 3))


if __name__ == "__main__":
    unittest.main()
