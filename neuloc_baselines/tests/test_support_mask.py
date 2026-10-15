from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from neuloc_lvsm.data import EpisodeDataset
from neuloc_lvsm.evaluate import predict
from neuloc_lvsm.model import LVSMNeuLoc, RayOutput
from neuloc_lvsm.solver import CarPose


class SupportMaskTest(unittest.TestCase):
    def _inputs(self, views=20, size=16):
        images = torch.rand(1, views, 3, size, size)
        intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, views, 1, 1)
        intrinsics[..., 0, 0] = 12.0
        intrinsics[..., 1, 1] = 12.0
        intrinsics[..., 0, 2] = size / 2
        intrinsics[..., 1, 2] = size / 2
        poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
        poses[..., 0, 3] = torch.arange(views).float()
        query = torch.rand(1, 3, size, size)
        query_k = intrinsics[:, 0]
        return images, intrinsics, poses, query, query_k

    def test_masked_support_tokens_do_not_change_latent(self):
        torch.manual_seed(3)
        model = LVSMNeuLoc(image_size=16, patch_size=8, dim=32, latent_count=4, encoder_layers=1, decoder_layers=1, heads=4)
        model.eval()
        images, intrinsics, poses, query, query_k = self._inputs()
        mask = torch.zeros(1, 20, dtype=torch.bool)
        mask[:, :4] = True
        changed = images.clone()
        changed[:, 4:] = torch.rand_like(changed[:, 4:]) * 100.0
        with torch.no_grad():
            a = model(images, intrinsics, poses, query, query_k, support_mask=mask)
            b = model(changed, intrinsics, poses, query, query_k, support_mask=mask)
        self.assertTrue(torch.allclose(a.latent, b.latent, atol=1e-6, rtol=1e-6))

    def test_dataset_pads_to_max_support_slots_and_returns_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def write(name, value):
                Image.new("RGB", (16, 16), color=(value, value, value)).save(root / name)
                return str(root / name)
            support = []
            for i in range(4):
                support.append({"image": write(f"s{i}.png", i), "c2w": torch.eye(4).tolist(), "intrinsics": [12.0, 12.0, 8.0, 8.0]})
            query = {"image": write("q.png", 9), "c2w": torch.eye(4).tolist(), "intrinsics": [12.0, 12.0, 8.0, 8.0]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"episodes": [{"support": support, "query": query}]}), encoding="utf-8")
            item = EpisodeDataset(manifest, image_size=16, support_views=8, max_support_views=20)[0]
        self.assertEqual(tuple(item["support_images"].shape), (20, 3, 16, 16))
        self.assertEqual(tuple(item["support_mask"].shape), (20,))
        self.assertEqual(int(item["support_mask"].sum()), 4)
        self.assertTrue(torch.all(item["support_images"][4:] == 0))

    def test_evaluation_forwards_support_mask_to_model(self):
        class RecordingModel:
            patch_size = 8

            def __call__(self, *args, **kwargs):
                self.support_mask = kwargs.get("support_mask")
                directions = torch.tensor([[[1.0, 0.0, 0.0]] * 4])
                moments = torch.zeros_like(directions)
                return RayOutput(latent=torch.zeros(1, 1, 2), directions=directions, moments=moments)

        batch = {
            "support_images": torch.zeros(1, 20, 3, 16, 16),
            "support_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 20, 1, 1),
            "support_poses": torch.eye(4).view(1, 1, 4, 4).repeat(1, 20, 1, 1),
            "support_type_ids": torch.ones(1, 20, dtype=torch.long),
            "support_mask": torch.tensor([[True] * 4 + [False] * 16]),
            "query_image": torch.zeros(1, 3, 16, 16),
            "query_intrinsics": torch.eye(3).unsqueeze(0),
            "query_pose": torch.eye(4).unsqueeze(0),
        }
        model = RecordingModel()
        fake_pose = CarPose(
            rotation=torch.eye(3).unsqueeze(0),
            center=torch.zeros(1, 3),
            translation=torch.zeros(1, 3),
        )
        with patch("neuloc_lvsm.evaluate.solve_car_pose", return_value=fake_pose):
            predict(model, batch, image_size=16)
        self.assertTrue(torch.equal(model.support_mask, batch["support_mask"]))


if __name__ == "__main__":
    unittest.main()
