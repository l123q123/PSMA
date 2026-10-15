from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from neuloc_lvsm.data import EpisodeDataset


class TokenBudgetDatasetTest(unittest.TestCase):
    def test_zero_support_views_uses_episode_count_and_reads_token_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
            pose = np.eye(4, dtype=np.float32).tolist()
            item = {"image": str(image), "c2w": pose, "intrinsics": [4.0, 4.0, 4.0, 4.0]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "episodes": [{
                    "support": [item, item, item],
                    "query": item,
                    "support_count": 3,
                    "view_token_count": 160,
                    "support_token_budget": 8192,
                    "anchor_poses": [pose, pose, pose, pose],
                    "anchor_frame_ids": [0, 1, 2, 3],
                }]
            }), encoding="utf-8")
            sample = EpisodeDataset(manifest, image_size=8, support_views=0)[0]
            self.assertEqual(tuple(sample["support_images"].shape[:1]), (3,))
            self.assertEqual(int(sample["view_token_count"]), 160)
            self.assertEqual(int(sample["support_token_budget"]), 8192)


if __name__ == "__main__":
    unittest.main()
