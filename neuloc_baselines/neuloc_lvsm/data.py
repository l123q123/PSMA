"""Episode dataset compatible with NeuLoc JSON manifests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import normalize_support_and_query


STAGE_TO_ID = {"global": 0, "semi": 1, "dense": 2, "fine": 3}


def stage_to_id(stage: object) -> int:
    """Encode rollout stage in the batch without changing old manifests."""
    return STAGE_TO_ID.get(str(stage).lower(), 0)


class EpisodeDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 128,
        support_views: int = 4,
        max_support_views: int | None = None,
    ) -> None:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.episodes = payload.get("episodes", payload)
        if not isinstance(self.episodes, list) or not self.episodes:
            raise ValueError("manifest must contain a non-empty episode list")
        if support_views < 0:
            raise ValueError("support_views must be non-negative; zero means use episode count")
        if max_support_views is not None and max_support_views < 2:
            raise ValueError("max_support_views must be at least two")
        self.image_size = image_size
        self.support_views = support_views
        self.max_support_views = max_support_views

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[index]
        support_items = episode["support"]
        target_views = self.max_support_views if self.max_support_views is not None else (
            self.support_views if self.support_views > 0 else len(support_items)
        )
        valid_count = int(episode.get("support_count", min(len(support_items), target_views)))
        valid_count = min(valid_count, target_views, len(support_items))
        if valid_count < 2:
            raise ValueError("episode must contain at least two valid support views")
        support = [self._load(item) for item in support_items[:valid_count]]
        query = self._load(episode["query"])
        support_images, support_k, support_pose = zip(*support)
        support_images = torch.stack(support_images)
        support_k = torch.stack(support_k)
        support_pose = torch.stack(support_pose)
        anchor_indices = episode.get("anchor_indices")
        anchor_poses = episode.get("anchor_poses")
        if anchor_poses is not None:
            anchor_poses = torch.as_tensor(anchor_poses, dtype=support_pose.dtype)
        support_pose, query_pose = normalize_support_and_query(
            support_pose,
            query[2],
            anchor_indices=anchor_indices,
            anchor_poses=anchor_poses,
        )
        support_type_ids = torch.ones(valid_count, dtype=torch.long)
        if anchor_indices is not None:
            indices = torch.as_tensor(anchor_indices, dtype=torch.long)
            if torch.any(indices >= valid_count):
                raise ValueError("anchor_indices must refer to valid support views")
            support_type_ids[indices] = 0
        if self.max_support_views is None:
            support_mask = torch.ones(valid_count, dtype=torch.bool)
        else:
            pad = target_views - valid_count
            if pad:
                support_images = torch.cat(
                    (support_images, torch.zeros(pad, *support_images.shape[1:], dtype=support_images.dtype)), dim=0
                )
                identity = torch.eye(4, dtype=support_pose.dtype).unsqueeze(0).expand(pad, -1, -1)
                support_pose = torch.cat((support_pose, identity), dim=0)
                identity_k = torch.eye(3, dtype=support_k.dtype).unsqueeze(0).expand(pad, -1, -1)
                support_k = torch.cat((support_k, identity_k), dim=0)
                support_type_ids = torch.cat((support_type_ids, torch.ones(pad, dtype=torch.long)), dim=0)
            support_mask = torch.zeros(target_views, dtype=torch.bool)
            support_mask[:valid_count] = True
        return {
            "support_images": support_images,
            "support_intrinsics": support_k,
            "support_poses": support_pose,
            "query_image": query[0],
            "query_intrinsics": query[1],
            "query_pose": query_pose,
            "support_type_ids": support_type_ids,
            "support_mask": support_mask,
            "view_token_count": torch.tensor(int(episode.get("view_token_count", 0)), dtype=torch.long),
            "support_token_budget": torch.tensor(int(episode.get("support_token_budget", 0)), dtype=torch.long),
            "stage_id": torch.tensor(stage_to_id(episode.get("stage", "global")), dtype=torch.long),
        }

    def _load(self, item: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = Path(item.get("image", item.get("image_path")))
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        if "w2c" in item and "fxfycxcy" in item:
            w2c = torch.as_tensor(item["w2c"], dtype=torch.float32)
            if w2c.shape == (3, 4):
                w2c = torch.cat((w2c, w2c.new_tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)
            if w2c.shape != (4, 4):
                raise ValueError("LVSM w2c must be [3, 4] or [4, 4]")
            pose = torch.linalg.inv(w2c)
            fx, fy, cx, cy = map(float, item["fxfycxcy"])
            intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        elif "c2w" in item and "intrinsics" in item:
            pose = torch.as_tensor(item["c2w"], dtype=torch.float32)
            if pose.shape == (3, 4):
                pose = torch.cat((pose, pose.new_tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)
            if pose.shape != (4, 4):
                raise ValueError("TartanAir c2w must be [3, 4] or [4, 4]")
            raw_intrinsics = torch.as_tensor(item["intrinsics"], dtype=torch.float32)
            if raw_intrinsics.shape == (4,):
                fx, fy, cx, cy = raw_intrinsics.tolist()
                intrinsics = torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
                )
            elif raw_intrinsics.shape == (3, 3):
                intrinsics = raw_intrinsics.clone()
            else:
                raise ValueError("intrinsics must be [fx, fy, cx, cy] or [3, 3]")
        else:
            from safetensors.torch import load_file
            camera_path = Path(item.get("camera", item.get("camera_path")))
            camera = load_file(str(camera_path))
            pose = camera["cam2world"].float()
            intrinsics = camera["intrinsics"].float().clone()
        intrinsics[0] *= self.image_size / width
        intrinsics[1] *= self.image_size / height
        return rgb, intrinsics, pose
