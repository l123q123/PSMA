"""Pose and ray geometry shared by the LVSM-NeuLoc pipeline."""

from __future__ import annotations

from typing import Sequence

import torch


def _anchor_transform(anchor_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one trajectory frame and scale from the fixed anchor poses."""
    if anchor_poses.ndim != 3 or anchor_poses.shape[-2:] != (4, 4):
        raise ValueError("anchor_poses must be [K, 4, 4]")
    if anchor_poses.shape[0] < 2:
        raise ValueError("at least two anchor poses are required")

    centers = anchor_poses[:, :3, 3]
    origin = centers.mean(dim=0)
    rotations = anchor_poses[:, :3, :3]
    forward = torch.nn.functional.normalize(rotations[:, :, 2].mean(dim=0), dim=0)
    down = rotations[:, :, 1].mean(dim=0)
    right = torch.nn.functional.normalize(torch.cross(down, forward, dim=0), dim=0)
    down = torch.nn.functional.normalize(torch.cross(forward, right, dim=0), dim=0)
    frame = torch.eye(4, dtype=anchor_poses.dtype, device=anchor_poses.device)
    frame[:3, :3] = torch.stack((right, down, forward), dim=1)
    frame[:3, 3] = origin
    world_to_anchor = torch.linalg.inv(frame)

    pairwise = torch.pdist(centers)
    nonzero = pairwise[pairwise > 1e-6]
    scale = (nonzero.median() if nonzero.numel() else centers.new_tensor(1.0)).clamp_min(1e-6)
    return world_to_anchor, scale


def normalize_support_and_query(
    support_poses: torch.Tensor,
    query_pose: torch.Tensor,
    anchor_indices: Sequence[int] | torch.Tensor | None = None,
    anchor_poses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize an episode using a fixed trajectory anchor when provided.

    New anchor manifests pass four support indices that are fixed for the whole
    trajectory. Legacy manifests keep the historical first-support behavior.
    The query is never used to construct the transform or scale.
    """
    if support_poses.ndim != 3 or support_poses.shape[-2:] != (4, 4):
        raise ValueError("support_poses must be [M, 4, 4]")
    if query_pose.shape != (4, 4):
        raise ValueError("query_pose must be [4, 4]")
    if support_poses.shape[0] < 2:
        raise ValueError("at least two support poses are required")

    if anchor_poses is not None:
        if anchor_poses.ndim != 3 or anchor_poses.shape[-2:] != (4, 4) or anchor_poses.shape[0] != 4:
            raise ValueError("anchor_poses must be [4, 4, 4]")
        world_to_anchor, scale = _anchor_transform(anchor_poses)
        support = world_to_anchor.unsqueeze(0) @ support_poses
        query = world_to_anchor @ query_pose
    elif anchor_indices is None:
        anchor_inverse = torch.linalg.inv(support_poses[0])
        support = anchor_inverse.unsqueeze(0) @ support_poses
        query = anchor_inverse @ query_pose
        centers = support[1:, :3, 3]
        distances = torch.linalg.vector_norm(centers, dim=-1)
        nonzero = distances[distances > 1e-6]
        scale = (nonzero.median() if nonzero.numel() else distances.new_tensor(1.0)).clamp_min(1e-6)
    else:
        indices = torch.as_tensor(anchor_indices, device=support_poses.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() != 4:
            raise ValueError("anchor_indices must contain exactly four support indices")
        if torch.any(indices < 0) or torch.any(indices >= support_poses.shape[0]):
            raise ValueError("anchor_indices are out of range")
        world_to_anchor, scale = _anchor_transform(support_poses.index_select(0, indices))
        support = world_to_anchor.unsqueeze(0) @ support_poses
        query = world_to_anchor @ query_pose
    support = support.clone()
    query = query.clone()
    support[:, :3, 3] /= scale
    query[:3, 3] /= scale
    return support, query
