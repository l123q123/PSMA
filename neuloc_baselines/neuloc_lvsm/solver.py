"""Differentiable known-intrinsics CaR pose solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CarPose:
    rotation: torch.Tensor
    center: torch.Tensor
    translation: torch.Tensor


@dataclass(frozen=True)
class RelativePosePrediction:
    """Per-support query pose predictions and their robust global fusion."""

    rotation: torch.Tensor
    translation: torch.Tensor
    confidence_logits: torch.Tensor
    candidate_rotation: torch.Tensor
    candidate_center: torch.Tensor
    weights: torch.Tensor
    fused: CarPose


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    x, y, z = vectors.unbind(-1)
    zeros = torch.zeros_like(x)
    return torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), -1).reshape(*vectors.shape[:-1], 3, 3)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert exponential-map rotations to matrices without a zero-angle singularity."""
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must end with three coordinates")
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    theta2 = theta.square()
    safe_theta = theta.clamp_min(1e-6)
    a_regular = torch.sin(theta) / safe_theta
    b_regular = (1.0 - torch.cos(theta)) / safe_theta.square()
    a = torch.where(theta < 1e-4, 1.0 - theta2 / 6.0 + theta2.square() / 120.0, a_regular)
    b = torch.where(theta < 1e-4, 0.5 - theta2 / 24.0 + theta2.square() / 720.0, b_regular)
    skew = _skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def transform_pluecker_c2w_to_local(
    directions: torch.Tensor, moments: torch.Tensor, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express world-frame Pluecker rays in the local frame of a c2w pose."""
    if directions.shape != moments.shape or directions.ndim != 3 or directions.shape[-1] != 3:
        raise ValueError("directions and moments must both be [B, N, 3]")
    if c2w.shape != (directions.shape[0], 4, 4):
        raise ValueError("c2w must be [B, 4, 4]")
    rotation = c2w[:, :3, :3].float()
    center = c2w[:, :3, 3].float()
    world_directions = directions.float()
    local_directions = (rotation.transpose(-1, -2) @ world_directions.transpose(-1, -2)).transpose(-1, -2)
    translated_moments = moments.float() - torch.cross(center[:, None], world_directions, dim=-1)
    local_moments = (rotation.transpose(-1, -2) @ translated_moments.transpose(-1, -2)).transpose(-1, -2)
    return local_directions, local_moments


def compose_c2w_local_delta(c2w: torch.Tensor, local_delta: torch.Tensor) -> torch.Tensor:
    """Apply a local-frame SE(3) update to a camera-to-world pose."""
    if c2w.shape[-2:] != (4, 4) or local_delta.shape[-2:] != (4, 4):
        raise ValueError("c2w and local_delta must end with [4, 4]")
    return c2w @ local_delta


def _weighted_so3_mean(rotations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Project a weighted matrix average back onto SO(3)."""
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must be [B, views, 3, 3]")
    if weights.shape != rotations.shape[:2]:
        raise ValueError("weights must be [B, views]")
    average = (weights[..., None, None] * rotations).sum(dim=1).float()
    u, _, vh = torch.linalg.svd(average, full_matrices=False)
    correction = torch.eye(3, dtype=average.dtype, device=average.device).expand_as(average).clone()
    correction[..., -1, -1] = torch.where(
        torch.linalg.det(u @ vh) < 0,
        torch.tensor(-1.0, dtype=average.dtype, device=average.device),
        torch.tensor(1.0, dtype=average.dtype, device=average.device),
    )
    return u @ correction @ vh


def _rotation_error(rotation: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    relative = reference.transpose(-1, -2).unsqueeze(1) @ rotation
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


def fuse_relative_pose_candidates(
    support_poses: torch.Tensor,
    relative_rotation: torch.Tensor,
    relative_translation: torch.Tensor,
    confidence_logits: torch.Tensor,
    support_mask: torch.Tensor | None = None,
    robust_center_scale: float = 0.25,
    robust_rotation_scale: float = 0.5235987755982988,
) -> tuple[CarPose, torch.Tensor]:
    """Compose `T(i <- query)` candidates and robustly fuse them in SE(3).

    `support_poses` and the returned pose are c2w.  Relative translations are
    expressed in support-camera coordinates.  The support nodes are known, so
    this is the one-query-variable form of a small pose graph optimization.
    """
    if support_poses.ndim != 4 or support_poses.shape[-2:] != (4, 4):
        raise ValueError("support_poses must be [B, views, 4, 4]")
    b, views = support_poses.shape[:2]
    if relative_rotation.shape != (b, views, 3, 3) or relative_translation.shape != (b, views, 3):
        raise ValueError("relative pose predictions must agree with support views")
    if confidence_logits.shape != (b, views):
        raise ValueError("confidence_logits must be [B, views]")
    if robust_center_scale <= 0 or robust_rotation_scale <= 0:
        raise ValueError("robust scales must be positive")
    if support_mask is None:
        support_mask = torch.ones((b, views), dtype=torch.bool, device=support_poses.device)
    if support_mask.shape != (b, views) or not torch.all(support_mask.any(dim=-1)):
        raise ValueError("each sample needs at least one valid support")
    support_mask = support_mask.to(dtype=torch.bool, device=support_poses.device)

    support_rotation = support_poses[..., :3, :3].float()
    support_center = support_poses[..., :3, 3].float()
    candidate_rotation = support_rotation @ relative_rotation.float()
    candidate_center = support_center + (support_rotation @ relative_translation.float().unsqueeze(-1)).squeeze(-1)
    masked_logits = confidence_logits.float().masked_fill(~support_mask, float("-inf"))
    learned_weights = torch.softmax(masked_logits, dim=-1)
    # Estimate residuals from a neutral consensus first.  Otherwise a single
    # accidentally overconfident prediction could define the initial pose and
    # prevent its own rejection.
    consensus_weights = support_mask.float()
    consensus_weights = consensus_weights / consensus_weights.sum(dim=-1, keepdim=True)
    # Coordinate median has a 50% breakdown point, which is useful here
    # because learned confidence is still uncalibrated at the start of tuning.
    invalid_center = candidate_center.masked_fill(~support_mask[..., None], float("nan"))
    initial_center = torch.nanmedian(invalid_center, dim=1).values
    initial_rotation = _weighted_so3_mean(candidate_rotation, consensus_weights)
    center_residual = torch.linalg.vector_norm(candidate_center - initial_center[:, None], dim=-1)
    rotation_residual = _rotation_error(candidate_rotation, initial_rotation)
    robust = 1.0 / (1.0 + (center_residual / robust_center_scale).pow(4))
    robust = robust / (1.0 + (rotation_residual / robust_rotation_scale).pow(4))
    weights = learned_weights * robust * support_mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    rotation = _weighted_so3_mean(candidate_rotation, weights)
    center = (weights[..., None] * candidate_center).sum(dim=1)
    translation = -(rotation.transpose(-1, -2) @ center.unsqueeze(-1)).squeeze(-1)
    return CarPose(rotation=rotation, center=center, translation=translation), weights


def _kabsch(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Solve Kabsch in FP32 even when the neural network uses BF16 autocast."""
    source = source.float()
    target = target.float()
    with torch.autocast(device_type=source.device.type, enabled=False):
        covariance = source.transpose(-1, -2) @ target
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        v = vh.transpose(-1, -2)
        correction = torch.eye(3, device=source.device, dtype=source.dtype).expand_as(covariance).clone()
        correction[..., -1, -1] = torch.where(
            torch.linalg.det(v @ u.transpose(-1, -2)) < 0,
            torch.tensor(-1.0, device=source.device, dtype=source.dtype),
            torch.tensor(1.0, device=source.device, dtype=source.dtype),
        )
        return v @ correction @ u.transpose(-1, -2)


def solve_camera_center(directions: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
    if directions.shape != moments.shape or directions.ndim != 3 or directions.shape[-1] != 3:
        raise ValueError("directions and moments must both be [B, N, 3]")
    directions = directions.float()
    moments = moments.float()
    with torch.autocast(device_type=directions.device.type, enabled=False):
        system = -_skew(directions).reshape(directions.shape[0], -1, 3)
        observations = moments.reshape(moments.shape[0], -1, 1)
        return torch.linalg.lstsq(system, observations).solution.squeeze(-1)


def solve_car_pose(
    directions: torch.Tensor, moments: torch.Tensor, camera_directions: torch.Tensor
) -> CarPose:
    if directions.shape != moments.shape or directions.shape != camera_directions.shape:
        raise ValueError("all ray tensors must have the same [B, N, 3] shape")
    with torch.autocast(device_type=directions.device.type, enabled=False):
        world = F.normalize(directions.float(), dim=-1)
        local = F.normalize(camera_directions.float(), dim=-1)
        rotation = _kabsch(local, world)
        center = solve_camera_center(world, moments.float())
        translation = -(rotation.transpose(-1, -2) @ center.unsqueeze(-1)).squeeze(-1)
        return CarPose(rotation=rotation, center=center, translation=translation)


def weighted_solve_car_pose(
    directions: torch.Tensor,
    moments: torch.Tensor,
    camera_directions: torch.Tensor,
    weights: torch.Tensor,
) -> CarPose:
    """Solve CaR with one non-negative confidence weight per ray."""
    if directions.shape != moments.shape or directions.shape != camera_directions.shape:
        raise ValueError("all ray tensors must have the same [B, N, 3] shape")
    if weights.shape != directions.shape[:2]:
        raise ValueError("weights must be [B, N]")
    if torch.any(weights < 0) or not torch.all(weights.sum(dim=-1) > 0):
        raise ValueError("each sample needs non-negative weights with positive sum")
    with torch.autocast(device_type=directions.device.type, enabled=False):
        world = F.normalize(directions.float(), dim=-1)
        local = F.normalize(camera_directions.float(), dim=-1)
        confidence = weights.float()
        covariance = local.transpose(-1, -2) @ (confidence[..., None] * world)
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        v = vh.transpose(-1, -2)
        correction = torch.eye(3, device=world.device, dtype=world.dtype).expand_as(covariance).clone()
        correction[..., -1, -1] = torch.where(
            torch.linalg.det(v @ u.transpose(-1, -2)) < 0,
            torch.tensor(-1.0, device=world.device, dtype=world.dtype),
            torch.tensor(1.0, device=world.device, dtype=world.dtype),
        )
        rotation = v @ correction @ u.transpose(-1, -2)
        system = -_skew(world) * confidence[..., None, None].sqrt()
        observations = moments.float() * confidence[..., None].sqrt()
        center = torch.linalg.lstsq(
            system.reshape(world.shape[0], -1, 3), observations.reshape(world.shape[0], -1, 1)
        ).solution.squeeze(-1)
        translation = -(rotation.transpose(-1, -2) @ center.unsqueeze(-1)).squeeze(-1)
        return CarPose(rotation=rotation, center=center, translation=translation)


def solve_point_pose(world_points: torch.Tensor, camera_points: torch.Tensor) -> CarPose:
    """Recover a c2w pose by Kabsch-aligning canonical camera points to world points."""
    if world_points.shape != camera_points.shape or world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError("world_points and camera_points must both be [B, N, 3]")
    with torch.autocast(device_type=world_points.device.type, enabled=False):
        world_points = world_points.float()
        camera_points = camera_points.float()
        source_mean = camera_points.mean(dim=-2, keepdim=True)
        target_mean = world_points.mean(dim=-2, keepdim=True)
        rotation = _kabsch(camera_points - source_mean, world_points - target_mean)
        center = (target_mean - (rotation @ source_mean.transpose(-1, -2)).transpose(-1, -2)).squeeze(-2)
        translation = -(rotation.transpose(-1, -2) @ center.unsqueeze(-1)).squeeze(-1)
        return CarPose(rotation=rotation, center=center, translation=translation)


def solve_grloc_pose(
    directions: torch.Tensor,
    world_points: torch.Tensor,
    camera_points: torch.Tensor,
) -> CarPose:
    """GRLoc composition: ray-map rotation plus point-map camera center."""
    ray_pose = solve_car_pose(directions, torch.zeros_like(directions), camera_points)
    point_pose = solve_point_pose(world_points, camera_points)
    translation = -(ray_pose.rotation.transpose(-1, -2) @ point_pose.center.unsqueeze(-1)).squeeze(-1)
    return CarPose(rotation=ray_pose.rotation, center=point_pose.center, translation=translation)
