# PASM v0 的 MegaDepth 几何训练目标和双项损失。

from __future__ import annotations

import numpy as np
import torch

from .sampling import PatchLayout


def point_match_targets(
    points_a: np.ndarray,
    points_b: np.ndarray,
    geometry_a: np.ndarray,
    geometry_b: np.ndarray,
    threshold_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将当前采样点与可靠几何投影对齐为点级监督目标。"""
    source = _as_points(points_a)
    target = _as_points(points_b)
    reference_a = _as_points(geometry_a)
    reference_b = _as_points(geometry_b)
    if len(reference_a) != len(reference_b) or threshold_px <= 0.0:
        raise ValueError("几何点对长度或几何阈值无效")
    targets = np.full(len(source), -1, dtype=np.int64)
    valid = np.zeros(len(source), dtype=bool)
    if len(reference_a) == 0 or len(source) == 0 or len(target) == 0:
        return torch.from_numpy(targets), torch.from_numpy(valid)
    nearest_geometry = _nearest_indices(source, reference_a)
    source_distance = np.linalg.norm(source - reference_a[nearest_geometry], axis=1)
    projected = reference_b[nearest_geometry]
    nearest_target = _nearest_indices(projected, target)
    target_distance = np.linalg.norm(projected - target[nearest_target], axis=1)
    valid = (source_distance <= threshold_px) & (target_distance <= threshold_px)
    targets[valid] = nearest_target[valid]
    return torch.from_numpy(targets), torch.from_numpy(valid)


def patch_support_targets(geometry_points: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """将可靠几何正点的空间覆盖度转换为 Patch 采样价值分布。"""
    points = _as_points(geometry_points)
    support = np.zeros(layout.config.patch_count, dtype=np.float32)
    if len(points) == 0:
        return support
    patch_ids = layout.patch_ids(points)
    support += np.bincount(patch_ids, minlength=layout.config.patch_count).astype(np.float32)
    return support / support.sum()


def pasm_v0_loss(
    edge_weights: torch.Tensor,
    target_indices: torch.Tensor,
    valid_rows: torch.Tensor,
    patch_values: torch.Tensor,
    patch_targets: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """计算可靠点级正对应和 Patch 价值分配的联合损失。"""
    if edge_weights.ndim != 2 or edge_weights.shape[0] != target_indices.numel():
        raise ValueError("边权和点级目标的形状不一致")
    if valid_rows.shape != target_indices.shape or patch_values.ndim != 1:
        raise ValueError("有效行或 Patch 价值的形状无效")
    targets = target_indices.to(edge_weights.device, dtype=torch.long)
    valid = valid_rows.to(edge_weights.device, dtype=torch.bool) & (targets >= 0)
    zero = edge_weights.sum() * 0.0 + patch_values.sum() * 0.0
    if torch.any(valid):
        rows = torch.arange(edge_weights.shape[0], device=edge_weights.device)[valid]
        match_loss = -torch.log(edge_weights[rows, targets[valid]].clamp_min(1e-8)).mean()
    else:
        match_loss = zero
    support = torch.as_tensor(patch_targets, dtype=patch_values.dtype, device=patch_values.device)
    if support.ndim != 1 or support.numel() != patch_values.numel():
        raise ValueError("Patch 正区域目标的形状无效")
    if torch.any(support > 0):
        distribution = patch_values.clamp_min(1e-8)
        distribution = distribution / distribution.sum().clamp_min(1e-8)
        value_loss = -(support * torch.log(distribution)).sum()
    else:
        value_loss = zero
    return match_loss + value_loss


def _as_points(points_xy: np.ndarray) -> np.ndarray:
    """验证并转换几何或采样坐标为二维浮点数组。"""
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("点坐标必须具有 [N, 2] 形状且全部有限")
    return points


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """返回每个查询点在参考点集合中的最近邻索引。"""
    distances = np.sum((query[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    return distances.argmin(axis=1)
