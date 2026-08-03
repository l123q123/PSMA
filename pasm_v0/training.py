# PASM v0 的 MegaDepth 几何训练目标和双项损失。

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as functional

from .sampling import PatchLayout
from .geometry import PairGeometry, projected_match_targets, projected_patch_support_targets
from .pipeline import RoundRecord


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


def pasm_v02_loss(
    rounds: list[RoundRecord],
    geometry: PairGeometry,
    layout: PatchLayout,
    config: object,
    teacher_a: np.ndarray | torch.Tensor,
    teacher_b: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """汇总三轮几何基础损失和双图密集质量教师损失。"""
    if not rounds:
        raise ValueError("v0.2 损失至少需要一轮记录")
    support_a, support_b = projected_patch_support_targets(
        geometry, layout, config.geometry_support_stride, config.geometry_depth_tolerance
    )
    base_terms = []
    for record in rounds:
        target_a, valid_a, target_b, valid_b = projected_match_targets(
            record.points_a, record.points_b, geometry,
            config.geometry_match_threshold_px, config.geometry_depth_tolerance,
        )
        base_terms.extend((
            pasm_v0_loss(record.edge_weights, target_a, valid_a, record.patch_values_a, support_a),
            pasm_v0_loss(record.edge_weights.transpose(0, 1), target_b, valid_b, record.patch_values_b, support_b),
        ))
    base_loss = torch.stack(base_terms).mean()
    first = rounds[0]
    if first.quality_logits_a is None or first.quality_logits_b is None:
        raise ValueError("v0.2 轮次记录缺少密集质量 logits")
    target_a = torch.as_tensor(teacher_a, dtype=first.quality_logits_a.dtype, device=first.quality_logits_a.device)
    target_b = torch.as_tensor(teacher_b, dtype=first.quality_logits_b.dtype, device=first.quality_logits_b.device)
    if target_a.shape != first.quality_logits_a.shape[1:] or target_b.shape != first.quality_logits_b.shape[1:]:
        raise ValueError("质量教师图尺寸与模型质量 logits 不一致")
    quality_loss = functional.binary_cross_entropy_with_logits(first.quality_logits_a[0], target_a)
    quality_loss = quality_loss + functional.binary_cross_entropy_with_logits(first.quality_logits_b[0], target_b)
    return base_loss + quality_loss, base_loss, quality_loss


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
