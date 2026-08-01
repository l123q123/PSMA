# PASM v0.2 前的软质量教师选点和快速量化工具。

from __future__ import annotations

import numpy as np
import torch

from .geometry import PairGeometry, visible_projected_points
from .matching import sample_feature_map
from .sampling import PatchLayout


def patch_candidate_points(
    layout: PatchLayout, candidates_per_axis: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """在每个管理 Patch 内生成均匀局部候选格。"""
    if candidates_per_axis <= 0:
        raise ValueError("每轴候选点数必须为正数")
    offsets = (np.arange(candidates_per_axis, dtype=np.float32) + 0.5) / candidates_per_axis
    candidates = []
    patch_ids = []
    for patch_id in range(layout.config.patch_count):
        x0, y0, x1, y1 = layout.bounds(patch_id)
        grid_y, grid_x = np.meshgrid(
            y0 + (y1 - y0) * offsets,
            x0 + (x1 - x0) * offsets,
            indexing="ij",
        )
        candidates.append(np.column_stack((grid_x.ravel(), grid_y.ravel())))
        patch_ids.extend([patch_id] * candidates_per_axis**2)
    return np.concatenate(candidates).astype(np.float32), np.asarray(patch_ids, dtype=np.int64)


def select_patch_teacher_points(
    points: np.ndarray,
    patch_ids: np.ndarray,
    scores: np.ndarray,
    layout: PatchLayout,
    points_per_patch: int = 4,
    nms_radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在每个 Patch 内经局部非极大值抑制选择固定数量的候选点。"""
    candidates = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    candidate_ids = np.asarray(patch_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(candidates) != len(candidate_ids) or len(candidates) != len(values):
        raise ValueError("候选点、Patch 索引和分数的长度必须一致")
    if points_per_patch <= 0 or nms_radius < 0.0:
        raise ValueError("每 Patch 点数和非极大值抑制半径无效")
    if np.any(candidate_ids < 0) or np.any(candidate_ids >= layout.config.patch_count):
        raise ValueError("候选点 Patch 索引超出范围")
    selected_points = []
    selected_ids = []
    selected_scores = []
    normalized_values = np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    for patch_id in range(layout.config.patch_count):
        indices = np.flatnonzero(candidate_ids == patch_id)
        if len(indices) < points_per_patch:
            raise ValueError("每个 Patch 的候选点不足")
        ordered = indices[np.argsort(-normalized_values[indices], kind="stable")]
        kept = []
        for index in ordered:
            if all(np.linalg.norm(candidates[index] - candidates[other]) > nms_radius for other in kept):
                kept.append(int(index))
            if len(kept) == points_per_patch:
                break
        for index in ordered:
            if len(kept) == points_per_patch:
                break
            if int(index) not in kept:
                kept.append(int(index))
        selected_points.append(candidates[kept])
        selected_ids.extend([patch_id] * points_per_patch)
        selected_scores.append(values[kept])
    return (
        np.concatenate(selected_points).astype(np.float32),
        np.asarray(selected_ids, dtype=np.int64),
        np.concatenate(selected_scores).astype(np.float32),
    )


def teacher_scores(
    source_points: np.ndarray,
    target_candidates: np.ndarray,
    descriptor_grid_source: np.ndarray,
    descriptor_grid_target: np.ndarray,
    geometry: PairGeometry,
    direction: str,
    depth_tolerance: float,
    temperature: float,
    exclusion_radius: float,
    device: str | torch.device = "cpu",
    negative_chunk_size: int = 512,
) -> np.ndarray:
    """计算几何可见且描述子明显优于错误位置的软质量分数。"""
    sources = _point_array(source_points, "源候选点")
    targets = _point_array(target_candidates, "目标候选点")
    if len(targets) == 0 or temperature <= 0.0 or exclusion_radius < 0.0 or negative_chunk_size <= 0:
        raise ValueError("目标候选点、温度、排除半径或分块大小无效")
    selected_device = torch.device(device)
    source_grid = _descriptor_grid(descriptor_grid_source, "源描述子图")
    target_grid = _descriptor_grid(descriptor_grid_target, "目标描述子图")
    if source_grid.shape[0] != target_grid.shape[0]:
        raise ValueError("两张图的描述子通道数必须一致")
    visible_indices, projected = visible_projected_points(
        sources, geometry, direction, depth_tolerance
    )
    scores = np.zeros(len(sources), dtype=np.float32)
    if len(visible_indices) == 0:
        return scores
    with torch.inference_mode():
        source_descriptors = _sample_descriptors(
            source_grid, sources[visible_indices], geometry.image_size, selected_device
        )
        positive_descriptors = _sample_descriptors(
            target_grid, projected, geometry.image_size, selected_device
        )
        target_descriptors = _sample_descriptors(
            target_grid, targets, geometry.image_size, selected_device
        )
        positive = (source_descriptors * positive_descriptors).sum(dim=1)
        negative = _hardest_negative_similarity(
            source_descriptors,
            target_descriptors,
            projected,
            targets,
            exclusion_radius,
            selected_device,
            negative_chunk_size,
        )
        values = torch.sigmoid((positive - negative) / temperature)
    scores[visible_indices] = values.cpu().numpy().astype(np.float32)
    return scores


def cosine_mutual_matches(
    descriptors_a: np.ndarray, descriptors_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据归一化余弦相似度返回两图描述子的互为最近邻。"""
    first = _descriptor_array(descriptors_a, "A 图点描述子")
    second = _descriptor_array(descriptors_b, "B 图点描述子")
    if first.shape[1] != second.shape[1]:
        raise ValueError("两图点描述子的通道数必须一致")
    if len(first) == 0 or len(second) == 0:
        empty_indices = np.empty((0,), dtype=np.int64)
        return empty_indices, empty_indices.copy(), np.empty((0,), dtype=np.float32)
    first /= np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-6)
    second /= np.linalg.norm(second, axis=1, keepdims=True).clip(min=1e-6)
    similarities = first @ second.T
    best_target = similarities.argmax(axis=1)
    best_source = similarities.argmax(axis=0)
    source = np.arange(len(first), dtype=np.int64)
    keep = best_source[best_target] == source
    return source[keep], best_target[keep].astype(np.int64), similarities[source[keep], best_target[keep]]


def geometry_coverage_rate(
    source_points: np.ndarray,
    target_points: np.ndarray,
    geometry: PairGeometry,
    direction: str,
    threshold_px: float,
    depth_tolerance: float,
) -> float:
    """计算可见源点的真实投影被目标选点集覆盖的比例。"""
    sources = _point_array(source_points, "源选点")
    targets = _point_array(target_points, "目标选点")
    if threshold_px <= 0.0:
        raise ValueError("覆盖阈值必须为正数")
    visible_indices, projected = visible_projected_points(
        sources, geometry, direction, depth_tolerance
    )
    if len(visible_indices) == 0 or len(targets) == 0:
        return 0.0
    distances = np.sum((projected[:, None, :] - targets[None, :, :]) ** 2, axis=2)
    return float(np.mean(distances.min(axis=1) <= threshold_px**2))


def mutual_geometry_inlier_rate(
    points_a: np.ndarray,
    points_b: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    geometry: PairGeometry,
    threshold_px: float,
    depth_tolerance: float,
) -> float:
    """计算所有互为最近邻边中满足双图几何投影的内点比例。"""
    first = _point_array(points_a, "A 图选点")
    second = _point_array(points_b, "B 图选点")
    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    target = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if len(source) != len(target) or threshold_px <= 0.0:
        raise ValueError("匹配索引长度或几何阈值无效")
    if np.any(source < 0) or np.any(source >= len(first)) or np.any(target < 0) or np.any(target >= len(second)):
        raise ValueError("匹配索引超出选点范围")
    if len(source) == 0:
        return 0.0
    visible_indices, projected = visible_projected_points(
        first, geometry, "a_to_b", depth_tolerance
    )
    expected = np.full((len(first), 2), np.nan, dtype=np.float32)
    expected[visible_indices] = projected
    matched_expected = expected[source]
    valid = np.isfinite(matched_expected).all(axis=1)
    distances = np.sum((matched_expected - second[target]) ** 2, axis=1)
    return float(np.mean(valid & (distances <= threshold_px**2)))


def sampled_descriptors(
    descriptor_grid: np.ndarray, points: np.ndarray, image_size: int, device: str | torch.device
) -> np.ndarray:
    """从稠密描述子图读取并返回一批归一化浮点描述子。"""
    selected_device = torch.device(device)
    grid = _descriptor_grid(descriptor_grid, "描述子图")
    coordinates = _point_array(points, "读取坐标")
    with torch.inference_mode():
        descriptors = _sample_descriptors(grid, coordinates, image_size, selected_device)
    return descriptors.cpu().numpy().astype(np.float32)


def _sample_descriptors(
    descriptor_grid: np.ndarray, points: np.ndarray, image_size: int, device: torch.device
) -> torch.Tensor:
    """从 NumPy 稠密描述子图读取并归一化一批浮点坐标的描述子。"""
    grid = torch.as_tensor(descriptor_grid, dtype=torch.float32, device=device)
    coordinates = torch.as_tensor(points, dtype=torch.float32, device=device)
    return sample_feature_map(grid, coordinates, (image_size, image_size))


def _hardest_negative_similarity(
    source_descriptors: torch.Tensor,
    target_descriptors: torch.Tensor,
    projected: np.ndarray,
    target_points: np.ndarray,
    exclusion_radius: float,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """分块查找每个源描述子在真实投影邻域外的最大余弦相似度。"""
    projected_tensor = torch.as_tensor(projected, dtype=torch.float32, device=device)
    targets_tensor = torch.as_tensor(target_points, dtype=torch.float32, device=device)
    values = []
    for start in range(0, len(source_descriptors), chunk_size):
        stop = min(start + chunk_size, len(source_descriptors))
        similarities = source_descriptors[start:stop] @ target_descriptors.T
        distances = (projected_tensor[start:stop, None, :] - targets_tensor[None, :, :]).square().sum(dim=2)
        similarities = similarities.masked_fill(distances <= exclusion_radius**2, -torch.inf)
        hardest = similarities.max(dim=1).values
        values.append(torch.where(torch.isfinite(hardest), hardest, torch.full_like(hardest, -1.0)))
    return torch.cat(values, dim=0)


def _point_array(points: np.ndarray, name: str) -> np.ndarray:
    """验证并返回形状规范的有限二维像素坐标。"""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限的 [N, 2] 坐标")
    return array


def _descriptor_grid(grid: np.ndarray, name: str) -> np.ndarray:
    """验证并返回连续存储的有限 CHW 稠密描述子图。"""
    array = np.ascontiguousarray(np.asarray(grid, dtype=np.float32))
    if array.ndim != 3 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限的 [C, H, W] 数组")
    return array


def _descriptor_array(descriptors: np.ndarray, name: str) -> np.ndarray:
    """验证并复制形状规范的二维点描述子数组。"""
    array = np.asarray(descriptors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限的 [N, C] 数组")
    return array.copy()
