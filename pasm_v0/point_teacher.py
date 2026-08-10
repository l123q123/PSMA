"""用于稠密点融合训练的几何候选点软教师。"""

from __future__ import annotations

import numpy as np

from .geometry import PairGeometry, visible_projected_points


def build_candidate_teacher(
    source_points: np.ndarray,
    target_points: np.ndarray,
    geometry: PairGeometry,
    direction: str,
    sigma_px: float,
    search_radius_px: float,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为源 DaD 候选点生成跨图可见且邻近目标候选的软标签。"""
    source = _point_array(source_points, "源候选点")
    target = _point_array(target_points, "目标候选点")
    if direction not in {"a_to_b", "b_to_a"}:
        raise ValueError("投影方向必须是 a_to_b 或 b_to_a")
    if sigma_px <= 0.0 or search_radius_px <= 0.0 or depth_tolerance < 0.0:
        raise ValueError("教师 sigma、搜索半径和深度容差无效")
    labels = np.zeros(len(source), dtype=np.float32)
    visible = np.zeros(len(source), dtype=bool)
    distances = np.full(len(source), np.inf, dtype=np.float32)
    source_indices, projected = visible_projected_points(
        source, geometry, direction, depth_tolerance
    )
    visible[source_indices] = True
    if len(source_indices) == 0 or len(target) == 0:
        return labels, visible, distances
    nearest = _nearest_distances(projected, target)
    within_radius = nearest <= search_radius_px
    valid_indices = source_indices[within_radius]
    valid_distances = nearest[within_radius]
    distances[valid_indices] = valid_distances.astype(np.float32)
    labels[valid_indices] = np.exp(-(valid_distances ** 2) / (2.0 * sigma_px ** 2)).astype(np.float32)
    return labels, visible, distances


def _nearest_distances(
    query_points: np.ndarray, target_points: np.ndarray, chunk_size: int = 2048
) -> np.ndarray:
    """分块计算每个查询点到目标集合的欧氏最近距离。"""
    if chunk_size <= 0:
        raise ValueError("最近邻分块大小必须为正数")
    distances = np.empty(len(query_points), dtype=np.float64)
    for start in range(0, len(query_points), chunk_size):
        stop = min(start + chunk_size, len(query_points))
        difference = query_points[start:stop, None, :] - target_points[None, :, :]
        distances[start:stop] = np.sqrt(np.sum(difference * difference, axis=2).min(axis=1))
    return distances


def _point_array(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限的 [N, 2] 数组")
    return array
