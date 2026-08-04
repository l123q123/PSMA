# PASM v0.3 的 DaD 检测结果与静态 Patch 预算选点。

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sampling import PatchLayout


@dataclass(frozen=True)
class DaDDetections:
    """保存 DaD 的稠密分数、NMS 亚像素坐标和对应分数。"""

    dense_scores: np.ndarray
    nms_points: np.ndarray
    nms_scores: np.ndarray


def select_patch_budgeted_points(
    detections: DaDDetections, layout: PatchLayout, budgets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按每个 Patch 的预算优先选择 DaD NMS 点，再按稠密分数补足。"""
    dense_scores, nms_points, nms_scores = _validated_detections(detections, layout.config.image_size)
    allocation = _validated_patch_budgets(budgets, layout)
    nms_patch_ids = layout.patch_ids(nms_points) if len(nms_points) else np.empty((0,), dtype=np.int64)
    used_cells: set[tuple[int, int]] = set()
    points = []
    patch_ids = []
    scores = []
    for patch_id, budget in enumerate(allocation):
        selected = 0
        candidates = np.flatnonzero(nms_patch_ids == patch_id)
        ordered = candidates[np.argsort(-nms_scores[candidates], kind="stable")]
        for index in ordered:
            cell = _point_cell(nms_points[index], layout.config.image_size)
            if cell in used_cells:
                continue
            points.append(nms_points[index])
            patch_ids.append(patch_id)
            scores.append(nms_scores[index])
            used_cells.add(cell)
            selected += 1
            if selected == budget:
                break
        if selected < budget:
            for x_value, y_value, score in _dense_patch_candidates(dense_scores, layout, patch_id):
                cell = (int(y_value), int(x_value))
                if cell in used_cells:
                    continue
                points.append(np.array((x_value + 0.5, y_value + 0.5), dtype=np.float32))
                patch_ids.append(patch_id)
                scores.append(score)
                used_cells.add(cell)
                selected += 1
                if selected == budget:
                    break
        if selected != budget:
            raise ValueError("DaD 稠密分数无法填满指定 Patch 预算")
    result_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    result_ids = np.asarray(patch_ids, dtype=np.int64)
    result_scores = np.asarray(scores, dtype=np.float32)
    _validate_patch_selection(result_points, result_ids, allocation, layout, used_cells)
    return result_points, result_ids, result_scores


def select_global_dad_points(
    detections: DaDDetections, image_size: int, point_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """在全图优先选择 DaD NMS 点，并用最高稠密分数补足固定点数。"""
    dense_scores, nms_points, nms_scores = _validated_detections(detections, image_size)
    if point_count <= 0 or point_count > image_size * image_size:
        raise ValueError("全局 DaD 点数必须位于图像像素数量范围内")
    used_cells: set[tuple[int, int]] = set()
    points = []
    scores = []
    for index in np.argsort(-nms_scores, kind="stable"):
        cell = _point_cell(nms_points[index], image_size)
        if cell in used_cells:
            continue
        points.append(nms_points[index])
        scores.append(nms_scores[index])
        used_cells.add(cell)
        if len(points) == point_count:
            break
    if len(points) < point_count:
        for flat_index in np.argsort(-dense_scores.reshape(-1), kind="stable"):
            y_value, x_value = divmod(int(flat_index), image_size)
            cell = (y_value, x_value)
            if cell in used_cells:
                continue
            points.append(np.array((x_value + 0.5, y_value + 0.5), dtype=np.float32))
            scores.append(dense_scores[y_value, x_value])
            used_cells.add(cell)
            if len(points) == point_count:
                break
    result_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    result_scores = np.asarray(scores, dtype=np.float32)
    if len(result_points) != point_count or len(used_cells) != point_count:
        raise ValueError("DaD 稠密分数无法填满全局点预算")
    return result_points, result_scores


def _validated_detections(
    detections: DaDDetections, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """验证并规范化 DaD 选点所需的稠密分数和 NMS 候选。"""
    dense_scores = np.asarray(detections.dense_scores, dtype=np.float64)
    nms_points = np.asarray(detections.nms_points, dtype=np.float64)
    nms_scores = np.asarray(detections.nms_scores, dtype=np.float64).reshape(-1)
    if image_size <= 0 or dense_scores.shape != (image_size, image_size) or not np.isfinite(dense_scores).all():
        raise ValueError("DaD 稠密分数必须是有限的 [图像高, 图像宽] 数组")
    if nms_points.ndim != 2 or nms_points.shape[1:] != (2,) or len(nms_points) != len(nms_scores):
        raise ValueError("DaD NMS 点和分数形状必须一致")
    if not np.isfinite(nms_points).all() or not np.isfinite(nms_scores).all():
        raise ValueError("DaD NMS 点和分数必须有限")
    if np.any(nms_points < 0.0) or np.any(nms_points >= image_size):
        raise ValueError("DaD NMS 点必须位于图像范围内")
    return dense_scores, nms_points, nms_scores


def _validated_patch_budgets(budgets: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """验证 Patch 预算与 PASM 总点数约束完全一致。"""
    allocation = np.asarray(budgets, dtype=np.int64).reshape(-1)
    if allocation.size != layout.config.patch_count or np.any(allocation < 0):
        raise ValueError("DaD Patch 预算必须是非负的完整 Patch 数组")
    if int(allocation.sum()) != layout.config.total_points:
        raise ValueError("DaD Patch 预算总数必须等于 PASM 总点数")
    return allocation


def _dense_patch_candidates(
    dense_scores: np.ndarray, layout: PatchLayout, patch_id: int
) -> list[tuple[int, int, float]]:
    """按分数降序返回中心属于指定 Patch 的全部稠密像素。"""
    x0, y0, x1, y1 = layout.bounds(patch_id)
    image_size = layout.config.image_size
    x_indices = np.arange(max(0, int(np.ceil(x0 - 0.5))), min(image_size, int(np.ceil(x1 - 0.5))))
    y_indices = np.arange(max(0, int(np.ceil(y0 - 0.5))), min(image_size, int(np.ceil(y1 - 0.5))))
    patch_scores = dense_scores[np.ix_(y_indices, x_indices)]
    result = []
    for local_index in np.argsort(-patch_scores.reshape(-1), kind="stable"):
        local_y, local_x = divmod(int(local_index), len(x_indices))
        result.append((int(x_indices[local_x]), int(y_indices[local_y]), float(patch_scores[local_y, local_x])))
    return result


def _point_cell(point_xy: np.ndarray, image_size: int) -> tuple[int, int]:
    """返回一个图像内浮点坐标所占用的整数像素位置。"""
    x_value, y_value = np.asarray(point_xy, dtype=np.float64)
    x_index = int(np.floor(x_value))
    y_index = int(np.floor(y_value))
    if not 0 <= x_index < image_size or not 0 <= y_index < image_size:
        raise ValueError("DaD 点对应的整数像素超出图像范围")
    return y_index, x_index


def _validate_patch_selection(
    points: np.ndarray,
    patch_ids: np.ndarray,
    budgets: np.ndarray,
    layout: PatchLayout,
    used_cells: set[tuple[int, int]],
) -> None:
    """验证选出的点数、坐标、Patch 归属和整数像素占用均满足约束。"""
    if len(points) != layout.config.total_points or len(points) != len(patch_ids):
        raise ValueError("DaD Patch 选点总数与 PASM 总点数不一致")
    if len(used_cells) != len(points) or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("DaD Patch 选点不得重复")
    if np.any(points < 0.0) or np.any(points >= layout.config.image_size):
        raise ValueError("DaD Patch 选点超出图像范围")
    if not np.array_equal(np.bincount(patch_ids, minlength=layout.config.patch_count), budgets):
        raise ValueError("DaD Patch 实际点数必须等于分配预算")
    if not np.array_equal(layout.patch_ids(points), patch_ids):
        raise ValueError("DaD Patch 选点必须位于声明 Patch 内")
