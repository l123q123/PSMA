# PASM v0 的 Patch 网格、预算分配和 Patch 内重采样。

from __future__ import annotations

import numpy as np

from .config import PasmV0Config


class PatchLayout:
    """将连续像素坐标映射到固定的管理 Patch 网格。"""

    def __init__(self, config: PasmV0Config) -> None:
        """根据配置创建正方形图像的管理 Patch 网格。"""
        self.config = config
        self.patch_width = config.image_size / config.patch_cols
        self.patch_height = config.image_size / config.patch_rows

    def bounds(self, patch_id: int) -> tuple[float, float, float, float]:
        """返回指定 Patch 的左上和右下像素边界。"""
        if not 0 <= patch_id < self.config.patch_count:
            raise ValueError("Patch 索引超出范围")
        row, col = divmod(patch_id, self.config.patch_cols)
        return (
            col * self.patch_width,
            row * self.patch_height,
            (col + 1) * self.patch_width,
            (row + 1) * self.patch_height,
        )

    def patch_ids(self, points_xy: np.ndarray) -> np.ndarray:
        """返回每个图像内坐标所属的 Patch 索引。"""
        points = _as_points(points_xy)
        if np.any(points < 0.0) or np.any(points >= self.config.image_size):
            raise ValueError("采样点必须位于图像范围内")
        columns = np.minimum(
            (points[:, 0] / self.patch_width).astype(np.int64), self.config.patch_cols - 1
        )
        rows = np.minimum(
            (points[:, 1] / self.patch_height).astype(np.int64), self.config.patch_rows - 1
        )
        return rows * self.config.patch_cols + columns

    def initial_points(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """在每个 Patch 内生成首轮固定数量的分层随机点。"""
        generator = np.random.default_rng(seed)
        if self.config.initial_patch_subset:
            patch_ids = generator.choice(
                self.config.patch_count, size=self.config.total_points, replace=False
            )
            points = np.concatenate(
                [_uniform_points(self.bounds(int(patch_id)), 1, generator) for patch_id in patch_ids],
                axis=0,
            )
            return points.astype(np.float32), patch_ids.astype(np.int64)
        count = self.config.minimum_for_round(0)
        points = []
        patch_ids = []
        for patch_id in range(self.config.patch_count):
            points.append(_uniform_points(self.bounds(patch_id), count, generator))
            patch_ids.extend([patch_id] * count)
        return np.concatenate(points, axis=0).astype(np.float32), np.asarray(patch_ids, dtype=np.int64)


def allocate_patch_budget(
    values: np.ndarray,
    config: PasmV0Config,
    round_index: int,
    reserved: np.ndarray | None = None,
) -> np.ndarray:
    """按 Patch 价值和锚点保留数量分配精确总预算。"""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.size != config.patch_count:
        raise ValueError("Patch 价值数量必须与网格大小一致")
    reservation = _reserved_counts(reserved, config)
    minimum = config.minimum_for_round(round_index)
    budget = np.maximum(np.full(config.patch_count, minimum, dtype=np.int64), reservation)
    if np.any(budget > config.max_points_per_patch):
        raise ValueError("锚点数量超过 Patch 点数上限")
    remaining = config.total_points - int(budget.sum())
    if remaining < 0:
        raise ValueError("锚点和最低点数超过总采样预算")
    if remaining == 0:
        return budget
    scores = np.maximum(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0)
    scores = np.power(scores, config.budget_score_power)
    while remaining > 0:
        eligible = budget < config.max_points_per_patch
        if not np.any(eligible):
            raise ValueError("Patch 点数上限无法容纳总预算")
        increments = _proportional_increments(
            scores, eligible, budget, remaining, config.max_points_per_patch
        )
        budget += increments
        remaining -= int(increments.sum())
    return budget


def resample_in_patches(
    layout: PatchLayout,
    previous_points: np.ndarray,
    previous_patch_ids: np.ndarray,
    point_quality: np.ndarray,
    budgets: np.ndarray,
    config: PasmV0Config,
    seed: int | None = None,
    anchor_points: np.ndarray | None = None,
    anchor_patch_ids: np.ndarray | None = None,
    anchor_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """原样保留锚点并在各自 Patch 内生成下一轮点集。"""
    points = _as_points(previous_points)
    patch_ids = np.asarray(previous_patch_ids, dtype=np.int64).reshape(-1)
    quality = np.asarray(point_quality, dtype=np.float64).reshape(-1)
    allocation = np.asarray(budgets, dtype=np.int64).reshape(-1)
    if len(points) != len(patch_ids) or len(points) != len(quality):
        raise ValueError("旧点、Patch 索引和点质量的长度必须一致")
    if allocation.size != config.patch_count or int(allocation.sum()) != config.total_points:
        raise ValueError("预算必须覆盖所有 Patch 且总和等于总点数")
    if np.any(allocation < 0) or np.any(allocation > config.max_points_per_patch):
        raise ValueError("Patch 预算超出允许范围")
    anchors, anchor_ids, scores = _anchor_arrays(
        anchor_points, anchor_patch_ids, anchor_scores, layout, config
    )
    anchor_counts = np.bincount(anchor_ids, minlength=config.patch_count)
    if np.any(anchor_counts > allocation):
        raise ValueError("Patch 预算不足以保留全部锚点")
    generator = np.random.default_rng(seed)
    sampled_points = []
    sampled_ids = []
    for patch_id, count in enumerate(allocation):
        if count == 0:
            continue
        mask = patch_ids == patch_id
        patch_points = points[mask]
        patch_quality = quality[mask]
        anchor_mask = anchor_ids == patch_id
        new_points = _resample_patch(
            layout.bounds(patch_id),
            patch_points,
            patch_quality,
            anchors[anchor_mask],
            scores[anchor_mask],
            int(count),
            config,
            generator,
        )
        sampled_points.append(new_points)
        sampled_ids.extend([patch_id] * int(count))
    return (
        np.concatenate(sampled_points, axis=0).astype(np.float32),
        np.asarray(sampled_ids, dtype=np.int64),
    )


def _as_points(points_xy: np.ndarray) -> np.ndarray:
    """将输入转换为有限的二维浮点坐标数组。"""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("坐标必须是形状为 [N, 2] 的有限数组")
    return points


def _reserved_counts(reserved: np.ndarray | None, config: PasmV0Config) -> np.ndarray:
    """验证并返回每个 Patch 必须保留的锚点数量。"""
    if reserved is None:
        return np.zeros(config.patch_count, dtype=np.int64)
    counts = np.asarray(reserved, dtype=np.int64).reshape(-1)
    if counts.size != config.patch_count or np.any(counts < 0):
        raise ValueError("锚点保留数量必须是非负的每 Patch 数组")
    return counts


def _anchor_arrays(
    anchor_points: np.ndarray | None,
    anchor_patch_ids: np.ndarray | None,
    anchor_scores: np.ndarray | None,
    layout: PatchLayout,
    config: PasmV0Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """验证锚点坐标、所属 Patch 和边置信度。"""
    if anchor_points is None and anchor_patch_ids is None and anchor_scores is None:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    if anchor_points is None or anchor_patch_ids is None or anchor_scores is None:
        raise ValueError("锚点坐标、Patch 索引和分数必须同时提供")
    points = _as_points(anchor_points)
    patch_ids = np.asarray(anchor_patch_ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(anchor_scores, dtype=np.float64).reshape(-1)
    if len(points) != len(patch_ids) or len(points) != len(scores):
        raise ValueError("锚点坐标、Patch 索引和分数长度必须一致")
    if np.any(patch_ids < 0) or np.any(patch_ids >= config.patch_count) or not np.isfinite(scores).all():
        raise ValueError("锚点 Patch 索引或分数无效")
    if not np.array_equal(layout.patch_ids(points), patch_ids):
        raise ValueError("锚点坐标必须位于声明的 Patch 内")
    return points, patch_ids, scores


def _uniform_points(
    bounds: tuple[float, float, float, float], count: int, generator: np.random.Generator
) -> np.ndarray:
    """在单个 Patch 的矩形边界内生成均匀随机坐标。"""
    x0, y0, x1, y1 = bounds
    random = generator.random((count, 2))
    points = np.column_stack((x0 + random[:, 0] * (x1 - x0), y0 + random[:, 1] * (y1 - y0)))
    upper = np.nextafter(
        np.asarray((x1, y1), dtype=np.float32), np.asarray((x0, y0), dtype=np.float32)
    )
    return np.minimum(points, upper)


def _proportional_increments(
    scores: np.ndarray,
    eligible: np.ndarray,
    budget: np.ndarray,
    remaining: int,
    max_points_per_patch: int,
) -> np.ndarray:
    """通过带容量限制的最大余数法分配当前剩余预算。"""
    capacity = np.where(eligible, max_points_per_patch - budget, 0)
    active_scores = np.where(eligible, scores, 0.0)
    if active_scores.sum() <= 0.0:
        active_scores = eligible.astype(np.float64)
    desired = remaining * active_scores / active_scores.sum()
    increments = np.minimum(np.floor(desired).astype(np.int64), capacity)
    left = remaining - int(increments.sum())
    if left > 0:
        fractions = desired - np.floor(desired)
        order = np.argsort(-fractions, kind="stable")
        for index in order:
            if left == 0:
                break
            if eligible[index] and increments[index] < capacity[index]:
                increments[index] += 1
                left -= 1
    return increments


def _resample_patch(
    bounds: tuple[float, float, float, float],
    old_points: np.ndarray,
    old_quality: np.ndarray,
    anchors: np.ndarray,
    anchor_scores: np.ndarray,
    count: int,
    config: PasmV0Config,
    generator: np.random.Generator,
) -> np.ndarray:
    """为单个 Patch 保留锚点并生成探索点和局部点。"""
    if len(anchors) > count:
        raise ValueError("单个 Patch 的锚点数量超过预算")
    if len(anchors):
        remaining = count - len(anchors)
        if remaining == 0:
            return anchors.copy()
        explore_count = max(1, int(np.ceil(remaining * config.exploration_ratio)))
        explore_count = min(explore_count, remaining)
        exploratory = _uniform_points(bounds, explore_count, generator)
        local_count = remaining - explore_count
        if local_count == 0:
            return np.concatenate((anchors, exploratory), axis=0)
        local = _jittered_points(bounds, anchors, anchor_scores, local_count, config, generator)
        return np.concatenate((anchors, exploratory, local), axis=0)
    explore_count = count if len(old_points) == 0 else max(1, int(np.ceil(count * config.exploration_ratio)))
    exploratory = _uniform_points(bounds, explore_count, generator)
    local_count = count - explore_count
    if local_count == 0:
        return exploratory
    return np.concatenate(
        (exploratory, _jittered_points(bounds, old_points, old_quality, local_count, config, generator)), axis=0
    )


def _jittered_points(
    bounds: tuple[float, float, float, float],
    centers: np.ndarray,
    quality: np.ndarray,
    count: int,
    config: PasmV0Config,
    generator: np.random.Generator,
) -> np.ndarray:
    """按分数选择中心并在其邻域生成裁剪后的局部采样点。"""
    weights = np.maximum(np.nan_to_num(quality, nan=0.0), 0.0)
    probabilities = None if weights.sum() <= 0.0 else weights / weights.sum()
    selected = centers[generator.choice(len(centers), size=count, p=probabilities)]
    x0, y0, x1, y1 = bounds
    scale = np.array(((x1 - x0) * config.jitter_ratio, (y1 - y0) * config.jitter_ratio))
    local = selected + generator.normal(scale=scale, size=selected.shape)
    maximum_x = float(np.nextafter(np.float32(x1), np.float32(x0)))
    maximum_y = float(np.nextafter(np.float32(y1), np.float32(y0)))
    local[:, 0] = np.clip(local[:, 0], x0, maximum_x)
    local[:, 1] = np.clip(local[:, 1], y0, maximum_y)
    return local
