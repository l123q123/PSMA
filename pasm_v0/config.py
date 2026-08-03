# PASM v0 的集中配置定义。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PasmV0Config:
    """保存 PASM v0 的网格、预算和重采样参数。"""

    image_size: int = 518
    patch_rows: int = 16
    patch_cols: int = 16
    quality_cells_per_axis: int = 8
    total_points: int = 1024
    iterations: int = 3
    round_min_points: tuple[int, ...] = (4, 1, 0)
    max_points_per_patch: int = 20
    exploration_ratio: float = 0.25
    jitter_ratio: float = 0.35
    temperature: float = 0.1
    match_threshold: float = 0.01
    anchor_threshold: float = 0.05
    budget_score_power: float = 2.0
    geometry_match_threshold_px: float = 6.0
    geometry_depth_tolerance: float = 0.01
    geometry_support_stride: int = 4

    def __post_init__(self) -> None:
        """验证配置能产生符合 v0 约束的采样预算。"""
        if (
            self.image_size <= 0
            or self.patch_rows <= 0
            or self.patch_cols <= 0
            or self.quality_cells_per_axis <= 0
        ):
            raise ValueError("图像尺寸和 Patch 网格必须为正数")
        if self.total_points <= 0 or self.iterations <= 0:
            raise ValueError("点数和迭代轮次必须为正数")
        if len(self.round_min_points) < self.iterations:
            raise ValueError("每轮都必须定义最低点数")
        if any(value < 0 for value in self.round_min_points):
            raise ValueError("Patch 最低点数不能为负数")
        if self.max_points_per_patch < max(self.round_min_points):
            raise ValueError("Patch 点数上限不能小于最低点数")
        if not 0.0 < self.exploration_ratio <= 1.0:
            raise ValueError("探索比例必须位于 (0, 1] 内")
        if (
            self.jitter_ratio < 0.0
            or self.temperature <= 0.0
            or self.match_threshold < 0.0
            or self.anchor_threshold < 0.0
            or self.budget_score_power <= 0.0
            or self.geometry_match_threshold_px <= 0.0
            or self.geometry_depth_tolerance < 0.0
            or self.geometry_support_stride <= 0
        ):
            raise ValueError("扰动、温度和阈值参数无效")
        if self.total_points != self.patch_count * self.round_min_points[0]:
            raise ValueError("首轮总点数必须等于 Patch 数乘以首轮最低点数")
        if self.total_points > self.patch_count * self.max_points_per_patch:
            raise ValueError("Patch 点数上限无法容纳总点数")
        if any(self.total_points < self.patch_count * value for value in self.round_min_points):
            raise ValueError("某轮最低点数超过总预算")

    @property
    def patch_count(self) -> int:
        """返回管理 Patch 的总数量。"""
        return self.patch_rows * self.patch_cols

    @property
    def quality_cell_count(self) -> int:
        """返回单个管理 Patch 内局部质量格的总数量。"""
        return self.quality_cells_per_axis**2

    def minimum_for_round(self, round_index: int) -> int:
        """返回指定轮次的每 Patch 最低点数。"""
        if not 0 <= round_index < self.iterations:
            raise ValueError("轮次索引超出范围")
        return self.round_min_points[round_index]
