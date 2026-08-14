# PASM v0 的集中配置定义。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PasmV0Config:
    """保存 PASM v0 的网格、预算和重采样参数。"""

    image_size: int = 518
    patch_rows: int = 16
    patch_cols: int = 16
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
    initial_patch_subset: bool = False

    def __post_init__(self) -> None:
        """验证配置能产生符合 v0 约束的采样预算。"""
        if self.image_size <= 0 or self.patch_rows <= 0 or self.patch_cols <= 0:
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
        if self.round_min_points[0] and self.total_points != self.patch_count * self.round_min_points[0]:
            raise ValueError("首轮总点数必须等于 Patch 数乘以首轮最低点数")
        if self.initial_patch_subset and not (
            self.minimum_for_round(0) == 0 and self.total_points <= self.patch_count
        ):
            raise ValueError("首轮 Patch 子集采样要求首轮最低点数为零且总点数不超过 Patch 数")
        if self.total_points > self.patch_count * self.max_points_per_patch:
            raise ValueError("Patch 点数上限无法容纳总点数")
        if any(self.total_points < self.patch_count * value for value in self.round_min_points):
            raise ValueError("某轮最低点数超过总预算")

    @property
    def patch_count(self) -> int:
        """返回管理 Patch 的总数量。"""
        return self.patch_rows * self.patch_cols

    def minimum_for_round(self, round_index: int) -> int:
        """返回指定轮次的每 Patch 最低点数。"""
        if not 0 <= round_index < self.iterations:
            raise ValueError("轮次索引超出范围")
        return self.round_min_points[round_index]


def dino_token_static_config(patch_grid: int = 37) -> PasmV0Config:
    """返回使用指定 DINO token 网格的单轮静态配置。"""
    if patch_grid not in (16, 37):
        raise ValueError("静态 Patch 网格只支持 16 或 37")
    return PasmV0Config(
        patch_rows=patch_grid,
        patch_cols=patch_grid,
        total_points=1024,
        iterations=1,
        round_min_points=(0,),
        max_points_per_patch=20 if patch_grid == 16 else 4,
        budget_score_power=1.0,
        geometry_support_stride=2,
    )


def legacy_iterative_37_config() -> PasmV0Config:
    """返回 37×37 首轮无放回采样的旧版三轮配置。"""
    return PasmV0Config(
        patch_rows=37,
        patch_cols=37,
        total_points=1024,
        iterations=3,
        round_min_points=(0, 0, 0),
        max_points_per_patch=20,
        budget_score_power=2.0,
        geometry_match_threshold_px=6.0,
        geometry_support_stride=2,
        initial_patch_subset=True,
    )
