# PASM v0 的三轮点级联合重采样主流程。

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .config import PasmV0Config
from .matching import (
    PairSemanticEncoder,
    global_edge_weights,
    patch_feedback,
    sample_feature_map,
    select_anchor_matches,
)
from .sampling import PatchLayout, allocate_patch_budget, resample_in_patches


@dataclass
class RoundRecord:
    """记录一轮采样位置、Patch 预算、边权和语义价值。"""

    points_a: np.ndarray
    points_b: np.ndarray
    patch_ids_a: np.ndarray
    patch_ids_b: np.ndarray
    budget_a: np.ndarray
    budget_b: np.ndarray
    edge_weights: torch.Tensor
    patch_values_a: torch.Tensor
    patch_values_b: torch.Tensor
    anchor_indices_a: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    anchor_indices_b: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    anchor_scores: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))


@dataclass
class PasmV0Result:
    """保存最终点级匹配状态和全部轮次历史。"""

    points_a: np.ndarray
    points_b: np.ndarray
    patch_ids_a: np.ndarray
    patch_ids_b: np.ndarray
    edge_weights: torch.Tensor
    rounds: list[RoundRecord]


class PasmV0:
    """执行 Patch 管预算、全局点级匹配和 Patch 内重采样。"""

    def __init__(self, semantic_encoder: PairSemanticEncoder, config: PasmV0Config) -> None:
        """绑定可训练语义编码器和固定 v0 采样配置。"""
        self.semantic_encoder = semantic_encoder
        self.config = config
        self.layout = PatchLayout(config)

    def run(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        descriptor_grid_a: np.ndarray | torch.Tensor,
        descriptor_grid_b: np.ndarray | torch.Tensor,
        seed: int | None = None,
    ) -> PasmV0Result:
        """对一对图像的语义 token 和局部特征执行完整三轮流程。"""
        _validate_patch_tokens(tokens_a, tokens_b, self.config.patch_count)
        device = next(self.semantic_encoder.parameters()).device
        tokens_a = tokens_a.to(device)
        tokens_b = tokens_b.to(device)
        grid_a = _feature_tensor(descriptor_grid_a, device)
        grid_b = _feature_tensor(descriptor_grid_b, device)
        points_a, patch_ids_a = self.layout.initial_points(seed)
        points_b, patch_ids_b = self.layout.initial_points(_next_seed(seed))
        budget_a = np.bincount(patch_ids_a, minlength=self.config.patch_count).astype(np.int64)
        budget_b = np.bincount(patch_ids_b, minlength=self.config.patch_count).astype(np.int64)
        rounds = []
        for round_index in range(self.config.iterations):
            semantic_a, semantic_b, value_a, value_b = self.semantic_encoder(tokens_a, tokens_b)
            descriptors_a = sample_feature_map(grid_a, torch.from_numpy(points_a), (self.config.image_size,) * 2)
            descriptors_b = sample_feature_map(grid_b, torch.from_numpy(points_b), (self.config.image_size,) * 2)
            ids_a = torch.from_numpy(patch_ids_a).to(device)
            ids_b = torch.from_numpy(patch_ids_b).to(device)
            edge_weights = global_edge_weights(
                descriptors_a,
                descriptors_b,
                semantic_a[0],
                semantic_b[0],
                ids_a,
                ids_b,
                self.config.temperature,
            )
            anchor_indices_a = np.empty((0,), dtype=np.int64)
            anchor_indices_b = np.empty((0,), dtype=np.int64)
            anchor_scores = np.empty((0,), dtype=np.float32)
            if round_index != self.config.iterations - 1:
                anchors = select_anchor_matches(
                    edge_weights,
                    patch_ids_a,
                    patch_ids_b,
                    self.config.anchor_threshold,
                    self.config.max_points_per_patch,
                    patch_count=self.config.patch_count,
                    minimum_per_patch=self.config.minimum_for_round(round_index + 1),
                    total_points=self.config.total_points,
                )
                anchor_indices_a = anchors.indices_a
                anchor_indices_b = anchors.indices_b
                anchor_scores = anchors.scores
            rounds.append(
                RoundRecord(
                    points_a=points_a.copy(),
                    points_b=points_b.copy(),
                    patch_ids_a=patch_ids_a.copy(),
                    patch_ids_b=patch_ids_b.copy(),
                    budget_a=budget_a.copy(),
                    budget_b=budget_b.copy(),
                    edge_weights=edge_weights,
                    patch_values_a=value_a[0],
                    patch_values_b=value_b[0],
                    anchor_indices_a=anchor_indices_a.copy(),
                    anchor_indices_b=anchor_indices_b.copy(),
                    anchor_scores=anchor_scores.copy(),
                )
            )
            if round_index == self.config.iterations - 1:
                break
            feedback_a = patch_feedback(edge_weights, ids_a, self.config.patch_count)
            feedback_b = patch_feedback(edge_weights.transpose(0, 1), ids_b, self.config.patch_count)
            anchor_counts_a = _anchor_counts(patch_ids_a, anchor_indices_a, self.config.patch_count)
            anchor_counts_b = _anchor_counts(patch_ids_b, anchor_indices_b, self.config.patch_count)
            budget_a = allocate_patch_budget(
                (value_a[0] + feedback_a).detach().cpu().numpy() + _normalized_anchor_support(anchor_counts_a),
                self.config,
                round_index + 1,
                reserved=anchor_counts_a,
            )
            budget_b = allocate_patch_budget(
                (value_b[0] + feedback_b).detach().cpu().numpy() + _normalized_anchor_support(anchor_counts_b),
                self.config,
                round_index + 1,
                reserved=anchor_counts_b,
            )
            quality_a = edge_weights.max(dim=1).values.detach().cpu().numpy()
            quality_b = edge_weights.max(dim=0).values.detach().cpu().numpy()
            points_a, patch_ids_a = resample_in_patches(
                self.layout,
                points_a,
                patch_ids_a,
                quality_a,
                budget_a,
                self.config,
                seed=_round_seed(seed, round_index, 0),
                anchor_points=points_a[anchor_indices_a],
                anchor_patch_ids=patch_ids_a[anchor_indices_a],
                anchor_scores=anchor_scores,
            )
            points_b, patch_ids_b = resample_in_patches(
                self.layout,
                points_b,
                patch_ids_b,
                quality_b,
                budget_b,
                self.config,
                seed=_round_seed(seed, round_index, 1),
                anchor_points=points_b[anchor_indices_b],
                anchor_patch_ids=patch_ids_b[anchor_indices_b],
                anchor_scores=anchor_scores,
            )
        final = rounds[-1]
        return PasmV0Result(
            points_a=final.points_a,
            points_b=final.points_b,
            patch_ids_a=final.patch_ids_a,
            patch_ids_b=final.patch_ids_b,
            edge_weights=final.edge_weights,
            rounds=rounds,
        )


def _feature_tensor(feature_grid: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """将 LoMa 局部描述子图转换为模型设备上的三维张量。"""
    grid = torch.as_tensor(feature_grid, dtype=torch.float32, device=device)
    if grid.ndim != 3:
        raise ValueError("局部描述子图必须具有 [C, H, W] 形状")
    return grid


def _validate_patch_tokens(tokens_a: torch.Tensor, tokens_b: torch.Tensor, patch_count: int) -> None:
    """验证两图输入 token 恰好对应全部管理 Patch。"""
    if tokens_a.ndim != 3 or tokens_b.ndim != 3:
        raise ValueError("输入 token 必须具有 [B, P, C] 形状")
    if tokens_a.shape != tokens_b.shape or tokens_a.shape[0] != 1:
        raise ValueError("v0 当前要求两图单样本 token 形状一致")
    if tokens_a.shape[1] != patch_count:
        raise ValueError("输入 token 数量必须等于管理 Patch 数量")


def _next_seed(seed: int | None) -> int | None:
    """为 B 图生成与 A 图不同但可复现的初始化种子。"""
    return None if seed is None else seed + 1


def _round_seed(seed: int | None, round_index: int, image_index: int) -> int | None:
    """为每轮和每张图生成不同且可复现的重采样种子。"""
    return None if seed is None else seed + 10 + round_index * 2 + image_index


def _anchor_counts(patch_ids: np.ndarray, indices: np.ndarray, patch_count: int) -> np.ndarray:
    """统计一张图中每个 Patch 已保留的锚点数量。"""
    return np.bincount(patch_ids[indices], minlength=patch_count).astype(np.int64)


def _normalized_anchor_support(counts: np.ndarray) -> np.ndarray:
    """将锚点数量转换为可叠加到预算分数的零到一支持值。"""
    maximum = int(counts.max()) if len(counts) else 0
    return counts.astype(np.float64) / maximum if maximum else np.zeros_like(counts, dtype=np.float64)
