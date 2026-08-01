# PASM v0 的双图语义条件化和全局点级匹配。

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class AnchorMatches:
    """保存可跨轮原样保留的高置信点级匹配边。"""

    indices_a: np.ndarray
    indices_b: np.ndarray
    scores: np.ndarray


class PairSemanticEncoder(nn.Module):
    """使用一层共享双向 Cross-Attention 条件化管理 Patch 语义。"""

    def __init__(self, token_dim: int, hidden_dim: int = 128, heads: int = 4) -> None:
        """创建语义投影、共享注意力层和 Patch 价值头。"""
        super().__init__()
        if token_dim <= 0 or hidden_dim <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError("语义维度和注意力头数无效")
        self.projection = nn.Linear(token_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回两图的条件化 Patch 语义和每个 Patch 的语义价值。"""
        _validate_tokens(tokens_a, tokens_b)
        base_a = self.normalization(self.projection(tokens_a))
        base_b = self.normalization(self.projection(tokens_b))
        message_a, _ = self.attention(base_a, base_b, base_b, need_weights=False)
        message_b, _ = self.attention(base_b, base_a, base_a, need_weights=False)
        semantic_a = functional.normalize(base_a + message_a, dim=-1, eps=1e-6)
        semantic_b = functional.normalize(base_b + message_b, dim=-1, eps=1e-6)
        value_a = torch.sigmoid(self.value_head(semantic_a).squeeze(-1))
        value_b = torch.sigmoid(self.value_head(semantic_b).squeeze(-1))
        return semantic_a, semantic_b, value_a, value_b


def sample_feature_map(
    feature_map: torch.Tensor, points_xy: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    """从稠密局部特征图双线性读取指定像素坐标的描述子。"""
    if feature_map.ndim != 3:
        raise ValueError("特征图必须具有 [C, H, W] 形状")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("采样点必须具有 [N, 2] 形状")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正数")
    points = points_xy.to(device=feature_map.device, dtype=feature_map.dtype)
    grid = points.clone()
    grid[:, 0] = (grid[:, 0] + 0.5) * 2.0 / width - 1.0
    grid[:, 1] = (grid[:, 1] + 0.5) * 2.0 / height - 1.0
    sampled = functional.grid_sample(
        feature_map.unsqueeze(0), grid.view(1, 1, -1, 2), align_corners=False, mode="bilinear"
    )
    return functional.normalize(sampled[0, :, 0].transpose(0, 1), dim=1, eps=1e-6)


def global_edge_weights(
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    semantic_a: torch.Tensor,
    semantic_b: torch.Tensor,
    patch_ids_a: torch.Tensor,
    patch_ids_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """计算所有当前点对的局部与联合语义 dual-softmax 边权。"""
    if temperature <= 0.0:
        raise ValueError("匹配温度必须为正数")
    _validate_descriptors(descriptors_a, descriptors_b)
    _validate_semantics(semantic_a, semantic_b, patch_ids_a, patch_ids_b)
    local_a = functional.normalize(descriptors_a, dim=1, eps=1e-6)
    local_b = functional.normalize(descriptors_b, dim=1, eps=1e-6)
    point_semantic_a = semantic_a[patch_ids_a]
    point_semantic_b = semantic_b[patch_ids_b]
    local_scores = local_a @ local_b.transpose(0, 1)
    semantic_scores = point_semantic_a @ point_semantic_b.transpose(0, 1)
    logits = (local_scores + semantic_scores) / temperature
    return torch.sqrt(functional.softmax(logits, dim=1) * functional.softmax(logits, dim=0))


def patch_feedback(
    edge_weights: torch.Tensor, patch_ids: torch.Tensor, patch_count: int
) -> torch.Tensor:
    """聚合每个源点的最强边质量为对应 Patch 的平均反馈。"""
    if edge_weights.ndim != 2 or edge_weights.shape[0] != patch_ids.numel():
        raise ValueError("边权和源点 Patch 索引的形状不一致")
    if patch_count <= 0 or torch.any(patch_ids < 0) or torch.any(patch_ids >= patch_count):
        raise ValueError("Patch 索引超出范围")
    quality = edge_weights.max(dim=1).values
    totals = torch.zeros(patch_count, dtype=quality.dtype, device=quality.device)
    counts = torch.zeros(patch_count, dtype=quality.dtype, device=quality.device)
    totals.scatter_add_(0, patch_ids, quality)
    counts.scatter_add_(0, patch_ids, torch.ones_like(quality))
    return totals / counts.clamp_min(1.0)


def mutual_match_indices(edge_weights: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """返回同时互为最强边且超过阈值的点索引。"""
    if edge_weights.ndim != 2 or threshold < 0.0:
        raise ValueError("边权矩阵或匹配阈值无效")
    best_b = edge_weights.argmax(dim=1)
    best_a = edge_weights.argmax(dim=0)
    source = torch.arange(edge_weights.shape[0], device=edge_weights.device)
    keep = best_a[best_b] == source
    keep &= edge_weights[source, best_b] >= threshold
    return source[keep], best_b[keep]


def select_anchor_matches(
    edge_weights: torch.Tensor,
    patch_ids_a: np.ndarray,
    patch_ids_b: np.ndarray,
    threshold: float,
    max_points_per_patch: int,
    patch_count: int | None = None,
    minimum_per_patch: int = 0,
    total_points: int | None = None,
) -> AnchorMatches:
    """按边权选择满足 Patch 容量和下一轮总预算的互相最佳锚点。"""
    if (
        edge_weights.ndim != 2
        or threshold < 0.0
        or max_points_per_patch <= 0
        or minimum_per_patch < 0
    ):
        raise ValueError("锚点边权、阈值或 Patch 上限无效")
    ids_a = np.asarray(patch_ids_a, dtype=np.int64).reshape(-1)
    ids_b = np.asarray(patch_ids_b, dtype=np.int64).reshape(-1)
    if ids_a.size != edge_weights.shape[0] or ids_b.size != edge_weights.shape[1]:
        raise ValueError("锚点 Patch 索引与边权矩阵形状不一致")
    if np.any(ids_a < 0) or np.any(ids_b < 0):
        raise ValueError("锚点 Patch 索引不能为负数")
    _validate_anchor_budget(patch_count, minimum_per_patch, total_points, max_points_per_patch, ids_a, ids_b)
    indices_a, indices_b = mutual_match_indices(edge_weights, threshold)
    if indices_a.numel() == 0:
        return _empty_anchors()
    scores = edge_weights[indices_a, indices_b]
    order = torch.argsort(scores, descending=True, stable=True)
    source = indices_a[order].detach().cpu().numpy().astype(np.int64, copy=False)
    target = indices_b[order].detach().cpu().numpy().astype(np.int64, copy=False)
    confidence = scores[order].detach().cpu().numpy().astype(np.float32, copy=False)
    counts_a: dict[int, int] = {}
    counts_b: dict[int, int] = {}
    required_a = np.full(patch_count, minimum_per_patch, dtype=np.int64) if patch_count else None
    required_b = np.full(patch_count, minimum_per_patch, dtype=np.int64) if patch_count else None
    total_required_a = int(required_a.sum()) if required_a is not None else 0
    total_required_b = int(required_b.sum()) if required_b is not None else 0
    selected_a = []
    selected_b = []
    selected_scores = []
    for index_a, index_b, score in zip(source, target, confidence, strict=True):
        patch_a = int(ids_a[index_a])
        patch_b = int(ids_b[index_b])
        if counts_a.get(patch_a, 0) >= max_points_per_patch:
            continue
        if counts_b.get(patch_b, 0) >= max_points_per_patch:
            continue
        next_a = counts_a.get(patch_a, 0) + 1
        next_b = counts_b.get(patch_b, 0) + 1
        if required_a is not None:
            added_a = max(minimum_per_patch, next_a) - int(required_a[patch_a])
            added_b = max(minimum_per_patch, next_b) - int(required_b[patch_b])
            if total_required_a + added_a > total_points or total_required_b + added_b > total_points:
                continue
            required_a[patch_a] += added_a
            required_b[patch_b] += added_b
            total_required_a += added_a
            total_required_b += added_b
        counts_a[patch_a] = next_a
        counts_b[patch_b] = next_b
        selected_a.append(int(index_a))
        selected_b.append(int(index_b))
        selected_scores.append(float(score))
    return AnchorMatches(
        indices_a=np.asarray(selected_a, dtype=np.int64),
        indices_b=np.asarray(selected_b, dtype=np.int64),
        scores=np.asarray(selected_scores, dtype=np.float32),
    )


def _empty_anchors() -> AnchorMatches:
    """返回统一形状的空锚点结果。"""
    return AnchorMatches(
        indices_a=np.empty((0,), dtype=np.int64),
        indices_b=np.empty((0,), dtype=np.int64),
        scores=np.empty((0,), dtype=np.float32),
    )


def _validate_anchor_budget(
    patch_count: int | None,
    minimum_per_patch: int,
    total_points: int | None,
    max_points_per_patch: int,
    ids_a: np.ndarray,
    ids_b: np.ndarray,
) -> None:
    """验证可选的下一轮锚点总预算约束。"""
    if patch_count is None:
        if minimum_per_patch != 0 or total_points is not None:
            raise ValueError("锚点总预算约束必须提供 Patch 总数")
        return
    if total_points is None:
        raise ValueError("锚点总预算约束必须提供总采样点数")
    if patch_count <= 0 or minimum_per_patch > max_points_per_patch:
        raise ValueError("锚点总预算的 Patch 参数无效")
    if np.any(ids_a >= patch_count) or np.any(ids_b >= patch_count):
        raise ValueError("锚点 Patch 索引超出范围")
    if total_points is not None and total_points < patch_count * minimum_per_patch:
        raise ValueError("总采样预算不足以覆盖 Patch 最低点数")


def _validate_tokens(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> None:
    """验证两图语义 token 的批量和通道维度一致。"""
    if tokens_a.ndim != 3 or tokens_b.ndim != 3:
        raise ValueError("语义 token 必须具有 [B, P, C] 形状")
    if tokens_a.shape[0] != tokens_b.shape[0] or tokens_a.shape[2] != tokens_b.shape[2]:
        raise ValueError("两图语义 token 的批量和通道维度必须一致")


def _validate_descriptors(descriptors_a: torch.Tensor, descriptors_b: torch.Tensor) -> None:
    """验证两图局部描述子的通道维度一致。"""
    if descriptors_a.ndim != 2 or descriptors_b.ndim != 2:
        raise ValueError("局部描述子必须具有 [N, C] 形状")
    if descriptors_a.shape[1] != descriptors_b.shape[1]:
        raise ValueError("两图局部描述子通道数必须一致")


def _validate_semantics(
    semantic_a: torch.Tensor,
    semantic_b: torch.Tensor,
    patch_ids_a: torch.Tensor,
    patch_ids_b: torch.Tensor,
) -> None:
    """验证语义 Patch 表和点所属 Patch 索引可用于全局匹配。"""
    if semantic_a.ndim != 2 or semantic_b.ndim != 2 or semantic_a.shape[1] != semantic_b.shape[1]:
        raise ValueError("Patch 语义必须具有相同通道数的 [P, C] 形状")
    if patch_ids_a.ndim != 1 or patch_ids_b.ndim != 1:
        raise ValueError("点所属 Patch 索引必须为一维")
    if torch.any(patch_ids_a < 0) or torch.any(patch_ids_a >= len(semantic_a)):
        raise ValueError("A 图 Patch 索引超出范围")
    if torch.any(patch_ids_b < 0) or torch.any(patch_ids_b >= len(semantic_b)):
        raise ValueError("B 图 Patch 索引超出范围")
