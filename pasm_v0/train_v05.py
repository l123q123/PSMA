"""训练 PASM v0.5 的残差稠密点融合头。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
from .dad import DaDDetections, FrozenDaDDetector, select_dad_candidate_pool
from .data import load_manifest
from .geometry import load_pair_geometry
from .infer import load_runner
from .point_fusion import create_point_fusion, point_candidate_logits
from .point_teacher import build_candidate_teacher
from .feature_cache import FeatureCache, cache_key


@dataclass(frozen=True)
class PointFusionTrainConfig:
    """v0.5 融合训练的可验证超参数。"""

    teacher_candidate_count: int = 4096
    teacher_sigma_px: float = 3.0
    teacher_search_radius_px: float = 12.0
    ranking_weight: float = 0.5
    ranking_margin: float = 0.2
    fusion_mode: str = "residual"
    ranking_mode: str = "hinge"
    ranking_temperature: float = 1.0
    suppression_weight: float = 1.0
    suppression_margin: float = 1.0
    border_margin_px: int = 8
    gate_penalty_scale: float = 3.0
    gate_quality_gamma: float = 2.0
    coverage_weight: float = 1.0
    local_preservation_weight: float = 0.25
    coverage_temperature: float = 0.5
    coverage_topk: int = 1024
    invisibility_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.teacher_candidate_count <= 0:
            raise ValueError("教师候选池大小必须为正数")
        if self.teacher_sigma_px <= 0.0 or self.teacher_search_radius_px <= 0.0:
            raise ValueError("教师 sigma 和搜索半径必须为正数")
        if self.ranking_weight < 0.0 or self.ranking_margin < 0.0:
            raise ValueError("排序损失权重和间隔不能为负数")
        if self.fusion_mode not in {"residual", "contrastive", "candidate_suppression", "semantic_coverage", "patch_gate"}:
            raise ValueError("融合模式必须是 residual、contrastive、candidate_suppression、semantic_coverage 或 patch_gate")
        if self.ranking_mode not in {"hinge", "soft_pairwise"}:
            raise ValueError("排序模式必须是 hinge 或 soft_pairwise")
        if self.ranking_temperature <= 0.0:
            raise ValueError("排序温度必须为正数")
        if self.suppression_weight < 0.0 or self.suppression_margin < 0.0:
            raise ValueError("候选抑制损失权重和间隔不能为负数")
        if self.border_margin_px < 0:
            raise ValueError("候选边界带宽不能为负数")
        if self.gate_penalty_scale <= 0.0 or self.gate_quality_gamma <= 0.0:
            raise ValueError("Patch 门控抑制尺度和指数必须为正数")
        if self.coverage_weight < 0.0 or self.local_preservation_weight < 0.0:
            raise ValueError("覆盖和局部保真损失权重不能为负数")
        if self.coverage_temperature <= 0.0 or self.coverage_topk <= 0:
            raise ValueError("覆盖温度和覆盖 Top-K 必须为正数")
        if self.invisibility_weight < 0.0:
            raise ValueError("几何不可见损失权重不能为负数")


def point_fusion_loss(
    logits_a: torch.Tensor,
    labels_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels_b: torch.Tensor,
    ranking_weight: float,
    ranking_margin: float = 0.2,
    ranking_mode: str = "hinge",
    ranking_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算双图候选点 soft BCE 与正负排序损失。"""
    _validate_loss_inputs(
        logits_a, labels_a, logits_b, labels_b, ranking_weight, ranking_margin,
        ranking_mode, ranking_temperature,
    )
    point = 0.5 * (
        F.binary_cross_entropy_with_logits(logits_a, labels_a)
        + F.binary_cross_entropy_with_logits(logits_b, labels_b)
    )
    rank_loss = _ranking_loss if ranking_mode == "hinge" else _soft_pairwise_ranking_loss
    if ranking_mode == "hinge":
        rank = 0.5 * (
            rank_loss(logits_a, labels_a, ranking_margin)
            + rank_loss(logits_b, labels_b, ranking_margin)
        )
    else:
        rank = 0.5 * (
            rank_loss(logits_a, labels_a, ranking_temperature)
            + rank_loss(logits_b, labels_b, ranking_temperature)
        )
    return point + ranking_weight * rank, point, rank


def candidate_suppression_loss(
    final_logits_a: torch.Tensor,
    penalties_a: torch.Tensor,
    labels_a: torch.Tensor,
    final_logits_b: torch.Tensor,
    penalties_b: torch.Tensor,
    labels_b: torch.Tensor,
    suppression_weight: float,
    suppression_margin: float,
    ranking_weight: float,
    ranking_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """保留几何正候选、压低可靠负候选，并忽略教师未知候选。"""
    if suppression_weight < 0.0 or suppression_margin < 0.0 or ranking_weight < 0.0:
        raise ValueError("候选抑制损失权重和间隔不能为负数")
    if ranking_temperature <= 0.0:
        raise ValueError("候选排序温度必须为正数")
    terms = [
        _candidate_suppression_view_loss(final_logits_a, penalties_a, labels_a, suppression_margin, ranking_temperature),
        _candidate_suppression_view_loss(final_logits_b, penalties_b, labels_b, suppression_margin, ranking_temperature),
    ]
    keep = 0.5 * (terms[0][0] + terms[1][0])
    suppress = 0.5 * (terms[0][1] + terms[1][1])
    rank = 0.5 * (terms[0][2] + terms[1][2])
    return keep + suppression_weight * suppress + ranking_weight * rank, keep, suppress, rank


def patch_gate_targets(
    labels: np.ndarray | torch.Tensor,
    visible: np.ndarray | torch.Tensor,
    candidates_xy: np.ndarray | torch.Tensor,
    image_size: int,
    patch_grid: int,
) -> torch.Tensor:
    """将候选几何教师按 Patch 取可见候选最大值，未知 Patch 标为 -1。"""
    label_tensor = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    visible_tensor = torch.as_tensor(visible, dtype=torch.bool).reshape(-1)
    candidates = torch.as_tensor(candidates_xy, dtype=torch.float32)
    if image_size <= 0 or patch_grid <= 0:
        raise ValueError("图像尺寸和 Patch 网格必须为正数")
    if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.shape[0] != label_tensor.numel():
        raise ValueError("候选坐标与教师标签形状不一致")
    if visible_tensor.shape != label_tensor.shape or not torch.isfinite(label_tensor).all():
        raise ValueError("候选可见性与教师标签形状无效")
    if torch.any(label_tensor < 0.0) or torch.any(label_tensor > 1.0):
        raise ValueError("候选教师标签必须位于 [0, 1]")
    if not torch.isfinite(candidates).all() or torch.any(candidates < 0.0) or torch.any(candidates >= image_size):
        raise ValueError("候选坐标超出图像范围")
    targets = torch.full((patch_grid * patch_grid,), -1.0, dtype=torch.float32)
    if not torch.any(visible_tensor):
        return targets
    cols = torch.floor(candidates[:, 0] * patch_grid / image_size).to(torch.long).clamp_(0, patch_grid - 1)
    rows = torch.floor(candidates[:, 1] * patch_grid / image_size).to(torch.long).clamp_(0, patch_grid - 1)
    patch_ids = rows * patch_grid + cols
    visible_ids = patch_ids[visible_tensor]
    targets[visible_ids.unique()] = 0.0
    targets.scatter_reduce_(0, visible_ids, label_tensor[visible_tensor], reduce="amax", include_self=True)
    return targets


def patch_gate_loss(
    gates_a: torch.Tensor,
    targets_a: torch.Tensor,
    gates_b: torch.Tensor,
    targets_b: torch.Tensor,
) -> torch.Tensor:
    """在可见教师 Patch 上独立拟合 DaD 放行概率。"""
    losses = []
    for gates, targets in ((gates_a, targets_a), (gates_b, targets_b)):
        target = torch.as_tensor(targets, dtype=gates.dtype, device=gates.device).reshape(-1)
        if gates.ndim != 1 or gates.shape != target.shape or not torch.isfinite(gates).all() or not torch.isfinite(target).all():
            raise ValueError("Patch 门控预测和教师必须是同形有限一维张量")
        if torch.any(gates < 0.0) or torch.any(gates > 1.0) or torch.any(target < -1.0) or torch.any(target > 1.0):
            raise ValueError("Patch 门控预测必须位于 [0,1]，教师必须位于 [-1,1]")
        valid = target >= 0.0
        losses.append(F.binary_cross_entropy(gates[valid], target[valid]) if torch.any(valid) else gates.sum() * 0.0)
    return 0.5 * (losses[0] + losses[1])


def patch_support_targets(
    labels: np.ndarray | torch.Tensor,
    visible: np.ndarray | torch.Tensor,
    candidates_xy: np.ndarray | torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Patch 聚合可见候选的几何软支持，返回支持值和可见有效掩码。"""
    label_tensor = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    visible_tensor = torch.as_tensor(visible, dtype=torch.bool).reshape(-1)
    candidates = torch.as_tensor(candidates_xy, dtype=torch.float32)
    height, width = _image_size(image_size)
    if patch_grid <= 0:
        raise ValueError("Patch 网格必须为正数")
    if candidates.ndim != 2 or candidates.shape != (label_tensor.numel(), 2):
        raise ValueError("候选坐标与教师标签形状不一致")
    if visible_tensor.shape != label_tensor.shape or not torch.isfinite(label_tensor).all():
        raise ValueError("候选可见性与教师标签形状无效")
    if torch.any(label_tensor < 0.0) or torch.any(label_tensor > 1.0):
        raise ValueError("候选教师标签必须位于 [0,1]")
    patch_count = patch_grid * patch_grid
    support = torch.zeros(patch_count, dtype=torch.float32)
    valid = torch.zeros(patch_count, dtype=torch.bool)
    if not torch.any(visible_tensor):
        return support, valid
    patch_ids = _candidate_patch_ids(candidates, height, width, patch_grid)
    visible_ids = patch_ids[visible_tensor]
    valid[visible_ids.unique()] = True
    support.scatter_reduce_(
        0,
        visible_ids,
        label_tensor[visible_tensor],
        reduce="amax",
        include_self=True,
    )
    return support, valid


def soft_patch_occupancy(
    candidate_logits: torch.Tensor,
    candidates_xy: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    """将候选分数转为可微 Top-K 软选择，并得到每个 Patch 的软占用。"""
    logits = _candidate_logit_vector(candidate_logits)
    height, width = _image_size(image_size)
    if patch_grid <= 0 or topk <= 0 or temperature <= 0.0:
        raise ValueError("Patch 网格、Top-K 和温度必须为正数")
    candidates = _candidate_coordinates(candidates_xy, logits, height, width)
    patch_ids = _candidate_patch_ids(candidates, height, width, patch_grid)
    threshold = torch.topk(logits.detach(), min(topk, logits.numel())).values[-1]
    selected = torch.sigmoid((logits - threshold) / temperature).clamp(max=1.0 - 1e-6)
    log_not_selected = torch.log1p(-selected)
    patch_log_not_selected = torch.zeros(
        patch_grid * patch_grid, dtype=logits.dtype, device=logits.device
    )
    patch_log_not_selected.scatter_add_(0, patch_ids, log_not_selected)
    return 1.0 - torch.exp(patch_log_not_selected)


def dense_soft_patch_occupancy(
    dense_logits: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    """将稠密融合 logits 转为可微 Top-K 软选择，聚合每个 Patch 的占用。"""
    if dense_logits.ndim != 4 or dense_logits.shape[:2] != (1, 1):
        raise ValueError("稠密融合 logits 必须是 [1, 1, H, W]")
    height, width = _image_size(image_size)
    if dense_logits.shape[-2:] != (height, width):
        raise ValueError("稠密融合 logits 与图像尺寸不一致")
    if patch_grid <= 0 or topk <= 0 or temperature <= 0.0:
        raise ValueError("Patch 网格、Top-K 和温度必须为正数")
    logits = dense_logits[0, 0]
    threshold = torch.topk(logits.detach().reshape(-1), min(topk, logits.numel())).values[-1]
    selected = torch.sigmoid((logits - threshold) / temperature).clamp(max=1.0 - 1e-6)
    patch_height = height / patch_grid
    patch_width = width / patch_grid
    row_ids = torch.div(torch.arange(height, device=logits.device)[:, None], patch_height, rounding_mode="floor").to(torch.long)
    col_ids = torch.div(torch.arange(width, device=logits.device)[None, :], patch_width, rounding_mode="floor").to(torch.long)
    patch_ids = (row_ids.clamp_max(patch_grid - 1) * patch_grid + col_ids.clamp_max(patch_grid - 1)).reshape(-1)
    log_not_selected = torch.log1p(-selected.reshape(-1))
    patch_log_not_selected = torch.zeros(patch_grid * patch_grid, dtype=logits.dtype, device=logits.device)
    patch_log_not_selected.scatter_add_(0, patch_ids, log_not_selected)
    return 1.0 - torch.exp(patch_log_not_selected)


def dense_geometric_invisibility_loss(
    dense_logits: torch.Tensor,
    fully_invisible_patches: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    """惩罚确认完全几何不可见 Patch 获得的稠密 Soft Top-K 占用。"""
    occupancy = dense_soft_patch_occupancy(dense_logits, image_size, patch_grid, topk, temperature)
    invisible = torch.as_tensor(fully_invisible_patches, dtype=torch.bool, device=occupancy.device).reshape(-1)
    if invisible.shape != occupancy.shape:
        raise ValueError("完全不可见 Patch 掩码必须与 Patch 数量一致")
    return occupancy[invisible].mean() if torch.any(invisible) else occupancy.sum() * 0.0


def fully_invisible_patch_mask(
    candidates_xy: torch.Tensor,
    visible: np.ndarray | torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
) -> torch.Tensor:
    """只标记拥有 DaD 候选且其中没有几何可见候选的 Patch。"""
    height, width = _image_size(image_size)
    candidates = torch.as_tensor(candidates_xy, dtype=torch.float32)
    visibility = torch.as_tensor(visible, dtype=torch.bool).reshape(-1)
    if candidates.ndim != 2 or candidates.shape != (visibility.numel(), 2):
        raise ValueError("候选坐标与候选可见性掩码形状不一致")
    patch_ids = _candidate_patch_ids(candidates, height, width, patch_grid)
    patch_count = patch_grid * patch_grid
    has_candidate = torch.zeros(patch_count, dtype=torch.bool)
    has_candidate[patch_ids.unique()] = True
    has_visible = torch.zeros_like(has_candidate)
    if torch.any(visibility):
        has_visible[patch_ids[visibility].unique()] = True
    return has_candidate & ~has_visible


def dense_semantic_coverage_loss(
    dense_logits: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    """要求有几何可靠支持的 Patch 在稠密融合 Top-K 中被代表。"""
    occupancy = dense_soft_patch_occupancy(dense_logits, image_size, patch_grid, topk, temperature)
    support_tensor, valid_tensor = _patch_support_and_valid(support, valid, occupancy)
    active = valid_tensor & (support_tensor > 0.0)
    if not torch.any(active):
        return occupancy.sum() * 0.0
    weights = support_tensor[active].detach()
    return -(weights * torch.log(occupancy[active].clamp_min(1e-6))).sum() / weights.sum().clamp_min(1e-6)


def dense_local_dad_preservation_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    temperature: float,
) -> torch.Tensor:
    """在有几何支持的 Patch 内保留稠密 DaD 响应的相对排序。"""
    if temperature <= 0.0:
        raise ValueError("局部保真温度必须为正数")
    if (
        fused_logits.ndim != 4
        or fused_logits.shape[:2] != (1, 1)
        or fused_logits.shape != base_logits.shape
    ):
        raise ValueError("融合/基础稠密 logits 必须是同形的 [1, 1, H, W]")
    height, width = _image_size(image_size)
    if fused_logits.shape[-2:] != (height, width):
        raise ValueError("稠密 logits 与图像尺寸不一致")
    reference = fused_logits.new_zeros(patch_grid * patch_grid)
    support_tensor, valid_tensor = _patch_support_and_valid(support, valid, reference)
    active = valid_tensor & (support_tensor > 0.0)
    if not torch.any(active):
        return fused_logits.sum() * 0.0
    patch_height = height / patch_grid
    patch_width = width / patch_grid
    row_ids = torch.div(
        torch.arange(height, device=fused_logits.device)[:, None], patch_height, rounding_mode="floor"
    ).to(torch.long)
    col_ids = torch.div(
        torch.arange(width, device=fused_logits.device)[None, :], patch_width, rounding_mode="floor"
    ).to(torch.long)
    patch_ids = row_ids.clamp_max(patch_grid - 1) * patch_grid + col_ids.clamp_max(patch_grid - 1)
    fused = fused_logits[0, 0]
    base = base_logits.detach()[0, 0]
    terms: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for patch_id in torch.nonzero(active, as_tuple=False).reshape(-1):
        member = patch_ids == patch_id
        target = torch.softmax(base[member] / temperature, dim=0)
        prediction = torch.log_softmax(fused[member] / temperature, dim=0)
        terms.append(F.kl_div(prediction, target, reduction="sum"))
        weights.append(support_tensor[patch_id].detach())
    weight_tensor = torch.stack(weights)
    return (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def geometric_invisibility_loss(
    candidate_logits: torch.Tensor,
    candidates_xy: torch.Tensor,
    visible: np.ndarray | torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    """惩罚完全几何不可见 Patch 获得的 Soft Top-K 占用。"""
    logits = _candidate_logit_vector(candidate_logits)
    height, width = _image_size(image_size)
    candidates = _candidate_coordinates(candidates_xy, logits, height, width)
    visible_mask = torch.as_tensor(visible, dtype=torch.bool, device=logits.device).reshape(-1)
    if visible_mask.shape != logits.shape:
        raise ValueError("候选可见性掩码必须与候选分数同形")
    if patch_grid <= 0 or topk <= 0 or temperature <= 0.0:
        raise ValueError("Patch 网格、Top-K 和温度必须为正数")
    patch_ids = _candidate_patch_ids(candidates, height, width, patch_grid)
    patch_count = patch_grid * patch_grid
    has_candidate = torch.zeros(patch_count, dtype=torch.bool, device=logits.device)
    has_candidate[patch_ids.unique()] = True
    has_visible = torch.zeros_like(has_candidate)
    if torch.any(visible_mask):
        has_visible[patch_ids[visible_mask].unique()] = True
    fully_invisible = has_candidate & ~has_visible
    if not torch.any(fully_invisible):
        return logits.sum() * 0.0
    occupancy = soft_patch_occupancy(
        logits, candidates, image_size, patch_grid, topk, temperature,
    )
    return occupancy[fully_invisible].mean()


def semantic_coverage_loss(
    candidate_logits: torch.Tensor,
    candidates_xy: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    """惩罚有几何可靠支持却未被最终候选集代表的 Patch。"""
    occupancy = soft_patch_occupancy(
        candidate_logits, candidates_xy, image_size, patch_grid, topk, temperature
    )
    support_tensor, valid_tensor = _patch_support_and_valid(support, valid, occupancy)
    active = valid_tensor & (support_tensor > 0.0)
    if not torch.any(active):
        return occupancy.sum() * 0.0
    weights = support_tensor[active].detach()
    return -(weights * torch.log(occupancy[active].clamp_min(1e-6))).sum() / weights.sum().clamp_min(1e-6)


def local_dad_preservation_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    candidates_xy: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    image_size: int | tuple[int, int],
    patch_grid: int,
    temperature: float,
) -> torch.Tensor:
    """在可靠 Patch 内保持融合前后 DaD 候选的相对排序分布。"""
    fused = _candidate_logit_vector(fused_logits)
    base = _candidate_logit_vector(base_logits, device=fused.device, dtype=fused.dtype)
    if fused.shape != base.shape or temperature <= 0.0:
        raise ValueError("融合/基础候选分数形状不一致，或温度无效")
    height, width = _image_size(image_size)
    candidates = _candidate_coordinates(candidates_xy, fused, height, width)
    patch_ids = _candidate_patch_ids(candidates, height, width, patch_grid)
    support_tensor, valid_tensor = _patch_support_and_valid(support, valid, fused.new_zeros(patch_grid * patch_grid))
    terms: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for patch_id in torch.nonzero(valid_tensor & (support_tensor > 0.0), as_tuple=False).reshape(-1):
        member = torch.nonzero(patch_ids == patch_id, as_tuple=False).reshape(-1)
        if member.numel() < 2:
            continue
        target = torch.softmax(base[member].detach() / temperature, dim=0)
        prediction = torch.log_softmax(fused[member] / temperature, dim=0)
        terms.append(F.kl_div(prediction, target, reduction="sum"))
        weights.append(support_tensor[patch_id].detach())
    if not terms:
        return fused.sum() * 0.0
    weight_tensor = torch.stack(weights)
    return (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def train_model(
    manifest_path: str | Path,
    dino_weights: str | Path,
    dad_weights: str | Path,
    semantic_checkpoint: str | Path,
    output_path: str | Path,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    device: str = "auto",
    max_pairs: int | None = None,
    patch_grid: int = 37,
    teacher_candidate_count: int = 4096,
    teacher_sigma_px: float = 3.0,
    teacher_search_radius_px: float = 12.0,
    ranking_weight: float = 0.5,
    fusion_mode: str = "residual",
    ranking_mode: str = "hinge",
    ranking_temperature: float = 1.0,
    suppression_weight: float = 1.0,
    suppression_margin: float = 1.0,
    border_margin_px: int = 8,
    gate_penalty_scale: float = 3.0,
    gate_quality_gamma: float = 2.0,
    coverage_weight: float = 1.0,
    local_preservation_weight: float = 0.25,
    coverage_temperature: float = 0.5,
    coverage_topk: int = 1024,
    invisibility_weight: float = 1.0,
    seed: int = 0,
    resume_path: str | Path | None = None,
    feature_cache_path: str | Path | None = None,
) -> Path:
    """冻结 DaD/DINO/语义编码器，仅训练 v0.5 稠密融合头。"""
    if epochs <= 0 or learning_rate <= 0.0 or patch_grid not in {16, 37}:
        raise ValueError("训练轮数、学习率或 Patch 网格无效")
    settings = PointFusionTrainConfig(
        teacher_candidate_count=teacher_candidate_count,
        teacher_sigma_px=teacher_sigma_px,
        teacher_search_radius_px=teacher_search_radius_px,
        ranking_weight=ranking_weight,
        fusion_mode=fusion_mode,
        ranking_mode=ranking_mode,
        ranking_temperature=ranking_temperature,
        suppression_weight=suppression_weight,
        suppression_margin=suppression_margin,
        border_margin_px=border_margin_px,
        gate_penalty_scale=gate_penalty_scale,
        gate_quality_gamma=gate_quality_gamma,
        coverage_weight=coverage_weight,
        local_preservation_weight=local_preservation_weight,
        coverage_temperature=coverage_temperature,
        coverage_topk=coverage_topk,
        invisibility_weight=invisibility_weight,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = resolve_device(device)
    source_runner = load_runner(semantic_checkpoint, selected_device)
    source_config = source_runner.config
    if source_config.patch_rows != patch_grid or source_config.patch_cols != patch_grid:
        raise ValueError("语义检查点的 Patch 网格与 --patch-grid 不一致")
    semantic_encoder = source_runner.semantic_encoder.eval()
    for parameter in semantic_encoder.parameters():
        parameter.requires_grad = False
    feature_cache = FeatureCache(feature_cache_path) if feature_cache_path is not None else None
    dino = None if feature_cache is not None else DINOv2PatchEncoder(dino_weights, source_config.image_size).to(selected_device).eval()
    detector = None if feature_cache is not None else FrozenDaDDetector(dad_weights, source_config.image_size, selected_device)
    fusion_kwargs = (
        {"penalty_scale": settings.gate_penalty_scale, "quality_gamma": settings.gate_quality_gamma}
        if settings.fusion_mode == "patch_gate" else {}
    )
    fusion = create_point_fusion(settings.fusion_mode, **fusion_kwargs).to(selected_device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=learning_rate)
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    if resume_path is not None:
        start_epoch, history, best_loss = _load_fusion_checkpoint(
            Path(resume_path), fusion, optimizer, selected_device, settings,
        )
        if start_epoch >= epochs:
            raise ValueError("目标训练轮数必须大于检查点已完成轮数")
    pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("最多训练图像对数必须为正数")
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("训练清单中没有正图像对")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        fusion.train()
        total_sum = point_sum = rank_sum = suppress_sum = coverage_sum = local_sum = invisibility_sum = 0.0
        progress = tqdm(pairs, total=len(pairs), unit="对", dynamic_ncols=True)
        for pair_index, pair in enumerate(progress, start=1):
            image_a = load_resized_rgb(pair.image_a, source_config.image_size)
            image_b = load_resized_rgb(pair.image_b, source_config.image_size)
            geometry = load_pair_geometry(pair, source_config.image_size)
            if feature_cache is None:
                detection_a = detector.detect(image_a)
                detection_b = detector.detect(image_b)
            else:
                detection_a = _detections_from_cache(feature_cache.load(cache_key(pair.image_a)))
                detection_b = _detections_from_cache(feature_cache.load(cache_key(pair.image_b)))
            candidates_a, _ = select_dad_candidate_pool(detection_a, settings.teacher_candidate_count)
            candidates_b, _ = select_dad_candidate_pool(detection_b, settings.teacher_candidate_count)
            labels_a, visible_a, _ = build_candidate_teacher(
                candidates_a, candidates_b, geometry, "a_to_b", settings.teacher_sigma_px,
                settings.teacher_search_radius_px, source_config.geometry_depth_tolerance,
            )
            labels_b, visible_b, _ = build_candidate_teacher(
                candidates_b, candidates_a, geometry, "b_to_a", settings.teacher_sigma_px,
                settings.teacher_search_radius_px, source_config.geometry_depth_tolerance,
            )
            with torch.inference_mode():
                if feature_cache is None:
                    images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(selected_device)
                    tokens = pool_dino_tokens(dino(images), source_config.patch_rows, source_config.patch_cols)
                else:
                    cache_a = feature_cache.load(cache_key(pair.image_a))
                    cache_b = feature_cache.load(cache_key(pair.image_b))
                    tokens = torch.from_numpy(np.stack((cache_a["tokens"], cache_b["tokens"]))).to(selected_device)
                semantic_a, semantic_b, _, _ = semantic_encoder(tokens[:1], tokens[1:])
            semantic_a = make_trainable_tensor(semantic_a)
            semantic_b = make_trainable_tensor(semantic_b)
            base_logits_a = _logit_tensor(detection_a.dense_logits, selected_device)
            base_logits_b = _logit_tensor(detection_b.dense_logits, selected_device)
            candidate_tensor_a = torch.from_numpy(candidates_a).to(selected_device)
            candidate_tensor_b = torch.from_numpy(candidates_b).to(selected_device)
            if settings.fusion_mode == "patch_gate":
                _, gates_a = fusion(base_logits_a, semantic_a, candidate_tensor_a, source_config.image_size)
                _, gates_b = fusion(base_logits_b, semantic_b, candidate_tensor_b, source_config.image_size)
                total = patch_gate_loss(
                    gates_a,
                    patch_gate_targets(labels_a, visible_a, candidates_a, source_config.image_size, source_config.patch_rows).to(selected_device),
                    gates_b,
                    patch_gate_targets(labels_b, visible_b, candidates_b, source_config.image_size, source_config.patch_rows).to(selected_device),
                )
                point = total
                suppress = total.detach() * 0.0
                rank = total.detach() * 0.0
                coverage = total.detach() * 0.0
                local = total.detach() * 0.0
                invisibility = total.detach() * 0.0
            elif settings.fusion_mode == "semantic_coverage":
                dad_features_a = _feature_tensor(detection_a.feature_map, selected_device)
                dad_features_b = _feature_tensor(detection_b.feature_map, selected_device)
                final_a = fusion(
                    dad_features_a, base_logits_a, semantic_a
                )
                final_b = fusion(
                    dad_features_b, base_logits_b, semantic_b
                )
                candidate_logits_a = point_candidate_logits(final_a, candidate_tensor_a, source_config.image_size)
                candidate_logits_b = point_candidate_logits(final_b, candidate_tensor_b, source_config.image_size)
                geo, point, rank = point_fusion_loss(
                    candidate_logits_a,
                    torch.from_numpy(labels_a).to(selected_device),
                    candidate_logits_b,
                    torch.from_numpy(labels_b).to(selected_device),
                    settings.ranking_weight,
                    settings.ranking_margin,
                    settings.ranking_mode,
                    settings.ranking_temperature,
                )
                support_a, valid_a = patch_support_targets(
                    labels_a, visible_a, candidates_a, source_config.image_size, source_config.patch_rows
                )
                support_b, valid_b = patch_support_targets(
                    labels_b, visible_b, candidates_b, source_config.image_size, source_config.patch_rows
                )
                coverage = 0.5 * (
                    dense_semantic_coverage_loss(
                        final_a, support_a, valid_a, source_config.image_size,
                        source_config.patch_rows, settings.coverage_topk, settings.coverage_temperature,
                    )
                    + dense_semantic_coverage_loss(
                        final_b, support_b, valid_b, source_config.image_size,
                        source_config.patch_rows, settings.coverage_topk, settings.coverage_temperature,
                    )
                )
                local = 0.5 * (
                    dense_local_dad_preservation_loss(
                        final_a, base_logits_a, support_a, valid_a, source_config.image_size,
                        source_config.patch_rows, settings.coverage_temperature,
                    )
                    + dense_local_dad_preservation_loss(
                        final_b, base_logits_b, support_b, valid_b, source_config.image_size,
                        source_config.patch_rows, settings.coverage_temperature,
                    )
                )
                invisibility = 0.5 * (
                    dense_geometric_invisibility_loss(
                        final_a,
                        fully_invisible_patch_mask(
                            candidate_tensor_a, visible_a, source_config.image_size, source_config.patch_rows,
                        ),
                        source_config.image_size,
                        source_config.patch_rows, settings.coverage_topk, settings.coverage_temperature,
                    )
                    + dense_geometric_invisibility_loss(
                        final_b,
                        fully_invisible_patch_mask(
                            candidate_tensor_b, visible_b, source_config.image_size, source_config.patch_rows,
                        ),
                        source_config.image_size,
                        source_config.patch_rows, settings.coverage_topk, settings.coverage_temperature,
                    )
                )
                total = (
                    geo
                    + settings.coverage_weight * coverage
                    + settings.local_preservation_weight * local
                    + settings.invisibility_weight * invisibility
                )
                suppress = total.detach() * 0.0
            elif settings.fusion_mode == "candidate_suppression":
                dad_features_a = _feature_tensor(detection_a.feature_map, selected_device)
                dad_features_b = _feature_tensor(detection_b.feature_map, selected_device)
                final_a, penalties_a = fusion(
                    dad_features_a, base_logits_a, semantic_a, candidate_tensor_a, source_config.image_size
                )
                final_b, penalties_b = fusion(
                    dad_features_b, base_logits_b, semantic_b, candidate_tensor_b, source_config.image_size
                )
                total, point, suppress, rank = candidate_suppression_loss(
                    final_a,
                    penalties_a,
                    candidate_supervision_labels(labels_a, visible_a).to(selected_device),
                    final_b,
                    penalties_b,
                    candidate_supervision_labels(labels_b, visible_b).to(selected_device),
                    settings.suppression_weight,
                    settings.suppression_margin,
                    settings.ranking_weight,
                    settings.ranking_temperature,
                )
                coverage = total.detach() * 0.0
                local = total.detach() * 0.0
                invisibility = total.detach() * 0.0
            else:
                dad_features_a = _feature_tensor(detection_a.feature_map, selected_device)
                dad_features_b = _feature_tensor(detection_b.feature_map, selected_device)
                final_a = fusion(dad_features_a, base_logits_a, semantic_a)
                final_b = fusion(dad_features_b, base_logits_b, semantic_b)
                logits_a = point_candidate_logits(final_a, candidate_tensor_a, source_config.image_size)
                logits_b = point_candidate_logits(final_b, candidate_tensor_b, source_config.image_size)
                total, point, rank = point_fusion_loss(
                    logits_a,
                    torch.from_numpy(labels_a).to(selected_device),
                    logits_b,
                    torch.from_numpy(labels_b).to(selected_device),
                    settings.ranking_weight,
                    settings.ranking_margin,
                    settings.ranking_mode,
                    settings.ranking_temperature,
                )
                suppress = total.detach() * 0.0
                coverage = total.detach() * 0.0
                local = total.detach() * 0.0
                invisibility = total.detach() * 0.0
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            total_sum += float(total.detach())
            point_sum += float(point.detach())
            suppress_sum += float(suppress.detach())
            rank_sum += float(rank.detach())
            coverage_sum += float(coverage.detach())
            local_sum += float(local.detach())
            invisibility_sum += float(invisibility.detach())
            progress.set_description(
                f"第 {epoch + 1}/{epochs} 轮 | {pair_index}/{len(pairs)} | "
                f"loss={total.detach():.4f} | avg={total_sum / pair_index:.4f} | "
                f"cover={coverage.detach():.4f} | local={local.detach():.4f} | "
                f"invisible={invisibility.detach():.4f}"
            )
        record = {
            "epoch": epoch + 1,
            "pairs": len(pairs),
            "total_loss": total_sum / len(pairs),
            "point_loss": point_sum / len(pairs),
            "suppression_loss": suppress_sum / len(pairs),
            "ranking_loss": rank_sum / len(pairs),
            "coverage_loss": coverage_sum / len(pairs),
            "local_preservation_loss": local_sum / len(pairs),
            "invisibility_loss": invisibility_sum / len(pairs),
        }
        history.append(record)
        is_best = float(record["total_loss"]) < best_loss
        best_loss = min(best_loss, float(record["total_loss"]))
        _save_checkpoint(output.with_name(f"{output.stem}.last{output.suffix}"), fusion, optimizer, source_config, semantic_checkpoint, settings, history, best_loss)
        if is_best:
            _save_checkpoint(output, fusion, optimizer, source_config, semantic_checkpoint, settings, history, best_loss)
    output.with_suffix(".history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _detections_from_cache(values: dict[str, np.ndarray]) -> DaDDetections:
    """从预计算数组恢复训练所需的 DaD 检测结果。"""
    logits = np.asarray(values["dense_logits"], dtype=np.float32)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("缓存 dense_logits 形状无效")
    shifted = logits - float(logits.max())
    scores = np.exp(shifted)
    scores /= float(scores.sum())
    return DaDDetections(
        dense_scores=scores.astype(np.float32),
        nms_points=np.asarray(values["nms_points"], dtype=np.float32),
        nms_scores=np.asarray(values["nms_scores"], dtype=np.float32),
        dense_logits=logits,
        feature_map=np.asarray(values["feature_map"], dtype=np.float32),
    )


def _save_checkpoint(
    path: Path,
    fusion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_config: object,
    semantic_checkpoint: str | Path,
    settings: PointFusionTrainConfig,
    history: list[dict[str, float | int]],
    best_loss: float,
) -> None:
    torch.save(
        {
            "format": "pasm_v05_dense_point_fusion",
            "fusion": fusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": len(history),
            "history": history,
            "best_loss": best_loss,
            "source_config": asdict(source_config),
            "semantic_checkpoint": str(Path(semantic_checkpoint)),
            "settings": asdict(settings),
            "fusion_mode": settings.fusion_mode,
            "ranking_mode": settings.ranking_mode,
            "ranking_temperature": settings.ranking_temperature,
        },
        path,
    )


def _load_fusion_checkpoint(
    path: str | Path,
    fusion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    settings: PointFusionTrainConfig,
) -> tuple[int, list[dict[str, float | int]], float]:
    """恢复 v0.5 融合头、优化器、历史和最佳损失。"""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"续训检查点不存在: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    if payload.get("format") != "pasm_v05_dense_point_fusion":
        raise ValueError("续训检查点不是 PASM v0.5 融合模型")
    if payload.get("fusion_mode") != settings.fusion_mode:
        raise ValueError("续训检查点的融合模式与当前配置不一致")
    saved_settings = payload.get("settings", {})
    for key in ("coverage_topk", "coverage_temperature", "invisibility_weight"):
        if key in saved_settings and saved_settings[key] != getattr(settings, key):
            raise ValueError(f"续训检查点的 {key} 与当前配置不一致")
    fusion.load_state_dict(payload["fusion"])
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    history = list(payload.get("history", []))
    epoch = int(payload.get("epoch", len(history)))
    if epoch != len(history):
        raise ValueError("续训检查点的 epoch 与 history 长度不一致")
    best_loss = float(payload.get("best_loss", min((float(x["total_loss"]) for x in history), default=float("inf"))))
    return epoch, history, best_loss


def _feature_tensor(feature_map: np.ndarray | None, device: torch.device) -> torch.Tensor:
    if feature_map is None:
        raise ValueError("DaD 检测结果缺少 32 通道特征图")
    array = np.asarray(feature_map, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 32 or not np.isfinite(array).all():
        raise ValueError("DaD 特征图必须是有限的 [H, W, 32] 数组")
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _logit_tensor(logits: np.ndarray | None, device: torch.device) -> torch.Tensor:
    if logits is None:
        raise ValueError("DaD 检测结果缺少稠密 logits")
    array = np.asarray(logits, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("DaD logits 必须是有限的 [H, W] 数组")
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def make_trainable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """在 inference_mode 边界外复制冻结骨干输出，供融合头反向传播使用。"""
    return tensor.detach().clone()


def candidate_supervision_labels(
    labels: np.ndarray | torch.Tensor,
    visible: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """将不可可靠投影的候选标为 ignore，其余保留正/负几何教师标签。"""
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    visible_tensor = torch.as_tensor(visible, dtype=torch.bool)
    if label_tensor.ndim != 1 or visible_tensor.shape != label_tensor.shape:
        raise ValueError("候选教师标签和可见性掩码必须是同形一维数组")
    if not torch.isfinite(label_tensor).all() or torch.any(label_tensor < 0.0) or torch.any(label_tensor > 1.0):
        raise ValueError("候选教师标签必须位于 [0, 1]")
    result = label_tensor.clone()
    result[~visible_tensor] = -1.0
    return result


def _image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正数")
    return int(height), int(width)


def _candidate_logit_vector(
    values: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    result = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
    if result.numel() == 0 or not torch.isfinite(result).all():
        raise ValueError("候选分数必须是非空有限一维张量")
    return result


def _candidate_coordinates(
    candidates_xy: torch.Tensor,
    logits: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    candidates = torch.as_tensor(candidates_xy, device=logits.device, dtype=logits.dtype)
    if candidates.ndim != 2 or candidates.shape != (logits.numel(), 2):
        raise ValueError("候选坐标必须与候选分数具有对应的 [N, 2] 形状")
    if not torch.isfinite(candidates).all():
        raise ValueError("候选坐标必须有限")
    if (
        torch.any(candidates[:, 0] < 0.0)
        or torch.any(candidates[:, 0] >= width)
        or torch.any(candidates[:, 1] < 0.0)
        or torch.any(candidates[:, 1] >= height)
    ):
        raise ValueError("候选坐标超出图像范围")
    return candidates


def _candidate_patch_ids(
    candidates_xy: torch.Tensor,
    height: int,
    width: int,
    patch_grid: int,
) -> torch.Tensor:
    if patch_grid <= 0:
        raise ValueError("Patch 网格必须为正数")
    cols = torch.floor(candidates_xy[:, 0] * patch_grid / width).to(torch.long).clamp_(0, patch_grid - 1)
    rows = torch.floor(candidates_xy[:, 1] * patch_grid / height).to(torch.long).clamp_(0, patch_grid - 1)
    return rows * patch_grid + cols


def _patch_support_and_valid(
    support: torch.Tensor,
    valid: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_tensor = torch.as_tensor(support, dtype=reference.dtype, device=reference.device).reshape(-1)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=reference.device).reshape(-1)
    if support_tensor.shape != reference.shape or valid_tensor.shape != reference.shape:
        raise ValueError("Patch 支持和有效掩码必须与 Patch 网格同形")
    if not torch.isfinite(support_tensor).all() or torch.any(support_tensor < 0.0):
        raise ValueError("Patch 支持必须为有限非负值")
    return support_tensor, valid_tensor


def _ranking_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    positive = logits[labels >= 0.5]
    negative = logits[labels <= 0.1]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    return torch.relu(torch.as_tensor(margin, device=logits.device, dtype=logits.dtype) - positive.mean() + negative.mean())


def _soft_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """使用标签差值加权的连续正负 pairwise 排序，始终保留平滑梯度。"""
    positive_mask = labels >= 0.5
    negative_mask = labels <= 0.1
    positive_logits = logits[positive_mask]
    negative_logits = logits[negative_mask]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return logits.sum() * 0.0
    logit_gap = positive_logits[:, None] - negative_logits[None, :]
    label_gap = labels[positive_mask][:, None] - labels[negative_mask][None, :]
    weights = label_gap.detach()
    losses = F.softplus(-logit_gap / temperature)
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def _candidate_suppression_view_loss(
    final_logits: torch.Tensor,
    penalties: torch.Tensor,
    labels: torch.Tensor,
    suppression_margin: float,
    ranking_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        final_logits.ndim != 1
        or penalties.shape != final_logits.shape
        or labels.shape != final_logits.shape
        or not torch.isfinite(final_logits).all()
        or not torch.isfinite(penalties).all()
        or not torch.isfinite(labels).all()
    ):
        raise ValueError("候选最终分数、抑制量和标签必须是同形有限一维张量")
    if torch.any(penalties < 0.0) or torch.any(labels < -1.0) or torch.any(labels > 1.0):
        raise ValueError("候选抑制量必须非负，标签必须位于 [-1, 1]")
    valid = labels >= 0.0
    positive = valid & (labels >= 0.5)
    negative = valid & (labels <= 0.1)
    zero = final_logits.sum() * 0.0
    if positive.any():
        weights = labels[positive]
        keep = (penalties[positive] * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    else:
        keep = zero
    if negative.any():
        suppress = F.softplus(
            torch.as_tensor(suppression_margin, dtype=penalties.dtype, device=penalties.device) - penalties[negative]
        ).mean()
    else:
        suppress = zero
    if positive.any() and negative.any():
        positive_logits = final_logits[positive]
        negative_logits = final_logits[negative]
        rank = F.softplus(
            -(positive_logits[:, None] - negative_logits[None, :]) / ranking_temperature
        ).mean()
    else:
        rank = zero
    return keep, suppress, rank


def _validate_loss_inputs(
    logits_a: torch.Tensor, labels_a: torch.Tensor, logits_b: torch.Tensor, labels_b: torch.Tensor,
    ranking_weight: float, ranking_margin: float, ranking_mode: str, ranking_temperature: float,
) -> None:
    if ranking_weight < 0.0 or ranking_margin < 0.0:
        raise ValueError("排序损失权重和间隔不能为负数")
    if ranking_mode not in {"hinge", "soft_pairwise"} or ranking_temperature <= 0.0:
        raise ValueError("排序模式或排序温度无效")
    for logits, labels in ((logits_a, labels_a), (logits_b, labels_b)):
        if logits.ndim != 1 or labels.shape != logits.shape or not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
            raise ValueError("候选 logits 和标签必须是同形有限一维张量")
        if torch.any(labels < 0.0) or torch.any(labels > 1.0):
            raise ValueError("软标签必须位于 [0, 1]")


def train_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    """将 CLI 文件字段转换为训练函数使用的参数名。"""
    options = vars(args).copy()
    options["manifest_path"] = options.pop("manifest")
    options["output_path"] = options.pop("output")
    options["resume_path"] = options.pop("resume", None)
    options["feature_cache_path"] = options.pop("feature_cache", None)
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 PASM v0.5 残差稠密点融合头")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--dad-weights", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--patch-grid", type=int, choices=(16, 37), default=37)
    parser.add_argument("--teacher-candidate-count", type=int, default=4096)
    parser.add_argument("--teacher-sigma-px", type=float, default=3.0)
    parser.add_argument("--teacher-search-radius-px", type=float, default=12.0)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--fusion-mode", choices=("residual", "contrastive", "candidate_suppression", "semantic_coverage", "patch_gate"), default="residual")
    parser.add_argument("--ranking-mode", choices=("hinge", "soft_pairwise"), default="hinge")
    parser.add_argument("--ranking-temperature", type=float, default=1.0)
    parser.add_argument("--suppression-weight", type=float, default=1.0)
    parser.add_argument("--suppression-margin", type=float, default=1.0)
    parser.add_argument("--border-margin-px", type=int, default=8)
    parser.add_argument("--gate-penalty-scale", type=float, default=3.0)
    parser.add_argument("--gate-quality-gamma", type=float, default=2.0)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--local-preservation-weight", type=float, default=0.25)
    parser.add_argument("--coverage-temperature", type=float, default=0.5)
    parser.add_argument("--coverage-topk", type=int, default=1024)
    parser.add_argument("--invisibility-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--feature-cache", type=Path, help="可选冻结 DINO/DaD 特征缓存目录")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(train_model(**train_options_from_args(args)))


if __name__ == "__main__":
    main()
