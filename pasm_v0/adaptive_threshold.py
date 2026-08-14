"""冻结融合响应图上的每图自适应停止阈值。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .point_fusion import point_candidate_logits


def threshold_supervision_labels(
    labels: torch.Tensor,
    visible: torch.Tensor,
    confirmed_invisible: torch.Tensor,
) -> torch.Tensor:
    """保留可见软标签，把明确不可见候选设为负样本，其余忽略。"""
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    visible_tensor = torch.as_tensor(visible, dtype=torch.bool)
    invisible_tensor = torch.as_tensor(confirmed_invisible, dtype=torch.bool)
    if (
        label_tensor.ndim != 1
        or visible_tensor.shape != label_tensor.shape
        or invisible_tensor.shape != label_tensor.shape
        or not torch.isfinite(label_tensor).all()
        or torch.any(label_tensor < 0.0)
        or torch.any(label_tensor > 1.0)
    ):
        raise ValueError("阈值教师标签、可见性和明确不可见掩码必须同形且标签位于 [0,1]")
    result = torch.full_like(label_tensor, -1.0)
    result[visible_tensor] = label_tensor[visible_tensor]
    result[~visible_tensor & invisible_tensor] = 0.0
    return result


class AdaptiveScoreThreshold(nn.Module):
    """从单图的双图条件语义 token 预测标准化响应阈值。"""

    def __init__(self, semantic_dim: int = 128) -> None:
        super().__init__()
        if semantic_dim <= 0:
            raise ValueError("语义维度必须为正数")
        self.output = nn.Linear(semantic_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        if semantic_tokens.ndim != 3 or semantic_tokens.shape[2] != self.output.in_features:
            raise ValueError("语义 token 必须是 [B, N, semantic_dim]")
        if not torch.isfinite(semantic_tokens).all():
            raise ValueError("语义 token 必须有限")
        return self.output(semantic_tokens.mean(dim=1)).squeeze(1)


def candidate_score_zscores(
    dense_logits: torch.Tensor,
    candidates_xy: torch.Tensor,
    image_size: int | tuple[int, int],
) -> torch.Tensor:
    """读取候选处的稠密融合响应，并在当前图内标准化。"""
    logits = point_candidate_logits(dense_logits, candidates_xy, image_size)
    if logits.numel() < 2:
        raise ValueError("自适应停止至少需要两个候选")
    return (logits - logits.mean()) / logits.std(unbiased=False).clamp_min(1e-6)


def adaptive_threshold_loss(
    dense_logits: torch.Tensor,
    semantic_tokens: torch.Tensor,
    candidates_xy: torch.Tensor,
    labels: torch.Tensor,
    threshold_head: nn.Module,
    image_size: int | tuple[int, int],
    temperature: float,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """学习使高质量候选越过每图独立停止阈值的软二分类损失。"""
    if temperature <= 0.0:
        raise ValueError("阈值温度必须为正数")
    if max_candidates <= 1:
        raise ValueError("阈值校准候选数必须大于 1")
    raw_logits = point_candidate_logits(dense_logits, candidates_xy, image_size)
    target = torch.as_tensor(labels, dtype=raw_logits.dtype, device=raw_logits.device).reshape(-1)
    if (
        target.shape != raw_logits.shape
        or not torch.isfinite(target).all()
        or torch.any(target < -1.0)
        or torch.any(target > 1.0)
    ):
        raise ValueError("阈值教师标签必须与候选同形且位于 [-1,1]")
    selected = torch.topk(raw_logits.detach(), min(max_candidates, raw_logits.numel())).indices
    selected_logits = raw_logits[selected]
    zscores = (selected_logits - selected_logits.mean()) / selected_logits.std(unbiased=False).clamp_min(1e-6)
    target = target[selected]
    threshold = threshold_head(semantic_tokens)
    if threshold.shape != (1,) or not torch.isfinite(threshold).all():
        raise ValueError("阈值头必须为单张图输出一个有限阈值")
    acceptance_logits = (zscores - threshold[0]) / temperature
    valid = target >= 0.0
    if not torch.any(valid):
        return acceptance_logits.sum() * 0.0, threshold
    return F.binary_cross_entropy_with_logits(acceptance_logits[valid], target[valid]), threshold


@torch.inference_mode()
def select_adaptive_dense_topk(
    dense_logits: torch.Tensor,
    semantic_tokens: torch.Tensor,
    threshold_head: nn.Module,
    max_points: int,
    point_selector: callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从原始稠密 NMS 峰中按每图阈值截断，不补足点数。"""
    points, scores = point_selector(dense_logits, max_points)
    if points.ndim != 2 or points.shape[1] != 2 or scores.shape != (len(points),):
        raise ValueError("稠密选点器返回形状无效")
    height, width = dense_logits.shape[-2:]
    pixels = points.clone()
    pixels[:, 0] = (pixels[:, 0] + 1.0) * width / 2.0
    pixels[:, 1] = (pixels[:, 1] + 1.0) * height / 2.0
    zscores = candidate_score_zscores(dense_logits, pixels, (height, width))
    threshold = threshold_head(semantic_tokens)
    if threshold.shape != (1,):
        raise ValueError("阈值头必须为单张图输出一个阈值")
    keep = zscores >= threshold[0]
    return points[keep], scores[keep], threshold[0]
