"""残差式稠密点响应融合。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DensePointFusion(nn.Module):
    """用双图语义 token 对冻结 DaD 基础 logits 做共享残差修正。"""

    def __init__(
        self,
        semantic_dim: int = 128,
        dad_channels: int = 32,
        hidden_channels: int = 64,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        if semantic_dim <= 0 or dad_channels <= 0 or hidden_channels <= 0:
            raise ValueError("融合头通道数必须为正数")
        self.semantic_projection = nn.Linear(semantic_dim, dad_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(dad_channels * 2 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8 if hidden_channels % 8 == 0 else 1, num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(
        self,
        dad_features: torch.Tensor,
        base_logits: torch.Tensor,
        semantic_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """返回与基础 DaD logits 同分辨率的最终稠密 logits。"""
        _validate_inputs(dad_features, base_logits, semantic_tokens)
        height, width = base_logits.shape[-2:]
        dad = F.interpolate(dad_features, size=(height, width), mode="bilinear", align_corners=False)
        token_count = semantic_tokens.shape[1]
        grid = math.isqrt(token_count)
        if grid * grid != token_count:
            raise ValueError("语义 token 数量必须是正方形网格")
        semantic = self.semantic_projection(semantic_tokens)
        semantic = semantic.transpose(1, 2).reshape(
            semantic_tokens.shape[0], -1, grid, grid
        )
        semantic = F.interpolate(semantic, size=(height, width), mode="bilinear", align_corners=False)
        delta = self.fusion(torch.cat((dad, semantic, base_logits), dim=1))
        return base_logits + self.alpha * delta


def point_candidate_logits(
    dense_logits: torch.Tensor,
    points_xy: torch.Tensor,
    image_size: int | tuple[int, int],
) -> torch.Tensor:
    """双线性读取指定像素坐标处的稠密 logits。"""
    if dense_logits.ndim != 4 or dense_logits.shape[0] != 1 or dense_logits.shape[1] != 1:
        raise ValueError("稠密 logits 必须具有 [1, 1, H, W] 形状")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("候选点必须具有 [N, 2] 形状")
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正数")
    points = points_xy.to(device=dense_logits.device, dtype=dense_logits.dtype)
    grid = points.clone()
    grid[:, 0] = (grid[:, 0] + 0.5) * 2.0 / width - 1.0
    grid[:, 1] = (grid[:, 1] + 0.5) * 2.0 / height - 1.0
    sampled = F.grid_sample(
        dense_logits,
        grid.view(1, 1, -1, 2),
        mode="bilinear",
        align_corners=False,
    )
    return sampled[0, 0, 0]


@torch.inference_mode()
def select_dense_topk(
    dense_logits: torch.Tensor, num_points: int = 1024, nms_size: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 DaD 规则从融合稠密 logits 中选择归一化 Top-K 点。"""
    if dense_logits.ndim != 4 or dense_logits.shape[0] != 1 or dense_logits.shape[1] != 1:
        raise ValueError("稠密 logits 必须具有 [1, 1, H, W] 形状")
    height, width = dense_logits.shape[-2:]
    if num_points <= 0 or num_points > height * width or nms_size <= 0:
        raise ValueError("Top-K 点数或 NMS 尺寸无效")
    from loma.detector.dad import sample_keypoints

    dense_probs = dense_logits.reshape(1, height * width).softmax(dim=-1).reshape(1, height, width)
    points, scores = sample_keypoints(
        dense_probs,
        num_samples=num_points,
        use_nms=True,
        nms_size=nms_size,
        sample_topk=True,
        increase_coverage=False,
        return_probs=True,
        subpixel=True,
        scoremap=dense_logits[:, 0],
        subpixel_temp=0.5,
    )
    return points[0], scores[0]


def _validate_inputs(
    dad_features: torch.Tensor,
    base_logits: torch.Tensor,
    semantic_tokens: torch.Tensor,
) -> None:
    if dad_features.ndim != 4 or base_logits.ndim != 4 or semantic_tokens.ndim != 3:
        raise ValueError("融合输入维度必须分别为 4、4、3")
    if dad_features.shape[0] != base_logits.shape[0] or semantic_tokens.shape[0] != base_logits.shape[0]:
        raise ValueError("融合输入的批次维度必须一致")
    if base_logits.shape[1] != 1 or semantic_tokens.shape[2] != 128:
        raise ValueError("基础 logits 必须单通道且语义 token 必须为 128 维")
    if not torch.isfinite(dad_features).all() or not torch.isfinite(base_logits).all() or not torch.isfinite(semantic_tokens).all():
        raise ValueError("融合输入必须是有限张量")
