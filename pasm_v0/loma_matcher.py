# PASM v0.3 的冻结完整 LoMa 指定点描述与 Transformer 匹配接口。

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class FrozenLoMaMatcher:
    """从本地权重加载完整冻结 LoMa-B128，并匹配外部给定的点集。"""

    def __init__(
        self,
        weights_path: str | Path,
        dad_weights_path: str | Path,
        dino_weights_path: str | Path,
        device: str | torch.device,
    ) -> None:
        """构建禁止网络下载的完整 LoMa-B128 匹配器。"""
        self.weights_path = Path(weights_path)
        self.dad_weights_path = Path(dad_weights_path)
        self.dino_weights_path = Path(dino_weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)
        if not self.dad_weights_path.is_file():
            raise FileNotFoundError(self.dad_weights_path)
        if not self.dino_weights_path.is_file():
            raise FileNotFoundError(self.dino_weights_path)
        self.device = torch.device(device)
        from loma import loma as loma_module
        from loma.descriptor import dedode as dedode_module
        from loma.detector import dad as dad_module

        loma_module.device = self.device
        dedode_module.device = self.device
        dad_module.device = self.device
        loma_state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        dad_state = torch.load(self.dad_weights_path, map_location="cpu", weights_only=False)
        dino_state = torch.load(self.dino_weights_path, map_location="cpu", weights_only=False)

        def load_local_weights(url: str, *args, **kwargs):
            """按 LoMa 构造过程中的下载地址返回对应的本地状态字典。"""
            location = str(url)
            if location.endswith("dad.pth"):
                return dad_state
            if "dinov2_vitl14" in location:
                return dino_state
            return loma_state

        with patch("torch.hub.load_state_dict_from_url", side_effect=load_local_weights):
            self.model = loma_module.LoMa(loma_b128_config())
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def match(
        self,
        image_a: np.ndarray | torch.Tensor,
        points_a: np.ndarray | torch.Tensor,
        image_b: np.ndarray | torch.Tensor,
        points_b: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """描述两图的指定点并返回 LoMa 互为最强匹配索引及连续置信度。"""
        tensor_a = _image_tensor(image_a, self.device)
        tensor_b = _image_tensor(image_b, self.device)
        if tensor_a.shape[-2:] != tensor_b.shape[-2:]:
            raise ValueError("LoMa 两张图的尺寸必须一致")
        height, width = tensor_a.shape[-2:]
        if height != width:
            raise ValueError("PASM v0.3 当前要求正方形输入图像")
        keypoints_a = pixel_keypoints_to_normalized(points_a, width).to(self.device)[None]
        keypoints_b = pixel_keypoints_to_normalized(points_b, width).to(self.device)[None]
        if keypoints_a.shape[1] == 0 or keypoints_b.shape[1] == 0:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty.copy(), np.empty((0,), dtype=np.float32)
        descriptions_a = self.model._descriptor.describe_keypoints(tensor_a, keypoints_a)["descriptions"]
        descriptions_b = self.model._descriptor.describe_keypoints(tensor_b, keypoints_b)["descriptions"]
        scores = self.model(keypoints_a, keypoints_b, descriptions_a, descriptions_b)["scores"]
        from loma.loma import filter_matches

        matches_a, _, confidence_a, _ = filter_matches(scores, float(self.model.cfg.filter_threshold))
        valid = matches_a[0] >= 0
        indices_a = torch.where(valid)[0]
        indices_b = matches_a[0, valid]
        return (
            indices_a.cpu().numpy().astype(np.int64),
            indices_b.cpu().numpy().astype(np.int64),
            confidence_a[0, valid].float().cpu().numpy().astype(np.float32),
        )


def loma_b128_config():
    """返回与本地 loma_B128.pth 权重严格对应的 LoMa 配置。"""
    from loma.loma import LoMa

    return LoMa.Cfg(
        input_dim=128,
        embed_dim=256,
        n_layers=9,
        num_heads=4,
        descriptor="dedode_b",
        compile=False,
        weights_url="local://loma_B128",
    )


def pixel_keypoints_to_normalized(
    points_xy: np.ndarray | torch.Tensor, image_size: int
) -> torch.Tensor:
    """将 PASM 像素坐标转换为 LoMa 使用的 [-1, 1] 归一化坐标。"""
    points = _point_tensor(points_xy)
    if image_size <= 0 or torch.any(points < 0.0) or torch.any(points >= image_size):
        raise ValueError("像素关键点必须位于图像范围内")
    return points * (2.0 / image_size) - 1.0


def normalized_keypoints_to_pixels(points_xy: np.ndarray | torch.Tensor, image_size: int) -> torch.Tensor:
    """将 LoMa 的 [-1, 1] 归一化关键点转换为 PASM 像素坐标。"""
    if image_size <= 0:
        raise ValueError("图像尺寸必须为正数")
    return (_point_tensor(points_xy) + 1.0) * (image_size / 2.0)


def _image_tensor(image: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """验证并将一张 CHW RGB 图像转换为指定设备上的单样本张量。"""
    tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor[None]
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3 or not torch.isfinite(tensor).all():
        raise ValueError("LoMa 输入图像必须具有 [3, H, W] 或 [1, 3, H, W] 形状")
    return tensor


def _point_tensor(points_xy: np.ndarray | torch.Tensor) -> torch.Tensor:
    """验证并转换一组有限的二维关键点坐标。"""
    points = torch.as_tensor(points_xy, dtype=torch.float32)
    if points.ndim != 2 or points.shape[1:] != (2,) or not torch.isfinite(points).all():
        raise ValueError("关键点必须是形状为 [N, 2] 的有限数组")
    return points
