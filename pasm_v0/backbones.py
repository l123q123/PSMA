# PASM v0 的冻结 DINOv2 和 LoMa 特征提取接口。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .config import PasmV0Config


@dataclass(frozen=True)
class ExtractedFeatures:
    """保存一对图像的管理 Patch 语义 token 和 LoMa 描述子图。"""

    tokens_a: torch.Tensor
    tokens_b: torch.Tensor
    descriptor_grid_a: np.ndarray
    descriptor_grid_b: np.ndarray


class DINOv2PatchEncoder(nn.Module):
    """加载冻结的 DINOv2 ViT-L/14 并输出图像 Patch token。"""

    def __init__(self, weights_path: str | Path, image_size: int = 518) -> None:
        """按输入尺寸构建骨干并读取本地 DINOv2 权重。"""
        super().__init__()
        weight_path = Path(weights_path)
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        if image_size % 14:
            raise ValueError("DINOv2 输入尺寸必须能被 14 整除")
        from loma.descriptor.transformer import vit_large

        self.image_size = image_size
        self.backbone = vit_large(
            img_size=image_size, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0
        )
        self.backbone.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False), strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv2PatchEncoder":
        """始终保持冻结骨干处于推理模式。"""
        super().train(False)
        self.backbone.eval()
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """返回与输入图像对应的归一化 DINO Patch token。"""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("图像必须具有 [B, 3, H, W] 形状")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError("图像尺寸与 DINOv2 配置不一致")
        return self.backbone.forward_features(images)["x_norm_patchtokens"]


class FrozenLoMaDescriptor:
    """加载冻结 LoMa-B128 并输出稠密局部描述子图。"""

    def __init__(self, weights_path: str | Path, device: str = "auto") -> None:
        """读取本地 LoMa 描述子权重并固定模型参数。"""
        from loma.descriptor.dedode import DeDoDeDescriptor
        from loma.device import device as loma_device

        weight_path = Path(weights_path)
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        selected_device = loma_device if device == "auto" else torch.device(device)
        self.model = DeDoDeDescriptor(
            DeDoDeDescriptor.Cfg(arch="dedode_b", compile=False, descriptor_dim=128)
        ).to(selected_device).eval()
        state = torch.load(weight_path, map_location=selected_device, weights_only=False)
        self.model.load_state_dict(
            {key.removeprefix("_descriptor."): value for key, value in state.items() if key.startswith("_descriptor.")},
            strict=True,
        )
        self.device = selected_device

    @torch.inference_mode()
    def describe_image(self, image: np.ndarray) -> np.ndarray:
        """计算单张 CHW RGB 图像的稠密 LoMa 描述子。"""
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None].to(self.device)
        return self.model(tensor)[0].float().cpu().numpy().astype(np.float32)


class FrozenFeatureExtractor:
    """将冻结骨干输出转换为 PASM v0 所需的两类特征。"""

    def __init__(
        self,
        config: PasmV0Config,
        dino_weights: str | Path,
        loma_weights: str | Path,
        device: str = "auto",
    ) -> None:
        """加载与 v0 图像尺寸一致的冻结 DINOv2 和 LoMa 权重。"""
        self.config = config
        self.device = resolve_device(device)
        self.dino = DINOv2PatchEncoder(dino_weights, image_size=config.image_size).to(self.device).eval()
        self.loma = FrozenLoMaDescriptor(loma_weights, device=str(self.device))

    def extract(self, image_a: np.ndarray, image_b: np.ndarray) -> ExtractedFeatures:
        """提取两张已缩放 RGB 图像的管理 Patch token 与局部描述子图。"""
        first = _image_tensor(image_a, self.config, self.device)
        second = _image_tensor(image_b, self.config, self.device)
        tokens = clone_frozen_tokens(self.dino(torch.stack((first, second), dim=0)))
        pooled = pool_dino_tokens(tokens, self.config.patch_rows, self.config.patch_cols)
        return ExtractedFeatures(
            tokens_a=pooled[:1],
            tokens_b=pooled[1:],
            descriptor_grid_a=self.loma.describe_image(np.asarray(image_a, dtype=np.float32)),
            descriptor_grid_b=self.loma.describe_image(np.asarray(image_b, dtype=np.float32)),
        )


def load_resized_rgb(path: str | Path, image_size: int) -> np.ndarray:
    """读取图像并转换为指定大小的归一化 CHW RGB 数组。"""
    from PIL import Image

    image_path = Path(path)
    if not image_path.is_file() or image_size <= 0:
        raise ValueError("图像路径不存在或尺寸无效")
    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)


def pool_dino_tokens(tokens: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """将规则 DINO token 网格平均池化为管理 Patch 网格。"""
    if tokens.ndim != 3 or rows <= 0 or cols <= 0:
        raise ValueError("DINO token 或管理 Patch 网格无效")
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise ValueError("DINO token 数必须构成正方形网格")
    grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
    pooled = functional.adaptive_avg_pool2d(grid, (rows, cols))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def clone_frozen_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """将冻结推理模式产生的 token 转为可安全反传的普通张量。"""
    return tokens.detach().clone()


def resolve_device(value: str) -> torch.device:
    """将自动或显式设备名称转换为 PyTorch 设备。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _image_tensor(image: np.ndarray, config: PasmV0Config, device: torch.device) -> torch.Tensor:
    """验证并将单张 CHW 图像转换为指定设备上的张量。"""
    array = np.asarray(image, dtype=np.float32)
    expected = (3, config.image_size, config.image_size)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError("图像必须是与配置一致的有限 CHW RGB 数组")
    return torch.from_numpy(array).to(device)
