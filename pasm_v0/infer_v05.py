"""PASM v0.5 的无几何稠密融合选点推理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .backbones import DINOv2PatchEncoder, pool_dino_tokens
from .dad import DaDDetections, FrozenDaDDetector
from .infer import load_runner
from .point_fusion import DensePointFusion, select_dense_topk


PointSelector = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class V05Runner:
    """融合冻结 DaD 与双图语义信息、但不接收任何几何输入的选点器。"""

    config: object
    dino: torch.nn.Module
    semantic_encoder: torch.nn.Module
    detector: FrozenDaDDetector
    fusion: DensePointFusion
    device: torch.device
    point_selector: PointSelector = select_dense_topk

    @torch.inference_mode()
    def select(
        self, image_a: np.ndarray, image_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """只根据输入 RGB 图像选出两图各 1024 个稠密融合点。"""
        _validate_image(image_a, self.config.image_size)
        _validate_image(image_b, self.config.image_size)
        detection_a = self.detector.detect(image_a)
        detection_b = self.detector.detect(image_b)
        images = torch.from_numpy(
            np.ascontiguousarray(np.stack((image_a, image_b), axis=0), dtype=np.float32)
        ).to(self.device)
        tokens = pool_dino_tokens(
            self.dino(images), self.config.patch_rows, self.config.patch_cols
        )
        semantic_a, semantic_b, _, _ = self.semantic_encoder(tokens[:1], tokens[1:])
        final_a = self.fusion(
            _feature_tensor(detection_a, self.device),
            _logit_tensor(detection_a, self.device),
            semantic_a,
        )
        final_b = self.fusion(
            _feature_tensor(detection_b, self.device),
            _logit_tensor(detection_b, self.device),
            semantic_b,
        )
        normalized_a, scores_a = self.point_selector(final_a, int(self.config.total_points))
        normalized_b, scores_b = self.point_selector(final_b, int(self.config.total_points))
        points_a = _normalized_to_pixels(normalized_a, self.config.image_size)
        points_b = _normalized_to_pixels(normalized_b, self.config.image_size)
        scores_a = _score_array(scores_a, self.config.total_points)
        scores_b = _score_array(scores_b, self.config.total_points)
        diagnostics: dict[str, object] = {
            "used_geometry": False,
            "fused_score_map_a": _score_map(final_a),
            "fused_score_map_b": _score_map(final_b),
            "base_score_map_a": np.asarray(detection_a.dense_scores, dtype=np.float32),
            "base_score_map_b": np.asarray(detection_b.dense_scores, dtype=np.float32),
        }
        return points_a, points_b, scores_a, scores_b, diagnostics


def load_v05_runner(
    checkpoint: str | Path,
    dino_weights: str | Path,
    dad_weights: str | Path,
    semantic_checkpoint: str | Path,
    device: str | torch.device,
) -> V05Runner:
    """读取 v0.5 融合头和全部冻结推理部件。"""
    selected_device = torch.device(device)
    payload = torch.load(checkpoint, map_location=selected_device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "pasm_v05_dense_point_fusion":
        raise ValueError("检查点不是 PASM v0.5 稠密融合模型")
    if "fusion" not in payload:
        raise ValueError("v0.5 检查点缺少融合头参数")
    source_runner = load_runner(semantic_checkpoint, selected_device)
    source_config = source_runner.config
    if dict(payload.get("source_config", {})) != asdict(source_config):
        raise ValueError("v0.5 检查点与提供的语义检查点配置不一致")
    fusion = DensePointFusion().to(selected_device).eval()
    fusion.load_state_dict(payload["fusion"], strict=True)
    dino = DINOv2PatchEncoder(dino_weights, source_config.image_size).to(selected_device).eval()
    detector = FrozenDaDDetector(dad_weights, source_config.image_size, selected_device)
    source_runner.semantic_encoder.eval()
    return V05Runner(
        config=source_config,
        dino=dino,
        semantic_encoder=source_runner.semantic_encoder,
        detector=detector,
        fusion=fusion,
        device=selected_device,
    )


def _feature_tensor(detection: DaDDetections, device: torch.device) -> torch.Tensor:
    feature_map = detection.feature_map
    array = np.asarray(feature_map, dtype=np.float32) if feature_map is not None else None
    if array is None or array.ndim != 3 or array.shape[2] != 32 or not np.isfinite(array).all():
        raise ValueError("DaD 检测结果缺少有限的 32 通道特征图")
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _logit_tensor(detection: DaDDetections, device: torch.device) -> torch.Tensor:
    logits = detection.dense_logits
    array = np.asarray(logits, dtype=np.float32) if logits is not None else None
    if array is None or array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("DaD 检测结果缺少有限的稠密 logits")
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def _normalized_to_pixels(points: torch.Tensor, image_size: int) -> np.ndarray:
    values = torch.as_tensor(points, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1:] != (2,) or not torch.isfinite(values).all():
        raise ValueError("融合选点器必须返回有限的 [N, 2] 归一化坐标")
    upper = float(np.nextafter(np.float32(image_size), np.float32(0.0)))
    pixels = torch.clamp((values + 1.0) * (float(image_size) / 2.0), min=0.0, max=upper)
    return pixels.cpu().numpy().astype(np.float32)


def _score_array(scores: torch.Tensor, point_count: int) -> np.ndarray:
    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if point_count <= 0 or len(values) != point_count or not torch.isfinite(values).all():
        raise ValueError("融合选点器未返回要求数量的有限分数")
    return values.cpu().numpy().astype(np.float32)


def _score_map(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError("融合 logits 必须是 [1, 1, H, W]")
    height, width = logits.shape[-2:]
    return logits.reshape(-1).softmax(dim=0).reshape(height, width).cpu().numpy().astype(np.float32)


def _validate_image(image: np.ndarray, image_size: int) -> None:
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (3, image_size, image_size) or not np.isfinite(array).all():
        raise ValueError("v0.5 推理输入必须是有限的固定尺寸 CHW RGB 图像")
