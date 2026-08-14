"""PASM v0.5 的无几何稠密融合选点推理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .backbones import DINOv2PatchEncoder, pool_dino_tokens
from .adaptive_threshold import AdaptiveScoreThreshold, select_adaptive_dense_topk
from .dad import DaDDetections, FrozenDaDDetector, select_dad_candidate_pool
from .infer import load_runner
from .point_fusion import create_point_fusion, select_candidate_topk, select_dense_topk


PointSelector = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class V05Runner:
    """融合冻结 DaD 与双图语义信息、但不接收任何几何输入的选点器。"""

    config: object
    dino: torch.nn.Module
    semantic_encoder: torch.nn.Module
    detector: FrozenDaDDetector
    fusion: torch.nn.Module
    device: torch.device
    point_selector: PointSelector = select_dense_topk
    fusion_mode: str = "residual"
    candidate_pool_size: int = 4096
    border_margin_px: int = 8
    adaptive_threshold: torch.nn.Module | None = None

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
        threshold_a = threshold_b = None
        if self.fusion_mode in {"candidate_suppression", "patch_gate"}:
            points_a, scores_a, score_map_a = _select_suppressed_candidates(
                self.fusion, detection_a, semantic_a, self.config.image_size, int(self.config.total_points),
                self.candidate_pool_size, self.border_margin_px, self.device, self.fusion_mode,
            )
            points_b, scores_b, score_map_b = _select_suppressed_candidates(
                self.fusion, detection_b, semantic_b, self.config.image_size, int(self.config.total_points),
                self.candidate_pool_size, self.border_margin_px, self.device, self.fusion_mode,
            )
        else:
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
            if self.adaptive_threshold is None:
                normalized_a, scores_a = self.point_selector(final_a, int(self.config.total_points))
                normalized_b, scores_b = self.point_selector(final_b, int(self.config.total_points))
                threshold_a = threshold_b = None
            else:
                normalized_a, scores_a, threshold_a = select_adaptive_dense_topk(
                    final_a, semantic_a, self.adaptive_threshold, int(self.config.total_points), self.point_selector,
                )
                normalized_b, scores_b, threshold_b = select_adaptive_dense_topk(
                    final_b, semantic_b, self.adaptive_threshold, int(self.config.total_points), self.point_selector,
                )
            points_a = _normalized_to_pixels(normalized_a, self.config.image_size)
            points_b = _normalized_to_pixels(normalized_b, self.config.image_size)
            scores_a = _score_array(scores_a, expected_count=None if self.adaptive_threshold is not None else int(self.config.total_points))
            scores_b = _score_array(scores_b, expected_count=None if self.adaptive_threshold is not None else int(self.config.total_points))
            score_map_a = _score_map(final_a)
            score_map_b = _score_map(final_b)
        diagnostics: dict[str, object] = {
            "used_geometry": False,
            "fused_score_map_a": score_map_a,
            "fused_score_map_b": score_map_b,
            "base_score_map_a": np.asarray(detection_a.dense_scores, dtype=np.float32),
            "base_score_map_b": np.asarray(detection_b.dense_scores, dtype=np.float32),
            "selected_count_a": int(len(points_a)),
            "selected_count_b": int(len(points_b)),
            "max_points": int(self.config.total_points),
            "adaptive_threshold_a": None if threshold_a is None else float(threshold_a.detach()),
            "adaptive_threshold_b": None if threshold_b is None else float(threshold_b.detach()),
        }
        return points_a, points_b, scores_a, scores_b, diagnostics


def load_v05_runner(
    checkpoint: str | Path,
    dino_weights: str | Path,
    dad_weights: str | Path,
    semantic_checkpoint: str | Path,
    device: str | torch.device,
    adaptive_threshold_checkpoint: str | Path | None = None,
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
    fusion_mode = str(payload.get("fusion_mode", "residual"))
    settings = dict(payload.get("settings", {}))
    fusion_kwargs = (
        {
            "penalty_scale": float(settings.get("gate_penalty_scale", 3.0)),
            "quality_gamma": float(settings.get("gate_quality_gamma", 2.0)),
        }
        if fusion_mode == "patch_gate" else {}
    )
    fusion = create_point_fusion(fusion_mode, **fusion_kwargs).to(selected_device).eval()
    fusion.load_state_dict(payload["fusion"], strict=True)
    dino = DINOv2PatchEncoder(dino_weights, source_config.image_size).to(selected_device).eval()
    detector = FrozenDaDDetector(dad_weights, source_config.image_size, selected_device)
    source_runner.semantic_encoder.eval()
    adaptive_threshold = _load_adaptive_threshold(
        adaptive_threshold_checkpoint, source_config, selected_device,
    )
    return V05Runner(
        config=source_config,
        dino=dino,
        semantic_encoder=source_runner.semantic_encoder,
        detector=detector,
        fusion=fusion,
        device=selected_device,
        fusion_mode=fusion_mode,
        candidate_pool_size=int(settings.get("teacher_candidate_count", 4096)),
        border_margin_px=int(settings.get("border_margin_px", 8)),
        adaptive_threshold=adaptive_threshold,
    )


def _select_suppressed_candidates(
    fusion: torch.nn.Module,
    detection: DaDDetections,
    semantic_tokens: torch.Tensor,
    image_size: int,
    total_points: int,
    candidate_pool_size: int,
    border_margin_px: int,
    device: torch.device,
    fusion_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates, _ = select_dad_candidate_pool(detection, candidate_pool_size)
    candidate_tensor = torch.from_numpy(candidates).to(device)
    base_logits = _logit_tensor(detection, device)
    if fusion_mode == "patch_gate":
        final_logits, _ = fusion(base_logits, semantic_tokens, candidate_tensor, image_size)
    else:
        final_logits, _ = fusion(
            _feature_tensor(detection, device), base_logits, semantic_tokens, candidate_tensor, image_size,
        )
    points, scores = select_candidate_topk(
        candidate_tensor, final_logits, total_points, image_size, border_margin_px
    )
    return (
        points.cpu().numpy().astype(np.float32),
        scores.cpu().numpy().astype(np.float32),
        _candidate_score_map(base_logits, candidate_tensor, final_logits),
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


def _score_array(scores: torch.Tensor, expected_count: int | None) -> np.ndarray:
    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if not torch.isfinite(values).all() or (expected_count is not None and len(values) != expected_count):
        raise ValueError("融合选点器未返回要求数量的有限分数")
    return values.cpu().numpy().astype(np.float32)


def _load_adaptive_threshold(
    checkpoint: str | Path | None, source_config: object, device: torch.device,
) -> AdaptiveScoreThreshold | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "pasm_v07_adaptive_threshold":
        raise ValueError("自适应阈值检查点格式无效")
    if dict(payload.get("source_config", {})) != asdict(source_config):
        raise ValueError("自适应阈值检查点与语义模型配置不一致")
    head = AdaptiveScoreThreshold().to(device).eval()
    head.load_state_dict(payload["threshold_head"], strict=True)
    return head


def _score_map(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError("融合 logits 必须是 [1, 1, H, W]")
    height, width = logits.shape[-2:]
    return logits.reshape(-1).softmax(dim=0).reshape(height, width).detach().cpu().numpy().astype(np.float32)


def _candidate_score_map(
    base_logits: torch.Tensor,
    candidates_xy: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> np.ndarray:
    """仅在固定 DaD 候选位置记录重打分结果，供评测热图诊断使用。"""
    if base_logits.ndim != 4 or base_logits.shape[:2] != (1, 1):
        raise ValueError("基础 DaD logits 必须是 [1, 1, H, W]")
    if candidates_xy.ndim != 2 or candidates_xy.shape[1] != 2 or candidate_logits.shape != (len(candidates_xy),):
        raise ValueError("候选热图输入形状无效")
    dense = torch.full_like(base_logits, float(candidate_logits.detach().min()) - 20.0)
    x = torch.floor(candidates_xy[:, 0]).to(torch.long)
    y = torch.floor(candidates_xy[:, 1]).to(torch.long)
    dense[0, 0, y, x] = candidate_logits
    return _score_map(dense)


def _validate_image(image: np.ndarray, image_size: int) -> None:
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (3, image_size, image_size) or not np.isfinite(array).all():
        raise ValueError("v0.5 推理输入必须是有限的固定尺寸 CHW RGB 图像")
