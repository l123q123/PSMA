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
from .dad import FrozenDaDDetector, select_dad_candidate_pool
from .data import load_manifest
from .geometry import load_pair_geometry
from .infer import load_runner
from .point_fusion import DensePointFusion, point_candidate_logits
from .point_teacher import build_candidate_teacher


@dataclass(frozen=True)
class PointFusionTrainConfig:
    """v0.5 融合训练的可验证超参数。"""

    teacher_candidate_count: int = 4096
    teacher_sigma_px: float = 3.0
    teacher_search_radius_px: float = 12.0
    ranking_weight: float = 0.5
    ranking_margin: float = 0.2

    def __post_init__(self) -> None:
        if self.teacher_candidate_count <= 0:
            raise ValueError("教师候选池大小必须为正数")
        if self.teacher_sigma_px <= 0.0 or self.teacher_search_radius_px <= 0.0:
            raise ValueError("教师 sigma 和搜索半径必须为正数")
        if self.ranking_weight < 0.0 or self.ranking_margin < 0.0:
            raise ValueError("排序损失权重和间隔不能为负数")


def point_fusion_loss(
    logits_a: torch.Tensor,
    labels_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels_b: torch.Tensor,
    ranking_weight: float,
    ranking_margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算双图候选点 soft BCE 与正负排序损失。"""
    _validate_loss_inputs(logits_a, labels_a, logits_b, labels_b, ranking_weight, ranking_margin)
    point = 0.5 * (
        F.binary_cross_entropy_with_logits(logits_a, labels_a)
        + F.binary_cross_entropy_with_logits(logits_b, labels_b)
    )
    rank = 0.5 * (
        _ranking_loss(logits_a, labels_a, ranking_margin)
        + _ranking_loss(logits_b, labels_b, ranking_margin)
    )
    return point + ranking_weight * rank, point, rank


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
    seed: int = 0,
) -> Path:
    """冻结 DaD/DINO/语义编码器，仅训练 v0.5 稠密融合头。"""
    if epochs <= 0 or learning_rate <= 0.0 or patch_grid not in {16, 37}:
        raise ValueError("训练轮数、学习率或 Patch 网格无效")
    settings = PointFusionTrainConfig(
        teacher_candidate_count=teacher_candidate_count,
        teacher_sigma_px=teacher_sigma_px,
        teacher_search_radius_px=teacher_search_radius_px,
        ranking_weight=ranking_weight,
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
    dino = DINOv2PatchEncoder(dino_weights, source_config.image_size).to(selected_device).eval()
    detector = FrozenDaDDetector(dad_weights, source_config.image_size, selected_device)
    fusion = DensePointFusion().to(selected_device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=learning_rate)
    pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("最多训练图像对数必须为正数")
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("训练清单中没有正图像对")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(epochs):
        fusion.train()
        total_sum = point_sum = rank_sum = 0.0
        progress = tqdm(pairs, total=len(pairs), unit="对", dynamic_ncols=True)
        for pair_index, pair in enumerate(progress, start=1):
            image_a = load_resized_rgb(pair.image_a, source_config.image_size)
            image_b = load_resized_rgb(pair.image_b, source_config.image_size)
            geometry = load_pair_geometry(pair, source_config.image_size)
            detection_a = detector.detect(image_a)
            detection_b = detector.detect(image_b)
            candidates_a, _ = select_dad_candidate_pool(detection_a, settings.teacher_candidate_count)
            candidates_b, _ = select_dad_candidate_pool(detection_b, settings.teacher_candidate_count)
            labels_a, _, _ = build_candidate_teacher(
                candidates_a, candidates_b, geometry, "a_to_b", settings.teacher_sigma_px,
                settings.teacher_search_radius_px, source_config.geometry_depth_tolerance,
            )
            labels_b, _, _ = build_candidate_teacher(
                candidates_b, candidates_a, geometry, "b_to_a", settings.teacher_sigma_px,
                settings.teacher_search_radius_px, source_config.geometry_depth_tolerance,
            )
            with torch.inference_mode():
                images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(selected_device)
                tokens = pool_dino_tokens(
                    dino(images), source_config.patch_rows, source_config.patch_cols
                )
                semantic_a, semantic_b, _, _ = semantic_encoder(tokens[:1], tokens[1:])
            semantic_a = make_trainable_tensor(semantic_a)
            semantic_b = make_trainable_tensor(semantic_b)
            final_a = fusion(
                _feature_tensor(detection_a.feature_map, selected_device),
                _logit_tensor(detection_a.dense_logits, selected_device),
                semantic_a,
            )
            final_b = fusion(
                _feature_tensor(detection_b.feature_map, selected_device),
                _logit_tensor(detection_b.dense_logits, selected_device),
                semantic_b,
            )
            logits_a = point_candidate_logits(
                final_a, torch.from_numpy(candidates_a).to(selected_device), source_config.image_size
            )
            logits_b = point_candidate_logits(
                final_b, torch.from_numpy(candidates_b).to(selected_device), source_config.image_size
            )
            total, point, rank = point_fusion_loss(
                logits_a,
                torch.from_numpy(labels_a).to(selected_device),
                logits_b,
                torch.from_numpy(labels_b).to(selected_device),
                settings.ranking_weight,
                settings.ranking_margin,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            total_sum += float(total.detach())
            point_sum += float(point.detach())
            rank_sum += float(rank.detach())
            progress.set_description(
                f"第 {epoch + 1}/{epochs} 轮 | {pair_index}/{len(pairs)} | "
                f"loss={total.detach():.4f} | avg={total_sum / pair_index:.4f}"
            )
        record = {
            "epoch": epoch + 1,
            "pairs": len(pairs),
            "total_loss": total_sum / len(pairs),
            "point_loss": point_sum / len(pairs),
            "ranking_loss": rank_sum / len(pairs),
        }
        history.append(record)
        is_best = float(record["total_loss"]) < best_loss
        best_loss = min(best_loss, float(record["total_loss"]))
        _save_checkpoint(output.with_name(f"{output.stem}.last{output.suffix}"), fusion, optimizer, source_config, semantic_checkpoint, settings, history, best_loss)
        if is_best:
            _save_checkpoint(output, fusion, optimizer, source_config, semantic_checkpoint, settings, history, best_loss)
    output.with_suffix(".history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _save_checkpoint(
    path: Path,
    fusion: DensePointFusion,
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
        },
        path,
    )


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


def _ranking_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    positive = logits[labels >= 0.5]
    negative = logits[labels <= 0.1]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    return torch.relu(torch.as_tensor(margin, device=logits.device, dtype=logits.dtype) - positive.mean() + negative.mean())


def _validate_loss_inputs(
    logits_a: torch.Tensor, labels_a: torch.Tensor, logits_b: torch.Tensor, labels_b: torch.Tensor,
    ranking_weight: float, ranking_margin: float,
) -> None:
    if ranking_weight < 0.0 or ranking_margin < 0.0:
        raise ValueError("排序损失权重和间隔不能为负数")
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(train_model(**train_options_from_args(args)))


if __name__ == "__main__":
    main()
