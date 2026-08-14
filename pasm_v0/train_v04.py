# 训练 PASM v0.4 的单轮双图 Patch 价值模型。

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from tqdm.auto import tqdm

from .backbones import FrozenFeatureExtractor, load_resized_rgb, pool_dino_tokens, resolve_device
from .config import PasmV0Config, dino_token_static_config
from .dad import FrozenDaDDetector, select_dad_candidate_pool, select_patch_budgeted_points
from .data import PairRecord, load_manifest
from .geometry import load_pair_geometry, projected_match_targets, projected_patch_support_fractions
from .matching import PairSemanticEncoder, global_edge_weights, sample_feature_map
from .sampling import PatchLayout, allocate_patch_budget
from .v04_targets import TargetKind, load_calibrated_patch_targets, load_patch_targets, normalize_patch_target


@dataclass(frozen=True)
class TeacherValidationPair:
    """保存 held-out 图像对及其固定的双图几何教师目标。"""

    pair: PairRecord
    raw_a: np.ndarray
    raw_b: np.ndarray
    distribution_a: np.ndarray
    distribution_b: np.ndarray


def train_model(
    manifest_path: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    dad_weights: str | Path,
    target_cache: str | Path,
    target_kind: TargetKind,
    output_path: str | Path,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    hidden_dim: int = 128,
    heads: int = 4,
    device: str = "auto",
    max_pairs: int | None = None,
    seed: int = 0,
    resume_path: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    loss_mode: str = "match_value",
    cross_attention_layers: int = 1,
    patch_grid: int = 37,
    match_candidate_count: int = 2048,
    match_weight: float = 0.1,
    validation_image_root: str | Path | None = None,
    validation_manifest: str | Path | None = None,
    validation_every: int = 2,
    early_stop_patience: int = 0,
    validation_max_pairs: int = 9,
) -> Path:
    """用指定 A/B Patch 监督训练单轮静态双图语义编码器。"""
    if epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("训练轮数和学习率必须为正数")
    if target_kind not in {"geometry", "quality", "geometry_calibrated"}:
        raise ValueError("目标类型必须是 geometry、quality 或 geometry_calibrated")
    if loss_mode not in {"match_value", "value_only", "calibrated_shape", "calibrated_match"}:
        raise ValueError("损失模式必须是 match_value、value_only、calibrated_shape 或 calibrated_match")
    if (target_kind == "geometry_calibrated") != (loss_mode in {"calibrated_shape", "calibrated_match"}):
        raise ValueError("geometry_calibrated 必须且只能使用 calibrated_shape 或 calibrated_match 损失")
    if cross_attention_layers <= 0:
        raise ValueError("Cross-Attention 层数必须为正数")
    if resume_path is not None and init_checkpoint is not None:
        raise ValueError("--resume 与 --init-checkpoint 不能同时使用")
    if match_candidate_count <= 0:
        raise ValueError("匹配候选池大小必须为正数")
    if match_weight < 0.0:
        raise ValueError("匹配损失权重不能为负数")
    if validation_every <= 0 or early_stop_patience < 0 or validation_max_pairs <= 0:
        raise ValueError("验证间隔和验证图像对数必须为正数，早停耐心值不能为负数")
    if validation_image_root is not None and validation_manifest is not None:
        raise ValueError("--validation-image-root 与 --validation-manifest 不能同时使用")
    set_training_seed(seed)
    selected_device = resolve_device(device)
    config = dino_token_static_config(patch_grid)
    pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("最多训练图像对数必须为正数")
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("清单中没有正图像对")
    print(f"训练设备：{selected_device} | 监督：{target_kind} | 图像对：{len(pairs)}", flush=True)
    extractor = FrozenFeatureExtractor(config, dino_weights, loma_weights, device=str(selected_device))
    detector = (
        FrozenDaDDetector(dad_weights, config.image_size, selected_device)
        if loss_mode != "calibrated_shape"
        else None
    )
    encoder = PairSemanticEncoder(
        token_dim=1024,
        hidden_dim=hidden_dim,
        heads=heads,
        cross_attention_layers=cross_attention_layers,
    ).to(selected_device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    output = Path(output_path)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_validation_metric = float("inf")
    validation_no_improvement = 0
    start_epoch = 0
    if init_checkpoint is not None:
        initialize_encoder_from_checkpoint(init_checkpoint, encoder, selected_device)
    if resume_path is not None:
        start_epoch, history, best_loss, best_validation_metric, validation_no_improvement = load_v04_checkpoint(
            resume_path, encoder, optimizer, selected_device, target_kind, loss_mode, cross_attention_layers
        )
        ensure_v04_checkpoint(
            output, encoder, optimizer, config, hidden_dim, heads, history, best_loss,
            target_kind, target_cache, loss_mode, cross_attention_layers,
            match_candidate_count, match_weight, best_validation_metric, validation_no_improvement,
        )
    if start_epoch >= epochs:
        raise ValueError("目标训练轮数必须大于检查点已完成轮数")
    layout = PatchLayout(config)
    validation_pairs: list[TeacherValidationPair] = []
    validation_enabled = validation_image_root is not None or validation_manifest is not None
    if validation_enabled:
        if loss_mode not in {"calibrated_shape", "calibrated_match"}:
            raise ValueError("几何教师早停当前只支持 geometry_calibrated 训练")
        validation_pairs = prepare_teacher_validation_pairs(
            validation_image_root,
            manifest_path,
            config,
            validation_max_pairs,
            validation_manifest=validation_manifest,
        )
        if best_validation_metric == float("-inf"):
            best_validation_metric = float("inf")
            validation_no_improvement = 0
    for epoch in range(start_epoch, epochs):
        encoder.train()
        total_sum = match_sum = value_sum = 0.0
        local_sum = regression_sum = shape_sum = 0.0
        progress = tqdm(pairs, total=len(pairs), unit="对", dynamic_ncols=True)
        for pair_index, pair in enumerate(progress, start=1):
            image_a = load_resized_rgb(pair.image_a, config.image_size)
            image_b = load_resized_rgb(pair.image_b, config.image_size)
            features = extractor.extract(image_a, image_b)
            semantic_a, semantic_b, values_a, values_b = encoder(features.tokens_a, features.tokens_b)
            if loss_mode in {"calibrated_shape", "calibrated_match"}:
                raw_a, raw_b, distribution_a, distribution_b = load_calibrated_patch_targets(
                    target_cache, pair.pair_id, config
                )
                local_a, regression_a, shape_a, total_a = v04_calibrated_shape_loss(
                    values_a[0], raw_a, distribution_a
                )
                local_b, regression_b, shape_b, total_b = v04_calibrated_shape_loss(
                    values_b[0], raw_b, distribution_b
                )
                calibrated_loss = 0.5 * (total_a + total_b)
                if loss_mode == "calibrated_match":
                    if detector is None:
                        raise RuntimeError("DaD 检测器未初始化")
                    geometry = load_pair_geometry(pair, config.image_size)
                    points_a, _ = select_dad_candidate_pool(
                        detector.detect(image_a), match_candidate_count
                    )
                    points_b, _ = select_dad_candidate_pool(
                        detector.detect(image_b), match_candidate_count
                    )
                    patch_ids_a = layout.patch_ids(points_a)
                    patch_ids_b = layout.patch_ids(points_b)
                    edge_weights = _edge_weights(
                        features.descriptor_grid_a,
                        features.descriptor_grid_b,
                        points_a,
                        points_b,
                        patch_ids_a,
                        patch_ids_b,
                        semantic_a[0],
                        semantic_b[0],
                        config,
                        selected_device,
                    )
                    target_a, valid_a, target_b, valid_b = projected_match_targets(
                        points_a,
                        points_b,
                        geometry,
                        config.geometry_match_threshold_px,
                        config.geometry_depth_tolerance,
                    )
                    match_loss = 0.5 * (
                        point_match_loss(edge_weights, target_a, valid_a)
                        + point_match_loss(edge_weights.transpose(0, 1), target_b, valid_b)
                    )
                    total_loss = calibrated_loss + match_weight * match_loss
                else:
                    total_loss = calibrated_loss
                    match_loss = total_loss * 0.0
                value_loss = calibrated_loss
                local_loss = 0.5 * (local_a + local_b)
                regression_loss = 0.5 * (regression_a + regression_b)
                shape_loss = 0.5 * (shape_a + shape_b)
            else:
                if detector is None:
                    raise RuntimeError("DaD 检测器未初始化")
                geometry = load_pair_geometry(pair, config.image_size)
                points_a, patch_ids_a = _select_dad_points(detector, image_a, values_a[0], layout)
                points_b, patch_ids_b = _select_dad_points(detector, image_b, values_b[0], layout)
                edge_weights = _edge_weights(
                    features.descriptor_grid_a,
                    features.descriptor_grid_b,
                    points_a,
                    points_b,
                    patch_ids_a,
                    patch_ids_b,
                    semantic_a[0],
                    semantic_b[0],
                    config,
                    selected_device,
                )
                target_a, valid_a, target_b, valid_b = projected_match_targets(
                    points_a,
                    points_b,
                    geometry,
                    config.geometry_match_threshold_px,
                    config.geometry_depth_tolerance,
                )
                patch_target_a, patch_target_b = load_patch_targets(
                    target_cache, pair.pair_id, config, target_kind
                )
                match_a, value_a, total_a = v04_loss_terms(
                    edge_weights, target_a, valid_a, values_a[0], patch_target_a, loss_mode
                )
                match_b, value_b, total_b = v04_loss_terms(
                    edge_weights.transpose(0, 1), target_b, valid_b, values_b[0], patch_target_b, loss_mode
                )
                match_loss = 0.5 * (match_a + match_b)
                value_loss = 0.5 * (value_a + value_b)
                total_loss = 0.5 * (total_a + total_b)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            total_sum += float(total_loss.detach())
            match_sum += float(match_loss.detach())
            value_sum += float(value_loss.detach())
            if loss_mode in {"calibrated_shape", "calibrated_match"}:
                local_sum += float(local_loss.detach())
                regression_sum += float(regression_loss.detach())
                shape_sum += float(shape_loss.detach())
            count = pair_index
            progress.set_description(
                f"第 {epoch + 1}/{epochs} 轮 | 图像对 {count}/{len(pairs)} | "
                f"total={total_loss.detach():.4f} | 平均 total={total_sum / count:.4f} | "
                f"match={match_sum / count:.4f} | value={value_sum / count:.4f}"
            )
        epoch_record = {
            "epoch": epoch + 1,
            "pairs": len(pairs),
            "total_loss": total_sum / len(pairs),
            "match_loss": match_sum / len(pairs),
            "value_loss": value_sum / len(pairs),
        }
        if loss_mode in {"calibrated_shape", "calibrated_match"}:
            epoch_record.update(
                local_loss=local_sum / len(pairs),
                regression_loss=regression_sum / len(pairs),
                shape_loss=shape_sum / len(pairs),
            )
        history.append(epoch_record)
        is_best = float(epoch_record["total_loss"]) < best_loss
        if is_best:
            best_loss = float(epoch_record["total_loss"])
        last_checkpoint = _last_checkpoint_path(output)
        save_v04_checkpoint(
            last_checkpoint, encoder, optimizer, config, hidden_dim, heads, history,
            best_loss, target_kind, target_cache, loss_mode,
            cross_attention_layers,
            match_candidate_count, match_weight, best_validation_metric, validation_no_improvement,
        )
        validation_stop = False
        if validation_enabled and (epoch + 1) % validation_every == 0:
            validation_metrics = evaluate_teacher_validation(
                extractor, encoder, validation_pairs, config, selected_device
            )
            metric = validation_metrics["total_loss"]
            epoch_record.update({f"validation_{name}": value for name, value in validation_metrics.items()})
            validation_improved, best_validation_metric, validation_no_improvement, validation_stop = update_validation_early_stopping(
                metric, best_validation_metric, validation_no_improvement, early_stop_patience
            )
            print(
                f"验证 epoch={epoch + 1} | teacher total={metric:.4f} | "
                f"最佳={best_validation_metric:.4f} | 未提升={validation_no_improvement}/"
                f"{early_stop_patience if early_stop_patience else '关闭'}",
                flush=True,
            )
        else:
            validation_improved = False
        save_v04_checkpoint(
            last_checkpoint, encoder, optimizer, config, hidden_dim, heads, history,
            best_loss, target_kind, target_cache, loss_mode,
            cross_attention_layers,
            match_candidate_count, match_weight, best_validation_metric, validation_no_improvement,
        )
        if (not validation_enabled and is_best) or (validation_enabled and validation_improved):
            save_v04_checkpoint(
                output, encoder, optimizer, config, hidden_dim, heads, history,
                best_loss, target_kind, target_cache, loss_mode,
                cross_attention_layers,
                match_candidate_count, match_weight, best_validation_metric, validation_no_improvement,
            )
        if validation_stop:
            print(f"验证教师损失连续 {early_stop_patience} 次未下降，提前停止训练。", flush=True)
            break
    output.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def v04_loss_terms(
    edge_weights: torch.Tensor,
    target_indices: torch.Tensor,
    valid_rows: torch.Tensor,
    patch_values: torch.Tensor,
    patch_target: np.ndarray | torch.Tensor,
    loss_mode: str = "match_value",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算单方向可微点匹配损失、Patch 价值损失及其和。"""
    if loss_mode not in {"match_value", "value_only"}:
        raise ValueError("损失模式必须是 match_value 或 value_only")
    if edge_weights.ndim != 2 or edge_weights.shape[0] != target_indices.numel():
        raise ValueError("边权和点级目标形状不一致")
    if valid_rows.shape != target_indices.shape or patch_values.ndim != 1:
        raise ValueError("有效行或 Patch 价值形状无效")
    targets = target_indices.to(edge_weights.device, dtype=torch.long)
    valid = valid_rows.to(edge_weights.device, dtype=torch.bool) & (targets >= 0)
    zero = edge_weights.sum() * 0.0 + patch_values.sum() * 0.0
    if torch.any(valid):
        rows = torch.arange(edge_weights.shape[0], device=edge_weights.device)[valid]
        match_loss = -torch.log(edge_weights[rows, targets[valid]].clamp_min(1e-8)).mean()
    else:
        match_loss = zero
    target = torch.as_tensor(patch_target, dtype=patch_values.dtype, device=patch_values.device)
    if target.ndim != 1 or target.numel() != patch_values.numel():
        raise ValueError("Patch 价值监督形状无效")
    if torch.any(target < 0.0) or not torch.isclose(target.sum(), torch.ones((), device=target.device, dtype=target.dtype), rtol=0.0, atol=1e-5):
        raise ValueError("Patch 价值监督必须是和为一的非负分布")
    distribution = patch_values.clamp_min(1e-8)
    distribution = distribution / distribution.sum().clamp_min(1e-8)
    value_loss = -(target * torch.log(distribution)).sum()
    total_loss = value_loss if loss_mode == "value_only" else match_loss + value_loss
    return match_loss, value_loss, total_loss


def point_match_loss(
    edge_weights: torch.Tensor,
    target_indices: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    """计算给定几何真值对应边的平均负对数双向软匹配权重。"""
    if edge_weights.ndim != 2 or edge_weights.shape[0] != target_indices.numel():
        raise ValueError("边权和点级目标形状不一致")
    if valid_rows.shape != target_indices.shape:
        raise ValueError("有效行形状无效")
    targets = target_indices.to(edge_weights.device, dtype=torch.long)
    valid = valid_rows.to(edge_weights.device, dtype=torch.bool) & (targets >= 0)
    if not torch.any(valid):
        return edge_weights.sum() * 0.0
    rows = torch.arange(edge_weights.shape[0], device=edge_weights.device)[valid]
    return -torch.log(edge_weights[rows, targets[valid]].clamp_min(1e-8)).mean()


def v04_calibrated_shape_loss(
    patch_values: torch.Tensor,
    raw_target: np.ndarray | torch.Tensor,
    distribution_target: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """以独立 raw support 校准 Patch 值，并保留弱的全图形状约束。"""
    if patch_values.ndim != 1:
        raise ValueError("Patch 预测必须是一维数组")
    raw = torch.as_tensor(raw_target, dtype=patch_values.dtype, device=patch_values.device)
    distribution = torch.as_tensor(
        distribution_target, dtype=patch_values.dtype, device=patch_values.device
    )
    if raw.shape != patch_values.shape or distribution.shape != patch_values.shape:
        raise ValueError("校准监督与 Patch 预测形状不一致")
    if torch.any(raw < 0.0) or torch.any(raw > 1.0):
        raise ValueError("raw support 必须位于 [0, 1]")
    if torch.any(distribution < 0.0) or not torch.isclose(
        distribution.sum(), torch.ones((), dtype=distribution.dtype, device=distribution.device),
        rtol=0.0, atol=1e-5,
    ):
        raise ValueError("shape 监督必须是和为一的非负分布")
    predictions = patch_values.clamp(1e-6, 1.0 - 1e-6)
    local_loss = functional.binary_cross_entropy(predictions, raw)
    regression_loss = functional.smooth_l1_loss(predictions, raw)
    normalized = predictions / predictions.sum().clamp_min(1e-8)
    shape_loss = functional.kl_div(torch.log(normalized), distribution, reduction="batchmean")
    total_loss = local_loss + 0.25 * regression_loss + 0.05 * shape_loss
    return local_loss, regression_loss, shape_loss, total_loss


def set_training_seed(seed: int) -> None:
    """固定 CPU 与 CUDA 随机状态，保证 A/B 从相同初始化开始。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_v04_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    optimizer: torch.optim.Optimizer,
    config: PasmV0Config,
    hidden_dim: int,
    heads: int,
    history: list[dict[str, float | int]],
    best_loss: float,
    target_kind: TargetKind,
    target_cache: str | Path,
    loss_mode: str = "match_value",
    cross_attention_layers: int = 1,
    match_candidate_count: int = 2048,
    match_weight: float = 0.1,
    best_validation_metric: float = float("inf"),
    validation_no_improvement: int = 0,
) -> None:
    """保存 v0.4 语义编码器、优化器、监督身份和完整续训状态。"""
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "semantic_encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": len(history),
            "history": history,
            "best_loss": best_loss,
            "config": asdict(config),
            "token_dim": 1024,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "target_kind": target_kind,
            "target_cache": str(Path(target_cache)),
            "loss_mode": loss_mode,
            "cross_attention_layers": cross_attention_layers,
            "match_candidate_count": match_candidate_count,
            "match_weight": match_weight,
            "best_validation_metric": best_validation_metric,
            "validation_no_improvement": validation_no_improvement,
        },
        checkpoint,
    )


def load_v04_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    expected_target_kind: TargetKind | None = None,
    expected_loss_mode: str | None = None,
    expected_cross_attention_layers: int | None = None,
) -> tuple[int, list[dict[str, float | int]], float, float, int]:
    """恢复 v0.4 检查点的语义编码器、优化器和训练历史。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "semantic_encoder" not in payload:
        raise ValueError("续训检查点不含语义编码器权重")
    if expected_target_kind is not None and payload.get("target_kind") != expected_target_kind:
        raise ValueError("续训检查点的 Patch 监督类型与当前训练不一致")
    if expected_loss_mode is not None and payload.get("loss_mode", "match_value") != expected_loss_mode:
        raise ValueError("续训检查点的损失模式与当前训练不一致")
    if (
        expected_cross_attention_layers is not None
        and int(payload.get("cross_attention_layers", 1)) != expected_cross_attention_layers
    ):
        raise ValueError("续训检查点的 Cross-Attention 层数与当前训练不一致")
    encoder.load_state_dict(payload["semantic_encoder"])
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    history = payload.get("history", [])
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise ValueError("续训检查点历史格式无效")
    epoch = int(payload.get("epoch", len(history)))
    best_loss = float(payload.get("best_loss", min((float(item["total_loss"]) for item in history), default=float("inf"))))
    best_validation_metric = float(payload.get("best_validation_metric", float("inf")))
    validation_no_improvement = int(payload.get("validation_no_improvement", 0))
    return epoch, [dict(item) for item in history], best_loss, best_validation_metric, validation_no_improvement


def initialize_encoder_from_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    device: torch.device | str,
) -> None:
    """只载入已有语义编码器权重，用于新目标的微调初始化。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "semantic_encoder" not in payload:
        raise ValueError("初始化检查点不含语义编码器权重")
    encoder.load_state_dict(payload["semantic_encoder"])


def ensure_v04_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    optimizer: torch.optim.Optimizer,
    config: PasmV0Config,
    hidden_dim: int,
    heads: int,
    history: list[dict[str, float | int]],
    best_loss: float,
    target_kind: TargetKind,
    target_cache: str | Path,
    loss_mode: str = "match_value",
    cross_attention_layers: int = 1,
    match_candidate_count: int = 2048,
    match_weight: float = 0.1,
    best_validation_metric: float = float("inf"),
    validation_no_improvement: int = 0,
) -> None:
    """在新输出路径不存在时将恢复状态保存为当前最佳模型。"""
    if not Path(path).is_file():
        save_v04_checkpoint(
            path, encoder, optimizer, config, hidden_dim, heads, history,
            best_loss, target_kind, target_cache, loss_mode, cross_attention_layers,
            match_candidate_count, match_weight, best_validation_metric, validation_no_improvement,
        )


def prepare_teacher_validation_pairs(
    image_root: str | Path | None,
    training_manifest: str | Path,
    config: PasmV0Config,
    max_pairs: int,
    validation_manifest: str | Path | None = None,
) -> list[TeacherValidationPair]:
    """一次生成 held-out 图像对的 stride=2 几何教师真值。

    传入 validation_manifest 时直接读取自定义清单；否则保留旧的固定九组加载逻辑。
    """
    from .evaluate_v03 import load_heldout_pairs, validate_heldout_against_training

    if validation_manifest is not None:
        pairs = [pair for pair in load_manifest(validation_manifest) if pair.label == 1][:max_pairs]
        if not pairs:
            raise ValueError("验证清单中没有正图像对")
    else:
        if image_root is None:
            raise ValueError("未指定验证图像根目录或验证清单")
        pairs = load_heldout_pairs(image_root)[:max_pairs]
    validate_heldout_against_training(pairs, training_manifest)
    layout = PatchLayout(config)
    result = []
    for index, pair in enumerate(pairs, start=1):
        geometry = load_pair_geometry(pair, config.image_size)
        raw_a, raw_b = projected_patch_support_fractions(
            geometry, layout, stride=2, depth_tolerance=config.geometry_depth_tolerance
        )
        result.append(TeacherValidationPair(
            pair=pair,
            raw_a=raw_a,
            raw_b=raw_b,
            distribution_a=normalize_patch_target(raw_a, config.patch_count),
            distribution_b=normalize_patch_target(raw_b, config.patch_count),
        ))
        print(f"准备验证教师 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
    return result


def evaluate_teacher_validation(
    extractor: FrozenFeatureExtractor,
    encoder: PairSemanticEncoder,
    validation_pairs: list[TeacherValidationPair],
    config: PasmV0Config,
    device: torch.device,
) -> dict[str, float]:
    """直接比较语义头 Patch 输出与 held-out 几何教师，不经过下游选点。"""
    if not validation_pairs:
        raise ValueError("验证教师图像对不能为空")
    was_training = encoder.training
    encoder.eval()
    total_sum = local_sum = regression_sum = shape_sum = 0.0
    with torch.no_grad():
        for item in validation_pairs:
            image_a = load_resized_rgb(item.pair.image_a, config.image_size)
            image_b = load_resized_rgb(item.pair.image_b, config.image_size)
            images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(device)
            tokens = pool_dino_tokens(
                extractor.dino(images), config.patch_rows, config.patch_cols
            )
            _, _, values_a, values_b = encoder(tokens[:1], tokens[1:])
            local_a, regression_a, shape_a, total_a = v04_calibrated_shape_loss(
                values_a[0], item.raw_a, item.distribution_a
            )
            local_b, regression_b, shape_b, total_b = v04_calibrated_shape_loss(
                values_b[0], item.raw_b, item.distribution_b
            )
            total_sum += float(0.5 * (total_a + total_b))
            local_sum += float(0.5 * (local_a + local_b))
            regression_sum += float(0.5 * (regression_a + regression_b))
            shape_sum += float(0.5 * (shape_a + shape_b))
    if was_training:
        encoder.train()
    count = len(validation_pairs)
    return {
        "total_loss": total_sum / count,
        "local_loss": local_sum / count,
        "regression_loss": regression_sum / count,
        "shape_loss": shape_sum / count,
    }


def update_validation_early_stopping(
    metric: float,
    best_metric: float,
    no_improvement_count: int,
    patience: int,
) -> tuple[bool, float, int, bool]:
    """根据教师验证损失是否严格下降，更新最佳值、耐心计数和可选停止标记。"""
    if not np.isfinite(metric) or (not np.isfinite(best_metric) and best_metric != float("inf")) or patience < 0 or no_improvement_count < 0:
        raise ValueError("验证早停状态无效")
    if metric < best_metric:
        return True, metric, 0, False
    next_count = no_improvement_count + 1
    return False, best_metric, next_count, patience > 0 and next_count >= patience


def _select_dad_points(
    detector: FrozenDaDDetector,
    image: np.ndarray,
    patch_values: torch.Tensor,
    layout: PatchLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """将当前价值头输出转换为静态预算，并用 DaD 在预算内选点。"""
    budget = allocate_patch_budget(
        patch_values.detach().cpu().numpy(), layout.config, round_index=0
    )
    points, patch_ids, _ = select_patch_budgeted_points(detector.detect(image), layout, budget)
    return points, patch_ids


def _edge_weights(
    descriptor_grid_a: np.ndarray,
    descriptor_grid_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    patch_ids_a: np.ndarray,
    patch_ids_b: np.ndarray,
    semantic_a: torch.Tensor,
    semantic_b: torch.Tensor,
    config: PasmV0Config,
    device: torch.device,
) -> torch.Tensor:
    """读取冻结 LoMa 局部描述子并与联合语义组成可微点对边权。"""
    grid_a = torch.as_tensor(descriptor_grid_a, dtype=torch.float32, device=device)
    grid_b = torch.as_tensor(descriptor_grid_b, dtype=torch.float32, device=device)
    descriptors_a = sample_feature_map(grid_a, torch.from_numpy(points_a), (config.image_size, config.image_size))
    descriptors_b = sample_feature_map(grid_b, torch.from_numpy(points_b), (config.image_size, config.image_size))
    return global_edge_weights(
        descriptors_a,
        descriptors_b,
        semantic_a,
        semantic_b,
        torch.from_numpy(patch_ids_a).to(device),
        torch.from_numpy(patch_ids_b).to(device),
        config.temperature,
    )


def _last_checkpoint_path(output: Path) -> Path:
    """返回与最佳 checkpoint 同目录的最后训练状态路径。"""
    return output.with_name(f"{output.stem}.last{output.suffix}")


def main() -> None:
    """解析 v0.4 A/B 训练参数并保存最佳语义编码器检查点。"""
    parser = argparse.ArgumentParser(description="训练 PASM v0.4 的单轮静态语义编码器")
    parser.add_argument("--manifest", type=Path, required=True, help="MegaDepth 训练图像对清单")
    parser.add_argument("--dino-weights", type=Path, required=True, help="DINOv2 权重路径")
    parser.add_argument("--loma-weights", type=Path, required=True, help="LoMa-B128 描述子权重路径")
    parser.add_argument("--dad-weights", type=Path, required=True, help="DaD 检测器权重路径")
    parser.add_argument("--target-cache", type=Path, required=True, help="对应 A/B Patch 监督缓存目录")
    parser.add_argument("--target-kind", choices=("geometry", "quality", "geometry_calibrated"), required=True, help="Patch 监督类型")
    parser.add_argument("--output", type=Path, required=True, help="最佳模型输出路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练总轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam 学习率")
    parser.add_argument("--hidden-dim", type=int, default=128, help="语义隐藏维度")
    parser.add_argument("--heads", type=int, default=4, help="双图交叉注意力头数")
    parser.add_argument("--cross-attention-layers", type=int, default=1, help="双向 Cross-Attention 层数")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--max-pairs", type=int, help="最多使用的正图像对数量")
    parser.add_argument("--seed", type=int, default=0, help="保留的实验随机种子记录")
    parser.add_argument("--resume", type=Path, help="恢复指定 v0.4 检查点并训练到目标总轮数")
    parser.add_argument("--init-checkpoint", type=Path, help="只加载语义编码器权重，并从第 0 轮开始微调")
    parser.add_argument("--value-only", action="store_true", help="只使用 Patch 价值损失训练")
    parser.add_argument(
        "--loss-mode", choices=("match_value", "value_only", "calibrated_shape", "calibrated_match"),
        help="显式指定训练损失；与 --value-only 不能同时使用",
    )
    parser.add_argument("--patch-grid", type=int, default=37, choices=(16, 37), help="Patch 网格边长")
    parser.add_argument("--match-candidate-count", type=int, default=2048, help="calibrated_match 的每图冻结 DaD 候选数")
    parser.add_argument("--match-weight", type=float, default=0.1, help="calibrated_match 点级匹配损失权重")
    parser.add_argument("--validation-image-root", type=Path, help="固定 held-out 验证图像、深度和相机目录；提供后按几何教师损失选最佳模型")
    parser.add_argument("--validation-manifest", type=Path, help="自定义 held-out 验证图像对清单；与 --validation-image-root 二选一")
    parser.add_argument("--validation-every", type=int, default=2, help="每隔多少 epoch 计算一次 held-out 几何教师损失")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="验证教师损失连续未下降多少次后提前停止；0 表示关闭早停")
    parser.add_argument("--validation-max-pairs", type=int, default=9, help="每次验证使用固定 held-out 的前几组")
    args = parser.parse_args()
    if args.value_only and args.loss_mode is not None:
        parser.error("--value-only 与 --loss-mode 不能同时使用")
    loss_mode = args.loss_mode or ("value_only" if args.value_only else "match_value")
    print(
        train_model(
            manifest_path=args.manifest,
            dino_weights=args.dino_weights,
            loma_weights=args.loma_weights,
            dad_weights=args.dad_weights,
            target_cache=args.target_cache,
            target_kind=args.target_kind,
            output_path=args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            device=args.device,
            max_pairs=args.max_pairs,
            seed=args.seed,
            resume_path=args.resume,
            init_checkpoint=args.init_checkpoint,
            loss_mode=loss_mode,
            cross_attention_layers=args.cross_attention_layers,
            patch_grid=args.patch_grid,
            match_candidate_count=args.match_candidate_count,
            match_weight=args.match_weight,
            validation_image_root=args.validation_image_root,
            validation_manifest=args.validation_manifest,
            validation_every=args.validation_every,
            early_stop_patience=args.early_stop_patience,
            validation_max_pairs=args.validation_max_pairs,
        )
    )


if __name__ == "__main__":
    main()
