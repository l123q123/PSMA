"""在冻结 v0.5 稠密融合响应图上训练每图自适应停止阈值。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .adaptive_threshold import AdaptiveScoreThreshold, adaptive_threshold_loss, threshold_supervision_labels
from .backbones import load_resized_rgb, pool_dino_tokens, resolve_device
from .dad import select_dad_candidate_pool
from .data import load_manifest
from .geometry import confirmed_invisible_projected_points, load_pair_geometry
from .infer_v05 import _feature_tensor, _logit_tensor, load_v05_runner
from .point_fusion import point_candidate_logits
from .point_teacher import build_candidate_teacher
from .train_v05 import make_trainable_tensor


def train_adaptive_threshold(
    manifest_path: str | Path,
    dino_weights: str | Path,
    dad_weights: str | Path,
    semantic_checkpoint: str | Path,
    fusion_checkpoint: str | Path,
    output_path: str | Path,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "auto",
    max_pairs: int | None = None,
    teacher_candidate_count: int = 4096,
    teacher_sigma_px: float = 3.0,
    teacher_search_radius_px: float = 12.0,
    temperature: float = 1.0,
    seed: int = 0,
) -> Path:
    """只训练语义阈值头；DINO、语义头、DaD 与融合头始终冻结。"""
    if epochs <= 0 or learning_rate <= 0.0 or teacher_candidate_count < 2 or temperature <= 0.0:
        raise ValueError("训练轮数、学习率、候选数量或温度无效")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = resolve_device(device)
    runner = load_v05_runner(
        fusion_checkpoint, dino_weights, dad_weights, semantic_checkpoint, selected_device,
    )
    if runner.fusion_mode != "semantic_coverage":
        raise ValueError("自适应阈值仅支持 semantic_coverage 稠密融合检查点")
    for module in (runner.dino, runner.semantic_encoder, runner.fusion):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    threshold_head = AdaptiveScoreThreshold().to(selected_device)
    optimizer = torch.optim.Adam(threshold_head.parameters(), lr=learning_rate)
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
        threshold_head.train()
        loss_sum = threshold_sum_a = threshold_sum_b = 0.0
        progress = tqdm(pairs, total=len(pairs), unit="对", dynamic_ncols=True)
        for index, pair in enumerate(progress, start=1):
            image_a = load_resized_rgb(pair.image_a, runner.config.image_size)
            image_b = load_resized_rgb(pair.image_b, runner.config.image_size)
            geometry = load_pair_geometry(pair, runner.config.image_size)
            detection_a = runner.detector.detect(image_a)
            detection_b = runner.detector.detect(image_b)
            candidates_a, _ = select_dad_candidate_pool(detection_a, teacher_candidate_count)
            candidates_b, _ = select_dad_candidate_pool(detection_b, teacher_candidate_count)
            labels_a, visible_a, _ = build_candidate_teacher(
                candidates_a, candidates_b, geometry, "a_to_b", teacher_sigma_px,
                teacher_search_radius_px, runner.config.geometry_depth_tolerance,
            )
            labels_b, visible_b, _ = build_candidate_teacher(
                candidates_b, candidates_a, geometry, "b_to_a", teacher_sigma_px,
                teacher_search_radius_px, runner.config.geometry_depth_tolerance,
            )
            with torch.inference_mode():
                images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(selected_device)
                tokens = pool_dino_tokens(runner.dino(images), runner.config.patch_rows, runner.config.patch_cols)
                semantic_a, semantic_b, _, _ = runner.semantic_encoder(tokens[:1], tokens[1:])
                final_a = runner.fusion(_feature_tensor(detection_a, selected_device), _logit_tensor(detection_a, selected_device), semantic_a)
                final_b = runner.fusion(_feature_tensor(detection_b, selected_device), _logit_tensor(detection_b, selected_device), semantic_b)
            semantic_a = make_trainable_tensor(semantic_a)
            semantic_b = make_trainable_tensor(semantic_b)
            candidate_a = torch.from_numpy(candidates_a).to(selected_device)
            candidate_b = torch.from_numpy(candidates_b).to(selected_device)
            invisible_a = confirmed_invisible_projected_points(
                candidates_a, geometry, "a_to_b", runner.config.geometry_depth_tolerance
            )
            invisible_b = confirmed_invisible_projected_points(
                candidates_b, geometry, "b_to_a", runner.config.geometry_depth_tolerance
            )
            target_a = threshold_supervision_labels(labels_a, visible_a, invisible_a).to(selected_device)
            target_b = threshold_supervision_labels(labels_b, visible_b, invisible_b).to(selected_device)
            loss_a, tau_a = adaptive_threshold_loss(
                final_a, semantic_a, candidate_a, target_a, threshold_head,
                runner.config.image_size, temperature, runner.config.total_points,
            )
            loss_b, tau_b = adaptive_threshold_loss(
                final_b, semantic_b, candidate_b, target_b, threshold_head,
                runner.config.image_size, temperature, runner.config.total_points,
            )
            loss = 0.5 * (loss_a + loss_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            threshold_sum_a += float(tau_a.detach())
            threshold_sum_b += float(tau_b.detach())
            progress.set_description(f"第 {epoch + 1}/{epochs} 轮 | {index}/{len(pairs)} | loss={loss:.4f} | avg={loss_sum / index:.4f} | tau=({tau_a.item():.3f},{tau_b.item():.3f})")
        record = {
            "epoch": epoch + 1,
            "pairs": len(pairs),
            "loss": loss_sum / len(pairs),
            "mean_threshold_a": threshold_sum_a / len(pairs),
            "mean_threshold_b": threshold_sum_b / len(pairs),
        }
        history.append(record)
        if record["loss"] < best_loss:
            best_loss = float(record["loss"])
            _save_checkpoint(
                output, threshold_head, runner.config, fusion_checkpoint, semantic_checkpoint,
                history, best_loss, temperature,
            )
        _save_checkpoint(
            output.with_name(f"{output.stem}.last{output.suffix}"), threshold_head, runner.config,
            fusion_checkpoint, semantic_checkpoint, history, best_loss, temperature,
        )
    output.with_suffix(".history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _save_checkpoint(
    path: Path,
    head: AdaptiveScoreThreshold,
    source_config: object,
    fusion_checkpoint: str | Path,
    semantic_checkpoint: str | Path,
    history: list[dict[str, float | int]],
    best_loss: float,
    temperature: float,
) -> None:
    torch.save({
        "format": "pasm_v07_adaptive_threshold",
        "threshold_head": head.state_dict(),
        "epoch": len(history),
        "history": history,
        "best_loss": best_loss,
        "fusion_checkpoint": str(Path(fusion_checkpoint)),
        "semantic_checkpoint": str(Path(semantic_checkpoint)),
        "source_config": asdict(source_config),
        "temperature": temperature,
    }, path)


def train_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    """将 CLI 文件字段转换为训练函数使用的参数名。"""
    options = vars(args).copy()
    options["manifest_path"] = options.pop("manifest")
    options["output_path"] = options.pop("output")
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 PASM v0.7 自适应停止阈值")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--dad-weights", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--teacher-candidate-count", type=int, default=4096)
    parser.add_argument("--teacher-sigma-px", type=float, default=3.0)
    parser.add_argument("--teacher-search-radius-px", type=float, default=12.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(train_adaptive_threshold(**train_options_from_args(args)))


if __name__ == "__main__":
    main()
