# 直接诊断 v0.6 calibrated 语义热图与 raw 几何教师的差异。

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
from .config import dino_token_static_config
from .diagnose_v04_distribution import calibrated_support_metrics, spearman, stats, top_iou
from .evaluate_v03 import load_heldout_pairs, save_patch_heatmap
from .geometry import load_pair_geometry, projected_patch_support_fractions
from .infer import load_runner
from .sampling import PatchLayout


def semantic_support_row(
    model: str, prediction: np.ndarray, teacher: np.ndarray
) -> dict[str, float | str]:
    """统计语义热图相对 raw geometry support 的直接误差和覆盖程度。"""
    predicted = np.asarray(prediction, dtype=np.float32).reshape(-1)
    target = np.asarray(teacher, dtype=np.float32).reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("预测和教师热图形状不一致")
    metrics = calibrated_support_metrics(predicted, target)
    predicted_stats = stats(predicted)
    teacher_stats = stats(target)
    return {
        "model": model,
        **metrics,
        "effective_patches": predicted_stats["effective"],
        "teacher_effective_patches": teacher_stats["effective"],
        "spearman": spearman(predicted, target),
        "top20_iou": top_iou(predicted, target),
    }


def diagnose_calibrated_semantics(
    image_root: str | Path,
    dino_weights: str | Path,
    baseline_checkpoint: str | Path,
    calibrated_checkpoint: str | Path,
    output: str | Path,
    device: str = "cuda",
    max_pairs: int = 9,
    patch_grid: int = 16,
    support_stride: int = 2,
) -> Path:
    """在固定 held-out 九组上直接比较教师、旧语义头和 calibrated 语义头。"""
    if max_pairs <= 0 or support_stride <= 0:
        raise ValueError("图像对数量和几何支持步长必须为正数")
    config = dino_token_static_config(patch_grid)
    layout = PatchLayout(config)
    selected_device = resolve_device(device)
    pairs = load_heldout_pairs(image_root)[:max_pairs]
    if not pairs:
        raise ValueError("没有找到 held-out 图像对")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    dino = DINOv2PatchEncoder(dino_weights, config.image_size).to(selected_device).eval()
    runners = {
        "baseline_static16": load_runner(baseline_checkpoint, selected_device),
        "calibrated_epoch40": load_runner(calibrated_checkpoint, selected_device),
    }
    for runner in runners.values():
        if runner.config != config:
            raise ValueError(f"checkpoint 配置不是 {patch_grid}×{patch_grid} 静态语义模型")
        runner.semantic_encoder.eval()

    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        image_a = load_resized_rgb(pair.image_a, config.image_size)
        image_b = load_resized_rgb(pair.image_b, config.image_size)
        geometry = load_pair_geometry(pair, config.image_size)
        teacher_a, teacher_b = projected_patch_support_fractions(
            geometry, layout, support_stride, config.geometry_depth_tolerance
        )
        images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(selected_device)
        with torch.inference_mode():
            tokens = pool_dino_tokens(dino(images), config.patch_rows, config.patch_cols)
            predictions = {
                model: tuple(value[0].cpu().numpy().reshape(-1) for value in runner.semantic_encoder(tokens[:1], tokens[1:])[2:])
                for model, runner in runners.items()
            }

        pair_root = root / f"{index:02d}_{pair.pair_id}"
        pair_root.mkdir(parents=True, exist_ok=True)
        save_patch_heatmap(pair_root / "teacher_raw_a.png", teacher_a, config.patch_rows, config.patch_cols, "A raw 几何教师")
        save_patch_heatmap(pair_root / "teacher_raw_b.png", teacher_b, config.patch_rows, config.patch_cols, "B raw 几何教师")
        payload: dict[str, np.ndarray] = {"teacher_raw_a": teacher_a, "teacher_raw_b": teacher_b}
        for side, teacher in (("a", teacher_a), ("b", teacher_b)):
            for model, prediction_pair in predictions.items():
                prediction = prediction_pair[0 if side == "a" else 1]
                save_patch_heatmap(
                    pair_root / f"{model}_{side}.png",
                    prediction,
                    config.patch_rows,
                    config.patch_cols,
                    f"{side.upper()} {model} 预测",
                )
                save_patch_heatmap(
                    pair_root / f"{model}_minus_teacher_{side}.png",
                    prediction - teacher,
                    config.patch_rows,
                    config.patch_cols,
                    f"{side.upper()} {model} - raw 教师",
                )
                rows.append({"pair_index": index, "pair_id": pair.pair_id, "side": side, **semantic_support_row(model, prediction, teacher)})
                payload[f"{model}_{side}"] = prediction
        np.savez_compressed(pair_root / "values.npz", **payload)
        print(f"完成 {index}/{len(pairs)}：{pair.pair_id}", flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="直接诊断 calibrated 语义热图与 raw 几何教师")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--calibrated-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=9)
    parser.add_argument("--patch-grid", type=int, choices=(16, 37), default=16)
    parser.add_argument("--support-stride", type=int, default=2)
    args = parser.parse_args()
    print(diagnose_calibrated_semantics(
        args.image_root, args.dino_weights, args.baseline_checkpoint, args.calibrated_checkpoint,
        args.output, args.device, args.max_pairs, args.patch_grid, args.support_stride,
    ))

if __name__ == "__main__":
    main()
