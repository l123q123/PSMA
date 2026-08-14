# 诊断 v0.4 教师目标、原始 Patch 价值和最终整数预算的分布差异。

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
from .config import dino_token_static_config
from .evaluate_v03 import load_heldout_pairs, save_patch_heatmap
from .geometry import load_pair_geometry, projected_patch_support_targets
from .infer import load_runner
from .sampling import PatchLayout, allocate_patch_budget


def grid_points(image_size: int, stride: int) -> np.ndarray:
    """生成密集教师计算点。"""
    axis = np.arange(0, image_size, stride, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float32)


def aggregate(points: np.ndarray, scores: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """将密集点分数按 Patch 求平均。"""
    ids = layout.patch_ids(points)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    sums = np.bincount(ids, weights=values, minlength=layout.config.patch_count)
    counts = np.bincount(ids, minlength=layout.config.patch_count)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    """将非负分数归一化为 Patch 概率分布。"""
    values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0).reshape(-1)
    return values / max(float(values.sum()), 1e-12)


def calibrated_support_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """返回连续 raw geometry support 的校准误差。"""
    predicted = np.asarray(prediction, dtype=np.float32).reshape(-1)
    expected = np.asarray(target, dtype=np.float32).reshape(-1)
    if predicted.shape != expected.shape or not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("预测和 raw support 必须是同形状的有限数组")
    difference = predicted - expected
    return {
        "raw_mae": float(np.abs(difference).mean()),
        "brier": float(np.square(difference).mean()),
    }


def stats(values: np.ndarray, total_points: int | None = None) -> dict[str, float]:
    """统计熵、有效 Patch 数和高分区域质量。"""
    probability = normalize(values)
    positive = probability > 1e-12
    entropy = float(-(probability[positive] * np.log(probability[positive])).sum())
    result = {
        "active": float(np.count_nonzero(positive)),
        "entropy": entropy,
        "effective": float(np.exp(entropy)),
        "top10_mass": float(np.sort(probability)[-max(1, len(probability) // 10):].sum()),
        "top20_mass": float(np.sort(probability)[-max(1, len(probability) // 5):].sum()),
    }
    if total_points is not None:
        result["cap4"] = float(np.count_nonzero(np.asarray(values) == 4))
        result["mean_active"] = float(total_points / max(result["active"], 1.0))
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个 Patch 分布的排序相关性。"""
    a = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    b = np.argsort(np.argsort(second, kind="stable"), kind="stable")
    return float(np.corrcoef(a, b)[0, 1])


def top_iou(first: np.ndarray, second: np.ndarray, fraction: float = 0.2) -> float:
    """计算两个分布最高区域的交并比。"""
    count = max(1, int(round(first.size * fraction)))
    a = set(np.argsort(first)[-count:])
    b = set(np.argsort(second)[-count:])
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    """计算 9 组 held-out 样本的教师、模型和预算分布统计。"""
    parser = argparse.ArgumentParser(description="诊断 v0.4 Patch 分布")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-pair", type=int, default=6)
    args = parser.parse_args()

    config = dino_token_static_config()
    layout = PatchLayout(config)
    device = resolve_device(args.device)
    pairs = load_heldout_pairs(args.image_root)[:9]
    if not pairs:
        raise ValueError("没有找到 held-out 图像对")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dino = DINOv2PatchEncoder(args.dino_weights, config.image_size).to(device).eval()
    runner = load_runner(args.checkpoint, device)
    rows: list[dict[str, object]] = []

    for index, pair in enumerate(pairs, start=1):
        image_a = load_resized_rgb(pair.image_a, config.image_size)
        image_b = load_resized_rgb(pair.image_b, config.image_size)
        geometry = load_pair_geometry(pair, config.image_size)
        support_a, support_b = projected_patch_support_targets(
            geometry, layout, 2, config.geometry_depth_tolerance
        )
        # 这里严格复用 v0.4 的几何教师：投影后按 2 像素间隔聚合到 37×37 Patch。
        images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(device)
        with torch.inference_mode():
            tokens = pool_dino_tokens(dino(images), config.patch_rows, config.patch_cols)
            _, _, values_a, values_b = runner.semantic_encoder(tokens[:1], tokens[1:])
        raw_a = values_a[0].cpu().numpy().reshape(-1)
        raw_b = values_b[0].cpu().numpy().reshape(-1)
        budget_a = allocate_patch_budget(raw_a, config, 0)
        budget_b = allocate_patch_budget(raw_b, config, 0)

        for side, target, raw, budget in (("a", support_a, raw_a, budget_a), ("b", support_b, raw_b, budget_b)):
            target_probability = normalize(target)
            raw_probability = normalize(raw)
            budget_probability = normalize(budget)
            target_stats = stats(target)
            raw_stats = stats(raw)
            budget_stats = stats(budget, config.total_points)
            row = {
                "pair_index": index,
                "pair_id": pair.pair_id,
                "side": side,
                "teacher": "geometry",
                **{f"teacher_{key}": value for key, value in target_stats.items()},
                **{f"raw_{key}": value for key, value in raw_stats.items()},
                **{f"budget_{key}": value for key, value in budget_stats.items()},
                "raw_teacher_spearman": spearman(target_probability, raw_probability),
                "budget_teacher_spearman": spearman(target_probability, budget_probability),
                "budget_raw_spearman": spearman(raw_probability, budget_probability),
                "budget_teacher_top20_iou": top_iou(target_probability, budget_probability),
                "budget_raw_top20_iou": top_iou(raw_probability, budget_probability),
            }
            rows.append(row)
            if index == args.visual_pair:
                pair_output = output / f"{index:02d}_{pair.pair_id}_{side}"
                pair_output.mkdir(parents=True, exist_ok=True)
                save_patch_heatmap(pair_output / "teacher_geometry.png", target, 37, 37, f"{index:02d} {side} 教师几何目标")
                save_patch_heatmap(pair_output / "raw_patch_value.png", raw, 37, 37, f"{index:02d} {side} 模型原始 Patch 价值")
                save_patch_heatmap(pair_output / "integer_budget.png", budget, 37, 37, f"{index:02d} {side} 最终整数预算")
                save_patch_heatmap(pair_output / "raw_minus_teacher.png", raw_probability - target_probability, 37, 37, f"{index:02d} {side} 原始价值-教师分布")

        print(f"完成 {index}/{len(pairs)}：{pair.pair_id}", flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
