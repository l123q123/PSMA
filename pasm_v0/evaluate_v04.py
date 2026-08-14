# 对比 PASM v0.4 几何真值和质量教师真值的静态 Patch 预算效果。

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
from .config import dino_token_static_config
from .dad import FrozenDaDDetector, select_patch_budgeted_points
from .evaluate_v03 import (
    SUMMARY_FIELDS as V03_SUMMARY_FIELDS,
    _alignment_columns,
    _metric_row,
    load_heldout_pairs,
    save_patch_heatmap,
    validate_heldout_against_training,
)
from .geometry import load_pair_geometry, projected_patch_support_targets
from .infer import load_runner, save_match_visualization, save_static_point_distribution, static_experiment_output_paths
from .loma_matcher import FrozenLoMaMatcher
from .sampling import PatchLayout, allocate_patch_budget


SUMMARY_FIELDS = [
    "pair_index", "pair_id", "target_kind", "mutual_matches", "inliers_3px", "inliers_6px",
    "inlier_rate_3px", "inlier_rate_6px", "coverage_3px", "coverage_6px",
    "high_confidence_matches", "high_confidence_inliers_3px", "high_confidence_inliers_6px",
    "high_confidence_inlier_rate_3px", "high_confidence_inlier_rate_6px",
    "active_patches_a", "active_patches_b", "cap_patches_a", "cap_patches_b",
    "budget_entropy_a", "budget_entropy_b",
    "budget_support_spearman_a", "budget_support_spearman_b",
    "budget_support_top26_iou_a", "budget_support_top26_iou_b",
]


def budget_distribution(
    budget: np.ndarray, max_points_per_patch: int = 4
) -> dict[str, int | float]:
    """统计 Patch 预算的活跃数量、熵和上限命中数量。"""
    values = np.asarray(budget, dtype=np.int64).reshape(-1)
    if max_points_per_patch <= 0 or values.size == 0 or np.any(values < 0) or np.any(values > max_points_per_patch):
        raise ValueError("Patch 预算必须是非负且不超过上限的一维数组")
    total = int(values.sum())
    if total == 0:
        entropy = 0.0
    else:
        probabilities = values[values > 0].astype(np.float64) / total
        entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        "active_patches": int(np.count_nonzero(values)),
        "cap_patches": int(np.count_nonzero(values == max_points_per_patch)),
        "budget_entropy": entropy,
    }


def budget_distribution_for_config(budget: np.ndarray, config) -> dict[str, int | float]:
    """使用当前静态配置的 Patch 上限统计预算。"""
    return budget_distribution(budget, max_points_per_patch=config.max_points_per_patch)


def evaluate_ablation(
    image_root: str | Path,
    training_manifest: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    dad_weights: str | Path,
    checkpoint_geometry: str | Path,
    checkpoint_quality: str | Path,
    output: str | Path,
    device: str = "auto",
    max_pairs: int = 9,
    support_stride: int = 2,
    compare_geometry: bool = False,
    patch_grid: int = 37,
) -> Path:
    """在同一批留出图像对上统一评测两个 Patch 监督模型。"""
    if max_pairs <= 0 or support_stride <= 0:
        raise ValueError("图像对数量和几何支持步长必须为正数")
    config = dino_token_static_config(patch_grid)
    selected_device = resolve_device(device)
    pairs = load_heldout_pairs(image_root)[:max_pairs]
    validate_heldout_against_training(pairs, training_manifest)
    specs = (
        ("geometry_deep", checkpoint_geometry, "geometry"),
        ("geometry_single", checkpoint_quality, "geometry"),
    ) if compare_geometry else (
        ("geometry", checkpoint_geometry, "geometry"),
        ("quality", checkpoint_quality, "quality"),
    )
    runners = {
        label: _load_v04_runner(checkpoint, expected_kind, selected_device)
        for label, checkpoint, expected_kind in specs
    }
    for target_kind, runner in runners.items():
        if runner.config != config:
            raise ValueError(f"{target_kind} 检查点配置不是 v0.4 {patch_grid}×{patch_grid} 静态配置")
    layout = PatchLayout(config)
    output_root = Path(output)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("评测输出必须是文件夹")
    output_root.mkdir(parents=True, exist_ok=True)
    dino = DINOv2PatchEncoder(dino_weights, config.image_size).to(selected_device).eval()
    detector = FrozenDaDDetector(dad_weights, config.image_size, selected_device)
    matcher = FrozenLoMaMatcher(loma_weights, dad_weights, dino_weights, selected_device)
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        print(f"准备评测 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
        image_a = load_resized_rgb(pair.image_a, config.image_size)
        image_b = load_resized_rgb(pair.image_b, config.image_size)
        geometry = load_pair_geometry(pair, config.image_size)
        detections_a = detector.detect(image_a)
        detections_b = detector.detect(image_b)
        images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(selected_device)
        with torch.inference_mode():
            tokens = pool_dino_tokens(dino(images), config.patch_rows, config.patch_cols)
        support_a, support_b = projected_patch_support_targets(
            geometry, layout, support_stride, config.geometry_depth_tolerance
        )
        for target_kind, runner in runners.items():
            with torch.inference_mode():
                _, _, values_a, values_b = runner.semantic_encoder(tokens[:1], tokens[1:])
            budget_a = allocate_patch_budget(values_a[0].cpu().numpy(), config, round_index=0)
            budget_b = allocate_patch_budget(values_b[0].cpu().numpy(), config, round_index=0)
            points_a, patch_ids_a, dad_scores_a = select_patch_budgeted_points(
                detections_a, layout, budget_a
            )
            points_b, patch_ids_b, dad_scores_b = select_patch_budgeted_points(
                detections_b, layout, budget_b
            )
            indices_a, indices_b, confidence = matcher.match(
                image_a, points_a, image_b, points_b
            )
            pair_root = output_root / target_kind / f"{index:02d}_{_safe_name(pair.pair_id)}"
            paths = static_experiment_output_paths(pair_root)
            matched_a = points_a[indices_a]
            matched_b = points_b[indices_b]
            np.savez_compressed(
                paths.result,
                mode=np.asarray("pasm_v04"),
                target_kind=np.asarray(target_kind),
                selected_points_a=points_a,
                selected_points_b=points_b,
                selected_patch_ids_a=patch_ids_a,
                selected_patch_ids_b=patch_ids_b,
                selected_dad_scores_a=dad_scores_a,
                selected_dad_scores_b=dad_scores_b,
                patch_budget_a=budget_a,
                patch_budget_b=budget_b,
                keypoints_a=matched_a,
                keypoints_b=matched_b,
                confidence=confidence,
            )
            save_static_point_distribution(
                paths.points, image_a, image_b, points_a, points_b, f"pasm_v04_{target_kind}"
            )
            save_match_visualization(paths.matches, image_a, image_b, matched_a, matched_b)
            keep = confidence >= 0.6
            save_match_visualization(
                paths.high_confidence_matches,
                image_a,
                image_b,
                matched_a[keep],
                matched_b[keep],
            )
            save_patch_heatmap(
                pair_root / "budget_a.png", budget_a, config.patch_rows, config.patch_cols,
                f"{target_kind} A Patch 预算",
            )
            save_patch_heatmap(
                pair_root / "budget_b.png", budget_b, config.patch_rows, config.patch_cols,
                f"{target_kind} B Patch 预算",
            )
            row = _metric_row(
                index,
                pair.pair_id,
                "pasm_v04",
                points_a,
                points_b,
                indices_a,
                indices_b,
                confidence,
                geometry,
                config.geometry_depth_tolerance,
            )
            row["target_kind"] = target_kind
            row.update(_alignment_columns(budget_a, budget_b, support_a, support_b))
            row.update({f"{key}_a": value for key, value in budget_distribution_for_config(budget_a, config).items()})
            row.update({f"{key}_b": value for key, value in budget_distribution_for_config(budget_b, config).items()})
            rows.append(row)
        print(f"完成评测 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
    _save_csv(output_root / "summary.csv", rows, SUMMARY_FIELDS)
    _save_csv(output_root / "aggregate.csv", aggregate_rows(rows), ["target_kind", "pairs", "mean_mutual_matches", "mean_inliers_3px", "mean_inliers_6px", "mean_inlier_rate_3px", "mean_inlier_rate_6px", "mean_active_patches_a", "mean_active_patches_b", "mean_budget_entropy_a", "mean_budget_entropy_b"])
    return output_root


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """按监督类型汇总评测结果的主要指标。"""
    result = []
    for target_kind in dict.fromkeys(str(row["target_kind"]) for row in rows):
        group = [row for row in rows if row.get("target_kind") == target_kind]
        if not group:
            continue
        result.append({
            "target_kind": target_kind,
            "pairs": len(group),
            "mean_mutual_matches": float(np.mean([row["mutual_matches"] for row in group])),
            "mean_inliers_3px": float(np.mean([row["inliers_3px"] for row in group])),
            "mean_inliers_6px": float(np.mean([row["inliers_6px"] for row in group])),
            "mean_inlier_rate_3px": float(np.mean([row["inlier_rate_3px"] for row in group])),
            "mean_inlier_rate_6px": float(np.mean([row["inlier_rate_6px"] for row in group])),
            "mean_active_patches_a": float(np.mean([row["active_patches_a"] for row in group])),
            "mean_active_patches_b": float(np.mean([row["active_patches_b"] for row in group])),
            "mean_budget_entropy_a": float(np.mean([row["budget_entropy_a"] for row in group])),
            "mean_budget_entropy_b": float(np.mean([row["budget_entropy_b"] for row in group])),
        })
    return result


def _load_v04_runner(checkpoint: str | Path, target_kind: str, device: torch.device):
    """加载并验证指定监督类型的 v0.4 语义编码器检查点。"""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not accepted_checkpoint_target_kind(
        str(payload.get("target_kind")), target_kind
    ):
        raise ValueError(f"检查点不是标记为 {target_kind} 的 v0.4 模型")
    runner = load_runner(checkpoint, device)
    runner.semantic_encoder.eval()
    return runner


def accepted_checkpoint_target_kind(actual_kind: str, expected_kind: str) -> bool:
    """判断 checkpoint 监督身份是否可作为指定评测分支的几何模型。"""
    if expected_kind == "geometry":
        return actual_kind in {"geometry", "geometry_calibrated"}
    return actual_kind == expected_kind


def _safe_name(value: str) -> str:
    """将图像对编号转换成安全的输出目录名。"""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def _save_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    """将评测行保存为 UTF-8 CSV 文件。"""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """解析命令行参数并执行 v0.4 A/B held-out 评测。"""
    parser = argparse.ArgumentParser(description="评测 PASM v0.4 的几何真值和质量教师模型")
    parser.add_argument("--image-root", type=Path, required=True, help="MegaDepth extracted 图像根目录")
    parser.add_argument("--training-manifest", type=Path, required=True, help="训练清单，用于检查留出图像独立性")
    parser.add_argument("--dino-weights", type=Path, required=True, help="DINOv2 权重路径")
    parser.add_argument("--loma-weights", type=Path, required=True, help="LoMa-B128 权重路径")
    parser.add_argument("--dad-weights", type=Path, required=True, help="DaD 权重路径")
    parser.add_argument("--checkpoint-geometry", type=Path, required=True, help="几何真值模型 checkpoint")
    parser.add_argument("--checkpoint-quality", type=Path, required=True, help="质量教师模型 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="A/B 评测输出文件夹")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--max-pairs", type=int, default=9, help="最多评测的留出图像对数量")
    parser.add_argument("--support-stride", type=int, default=2, help="几何支持统计采样步长")
    parser.add_argument("--compare-geometry", action="store_true", help="将两个 checkpoint 作为 geometry 模型对比")
    parser.add_argument("--patch-grid", type=int, default=37, choices=(16, 37), help="Patch 网格边长")
    args = parser.parse_args()
    print(
        evaluate_ablation(
            args.image_root,
            args.training_manifest,
            args.dino_weights,
            args.loma_weights,
            args.dad_weights,
            args.checkpoint_geometry,
            args.checkpoint_quality,
            args.output,
            args.device,
            args.max_pairs,
            args.support_stride,
            args.compare_geometry,
            args.patch_grid,
        )
    )


if __name__ == "__main__":
    main()
