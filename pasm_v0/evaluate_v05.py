"""固定九组留出图像对上的 PASM v0.5 稠密融合评测。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .backbones import load_resized_rgb, resolve_device
from .evaluate_v03 import _metric_row, load_heldout_pairs, validate_heldout_against_training
from .geometry import load_pair_geometry
from .infer import save_match_visualization, save_static_point_distribution, static_experiment_output_paths
from .infer_v05 import load_v05_runner
from .loma_matcher import FrozenLoMaMatcher


SUMMARY_FIELDS = [
    "pair_index", "pair_id", "method", "mutual_matches", "inliers_3px", "inliers_6px",
    "inlier_rate_3px", "inlier_rate_6px", "coverage_3px", "coverage_6px",
    "high_confidence_matches", "high_confidence_inliers_3px", "high_confidence_inliers_6px",
    "high_confidence_inlier_rate_3px", "high_confidence_inlier_rate_6px",
]

AGGREGATE_FIELDS = [
    "method", "pairs", "mean_mutual_matches", "mean_inliers_3px", "mean_inliers_6px",
    "mean_inlier_rate_3px", "mean_inlier_rate_6px", "mean_coverage_3px", "mean_coverage_6px",
    "global_inlier_rate_3px", "global_inlier_rate_6px",
]


def evaluate_pairs(
    image_root: str | Path,
    training_manifest: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    dad_weights: str | Path,
    fusion_checkpoint: str | Path,
    semantic_checkpoint: str | Path,
    output: str | Path,
    device: str = "auto",
    max_pairs: int = 9,
) -> Path:
    """不向 v0.5 选点器传递几何信息，只在选点后计算留出集指标。"""
    if max_pairs <= 0:
        raise ValueError("最多评测图像对数必须为正数")
    output_root = Path(output)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("--output 必须是评测输出文件夹")
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = load_heldout_pairs(image_root)[:max_pairs]
    validate_heldout_against_training(pairs, training_manifest)
    selected_device = resolve_device(device)
    runner = load_v05_runner(
        fusion_checkpoint, dino_weights, dad_weights, semantic_checkpoint, selected_device
    )
    matcher = FrozenLoMaMatcher(loma_weights, dad_weights, dino_weights, selected_device)
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        print(f"v0.5 选点 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
        image_a = load_resized_rgb(pair.image_a, runner.config.image_size)
        image_b = load_resized_rgb(pair.image_b, runner.config.image_size)
        points_a, points_b, scores_a, scores_b, diagnostics = runner.select(image_a, image_b)
        if diagnostics.get("used_geometry") is not False:
            raise RuntimeError("v0.5 runner 不得在推理阶段使用几何信息")
        indices_a, indices_b, confidence = matcher.match(image_a, points_a, image_b, points_b)
        geometry = load_pair_geometry(pair, runner.config.image_size)
        row = _metric_row(
            index,
            pair.pair_id,
            "pasm_v05",
            points_a,
            points_b,
            indices_a,
            indices_b,
            confidence,
            geometry,
            runner.config.geometry_depth_tolerance,
        )
        rows.append(row)
        _save_pair_outputs(
            output_root / f"{index:02d}_{_safe_name(pair.pair_id)}",
            image_a,
            image_b,
            points_a,
            points_b,
            scores_a,
            scores_b,
            indices_a,
            indices_b,
            confidence,
            diagnostics,
        )
    _save_csv(output_root / "summary.csv", rows, SUMMARY_FIELDS)
    _save_csv(output_root / "aggregate.csv", aggregate_rows(rows), AGGREGATE_FIELDS)
    return output_root


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """汇总每种 v0.5 选点方法的逐对均值和总内点率。"""
    result: list[dict[str, object]] = []
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        group = [row for row in rows if row["method"] == method]
        total_matches = sum(int(row["mutual_matches"]) for row in group)
        result.append({
            "method": method,
            "pairs": len(group),
            "mean_mutual_matches": float(np.mean([float(row["mutual_matches"]) for row in group])),
            "mean_inliers_3px": float(np.mean([float(row["inliers_3px"]) for row in group])),
            "mean_inliers_6px": float(np.mean([float(row["inliers_6px"]) for row in group])),
            "mean_inlier_rate_3px": float(np.mean([float(row["inlier_rate_3px"]) for row in group])),
            "mean_inlier_rate_6px": float(np.mean([float(row["inlier_rate_6px"]) for row in group])),
            "mean_coverage_3px": float(np.mean([float(row["coverage_3px"]) for row in group])),
            "mean_coverage_6px": float(np.mean([float(row["coverage_6px"]) for row in group])),
            "global_inlier_rate_3px": (
                sum(int(row["inliers_3px"]) for row in group) / total_matches if total_matches else 0.0
            ),
            "global_inlier_rate_6px": (
                sum(int(row["inliers_6px"]) for row in group) / total_matches if total_matches else 0.0
            ),
        })
    return result


def save_dense_score_heatmap(path: str | Path, score_map: np.ndarray, title: str) -> None:
    """保存无 Patch 网格、直接来自 v0.5 稠密响应的伪彩热图。"""
    from PIL import Image, ImageDraw

    scores = np.asarray(score_map, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("稠密分数热图必须是有限的二维非负数组")
    log_scores = np.log(np.maximum(scores, np.finfo(np.float64).tiny))
    low, high = float(log_scores.min()), float(log_scores.max())
    level = np.zeros_like(log_scores) if high == low else (log_scores - low) / (high - low)
    red = np.clip(255.0 * level, 0.0, 255.0)
    green = np.clip(255.0 * (1.0 - np.abs(level - 0.5) * 2.0), 0.0, 255.0)
    blue = np.clip(255.0 * (1.0 - level), 0.0, 255.0)
    heatmap = np.stack((red, green, blue), axis=2).astype(np.uint8)
    image = Image.fromarray(heatmap, mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 24), "white")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((4, 5), title, fill="black")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def _save_pair_outputs(
    directory: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    confidence: np.ndarray,
    diagnostics: dict[str, object],
) -> None:
    """保存单个留出图像对的点、匹配、响应图及原始数值。"""
    paths = static_experiment_output_paths(directory)
    matched_a = points_a[indices_a]
    matched_b = points_b[indices_b]
    np.savez_compressed(
        paths.result,
        mode=np.asarray("pasm_v05"),
        selected_points_a=points_a,
        selected_points_b=points_b,
        selected_scores_a=scores_a,
        selected_scores_b=scores_b,
        keypoints_a=matched_a,
        keypoints_b=matched_b,
        confidence=confidence,
        fused_score_map_a=np.asarray(diagnostics["fused_score_map_a"], dtype=np.float32),
        fused_score_map_b=np.asarray(diagnostics["fused_score_map_b"], dtype=np.float32),
        base_score_map_a=np.asarray(diagnostics["base_score_map_a"], dtype=np.float32),
        base_score_map_b=np.asarray(diagnostics["base_score_map_b"], dtype=np.float32),
    )
    save_static_point_distribution(paths.points, image_a, image_b, points_a, points_b, "pasm_v05")
    save_match_visualization(paths.matches, image_a, image_b, matched_a, matched_b)
    high_confidence = confidence >= 0.6
    save_match_visualization(
        paths.high_confidence_matches,
        image_a,
        image_b,
        matched_a[high_confidence],
        matched_b[high_confidence],
    )
    save_dense_score_heatmap(directory / "fused_scores_a.png", diagnostics["fused_score_map_a"], "v0.5 A 融合稠密响应")
    save_dense_score_heatmap(directory / "fused_scores_b.png", diagnostics["fused_score_map_b"], "v0.5 B 融合稠密响应")
    save_dense_score_heatmap(directory / "base_scores_a.png", diagnostics["base_score_map_a"], "DaD A 基础响应")
    save_dense_score_heatmap(directory / "base_scores_b.png", diagnostics["base_score_map_b"], "DaD B 基础响应")


def _save_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 PASM v0.5 无 Patch 预算的稠密融合选点")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--loma-weights", type=Path, required=True)
    parser.add_argument("--dad-weights", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=9)
    args = parser.parse_args()
    print(
        evaluate_pairs(
            args.image_root,
            args.training_manifest,
            args.dino_weights,
            args.loma_weights,
            args.dad_weights,
            args.fusion_checkpoint,
            args.semantic_checkpoint,
            args.output,
            args.device,
            args.max_pairs,
        )
    )


if __name__ == "__main__":
    main()
