"""诊断 v0.5 稠密融合热图与全局 Top-K 选点之间的空间偏置。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def nms_peak_mask(scores: np.ndarray, nms_size: int = 3) -> np.ndarray:
    """用与推理相同的方形局部极大值规则近似重建 NMS 候选。"""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or nms_size <= 0 or nms_size % 2 != 1:
        raise ValueError("分数图必须有限二维数组，NMS 尺寸必须为正奇数")
    radius = nms_size // 2
    padded = np.pad(values, radius, mode="constant", constant_values=-np.inf)
    maximum = np.full_like(values, -np.inf)
    for y_offset in range(nms_size):
        for x_offset in range(nms_size):
            maximum = np.maximum(maximum, padded[y_offset:y_offset + values.shape[0], x_offset:x_offset + values.shape[1]])
    return values == maximum


def selection_grid_counts(
    candidate_mask: np.ndarray,
    selected_points: np.ndarray,
    grid_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """统计每个空间格中的 NMS 候选数与最终 Top-K 入选数。"""
    candidates = np.asarray(candidate_mask, dtype=bool)
    points = np.asarray(selected_points, dtype=np.float64)
    if candidates.ndim != 2 or grid_size <= 0:
        raise ValueError("候选掩码必须为二维数组，网格尺寸必须为正数")
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("入选点必须为有限的 [N, 2] 数组")
    height, width = candidates.shape
    if np.any(points[:, 0] < 0.0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0.0) or np.any(points[:, 1] >= height):
        raise ValueError("入选点超出分数图范围")
    y_indices, x_indices = np.nonzero(candidates)
    candidate_grid_y = np.minimum((y_indices * grid_size) // height, grid_size - 1)
    candidate_grid_x = np.minimum((x_indices * grid_size) // width, grid_size - 1)
    selected_grid_x = np.minimum((points[:, 0] * grid_size / width).astype(np.int64), grid_size - 1)
    selected_grid_y = np.minimum((points[:, 1] * grid_size / height).astype(np.int64), grid_size - 1)
    candidate_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    selected_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    np.add.at(candidate_counts, (candidate_grid_y, candidate_grid_x), 1)
    np.add.at(selected_counts, (selected_grid_y, selected_grid_x), 1)
    return candidate_counts, selected_counts


def selection_statistics(
    fused_scores: np.ndarray,
    base_scores: np.ndarray,
    selected_points: np.ndarray,
    selected_scores: np.ndarray,
    grid_size: int = 16,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """产生单幅图的全局阈值、NMS 数量和逐格候选/入选统计。"""
    fused = _score_map(fused_scores, "融合")
    base = _score_map(base_scores, "DaD")
    if fused.shape != base.shape:
        raise ValueError("融合与 DaD 分数图尺寸必须一致")
    points = np.asarray(selected_points, dtype=np.float64)
    scores = np.asarray(selected_scores, dtype=np.float64).reshape(-1)
    if len(points) != len(scores) or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("入选点及其分数必须一一对应且非空")
    fused_peaks = nms_peak_mask(fused)
    base_peaks = nms_peak_mask(base)
    fused_candidates, selected = selection_grid_counts(fused_peaks, points, grid_size)
    base_candidates, _ = selection_grid_counts(base_peaks, points, grid_size)
    log_delta = np.log(fused.clip(np.finfo(np.float64).tiny)) - np.log(base.clip(np.finfo(np.float64).tiny))
    log_delta -= log_delta.mean()
    height, width = fused.shape
    rows: list[dict[str, object]] = []
    for row_index in range(grid_size):
        y0, y1 = (row_index * height) // grid_size, ((row_index + 1) * height) // grid_size
        for column_index in range(grid_size):
            x0, x1 = (column_index * width) // grid_size, ((column_index + 1) * width) // grid_size
            rows.append({
                "grid_row": row_index,
                "grid_col": column_index,
                "fused_nms_candidates": int(fused_candidates[row_index, column_index]),
                "base_nms_candidates": int(base_candidates[row_index, column_index]),
                "selected_points": int(selected[row_index, column_index]),
                "mean_centered_fusion_delta": float(log_delta[y0:y1, x0:x1].mean()),
            })
    summary: dict[str, object] = {
        "image_height": height,
        "image_width": width,
        "grid_size": grid_size,
        "selected_points": int(len(points)),
        "selected_score_threshold": float(scores.min()),
        "fused_nms_candidates": int(fused_peaks.sum()),
        "base_nms_candidates": int(base_peaks.sum()),
        "occupied_grid_cells": int(np.count_nonzero(selected)),
        "max_selected_in_one_grid_cell": int(selected.max()),
        "top_grid_selected_share": float(selected.max() / len(points)),
    }
    return summary, rows


def diagnose_results(input_path: str | Path, output_dir: str | Path, grid_size: int = 16) -> Path:
    """诊断单个 result.npz 或一个评测目录中的全部 v0.5 保存结果。"""
    source = Path(input_path)
    targets = [source] if source.is_file() else sorted(source.rglob("result.npz"))
    if not targets:
        raise FileNotFoundError(f"未找到 result.npz：{source}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, object]] = []
    for index, target in enumerate(targets, start=1):
        with np.load(target) as archive:
            for required in ("fused_score_map_a", "fused_score_map_b", "base_score_map_a", "base_score_map_b", "selected_points_a", "selected_points_b", "selected_scores_a", "selected_scores_b"):
                if required not in archive:
                    raise ValueError(f"{target} 不含 v0.5 诊断字段：{required}")
            pair_output = output / f"{index:02d}_{target.parent.name}"
            pair_output.mkdir(parents=True, exist_ok=True)
            for view in ("a", "b"):
                summary, rows = selection_statistics(
                    archive[f"fused_score_map_{view}"], archive[f"base_score_map_{view}"],
                    archive[f"selected_points_{view}"], archive[f"selected_scores_{view}"], grid_size,
                )
                summary.update(result=str(target), view=view)
                (pair_output / f"selection_summary_{view}.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _write_csv(pair_output / f"selection_grid_{view}.csv", rows)
                aggregate.append(summary)
    _write_csv(output / "selection_summary.csv", aggregate)
    return output


def _score_map(scores: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError(f"{name}分数图必须是有限的非负二维数组")
    return values / values.sum()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断 v0.5 融合热图与全局 Top-K 选点的空间偏置")
    parser.add_argument("--input", type=Path, required=True, help="v0.5 的 result.npz 或其评测输出目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    args = parser.parse_args()
    print(diagnose_results(args.input, args.output, args.grid_size))


if __name__ == "__main__":
    main()
