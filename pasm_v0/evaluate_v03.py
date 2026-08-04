# 批量评测 PASM v0.3 的 DaD 全局基线与联合 Patch 预算，并输出几何统计。

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pasm_v0.backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
    from pasm_v0.dad import FrozenDaDDetector, select_global_dad_points, select_patch_budgeted_points
    from pasm_v0.data import PairRecord, load_manifest
    from pasm_v0.geometry import PairGeometry, load_pair_geometry, projected_patch_support_targets, visible_projected_points
    from pasm_v0.infer import load_runner, save_match_visualization, save_static_point_distribution, static_experiment_output_paths
    from pasm_v0.loma_matcher import FrozenLoMaMatcher
    from pasm_v0.quality_teacher import geometry_coverage_rate
    from pasm_v0.sampling import PatchLayout, allocate_patch_budget
else:
    from .backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
    from .dad import FrozenDaDDetector, select_global_dad_points, select_patch_budgeted_points
    from .data import PairRecord, load_manifest
    from .geometry import PairGeometry, load_pair_geometry, projected_patch_support_targets, visible_projected_points
    from .infer import load_runner, save_match_visualization, save_static_point_distribution, static_experiment_output_paths
    from .loma_matcher import FrozenLoMaMatcher
    from .quality_teacher import geometry_coverage_rate
    from .sampling import PatchLayout, allocate_patch_budget


SUMMARY_FIELDS = [
    "pair_index", "pair_id", "method", "mutual_matches", "inliers_3px", "inliers_6px",
    "inlier_rate_3px", "inlier_rate_6px", "coverage_3px", "coverage_6px",
    "high_confidence_matches", "high_confidence_inliers_3px", "high_confidence_inliers_6px",
    "high_confidence_inlier_rate_3px", "high_confidence_inlier_rate_6px",
    "budget_support_spearman_a", "budget_support_spearman_b",
    "budget_support_top26_iou_a", "budget_support_top26_iou_b",
]


@dataclass
class PairState:
    """保存一组图像在三个冻结阶段间传递的选点数据。"""

    index: int
    pair: PairRecord
    image_a: np.ndarray
    image_b: np.ndarray
    budget_a: np.ndarray | None = None
    budget_b: np.ndarray | None = None
    global_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    global_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    pasm_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    pasm_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def load_heldout_pairs(image_root: str | Path) -> list[PairRecord]:
    """读取固定九组相对文件名，并补齐 RGB、深度和相机路径。"""
    root = Path(image_root)
    source = Path(__file__).with_name("heldout_v03_pairs.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("固定留出图像对文件无有效 pairs")
    records = []
    for raw in raw_pairs:
        if not isinstance(raw, list) or len(raw) != 2 or not all(isinstance(name, str) for name in raw):
            raise ValueError("固定留出图像对必须由两个文件名组成")
        name_a, name_b = raw
        image_a = root / name_a
        image_b = root / name_b
        pair = PairRecord(
            pair_id=f"{name_a}__{name_b}", image_a=image_a, image_b=image_b, label=1,
            geometry_cache=None, depth_a=image_a.with_suffix(".exr"), depth_b=image_b.with_suffix(".exr"),
            camera_a=image_a.with_suffix(".safetensor"), camera_b=image_b.with_suffix(".safetensor"),
        )
        for path in (pair.image_a, pair.image_b, pair.depth_a, pair.depth_b, pair.camera_a, pair.camera_b):
            if path is None or not path.is_file():
                raise FileNotFoundError(path)
        records.append(pair)
    return records


def validate_heldout_against_training(heldout_pairs: Sequence[PairRecord], training_manifest: str | Path) -> None:
    """确保留出图像从未出现在训练正图像对的任一侧。"""
    training_images = {
        path.resolve()
        for pair in load_manifest(training_manifest)
        if pair.label == 1
        for path in (pair.image_a, pair.image_b)
    }
    for pair in heldout_pairs:
        if pair.image_a.resolve() in training_images or pair.image_b.resolve() in training_images:
            raise ValueError("留出评测图像出现在训练正图像对中")


def save_patch_heatmap(path: str | Path, values: np.ndarray, rows: int, cols: int, title: str) -> None:
    """将 16×16 Patch 数值保存为不叠加原图的蓝白红热力图。"""
    grid = np.asarray(values, dtype=np.float64).reshape(rows, cols)
    if not np.isfinite(grid).all():
        raise ValueError("Patch 热图数值必须有限")
    minimum, maximum = float(grid.min()), float(grid.max())
    scale = np.zeros_like(grid) if maximum == minimum else (grid - minimum) / (maximum - minimum)
    cell, header = 32, 30
    canvas = Image.new("RGB", (cols * cell, rows * cell + header), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(rows):
        for col in range(cols):
            value = float(scale[row, col])
            if value <= 0.5:
                ratio = value * 2.0
                color = tuple(int(round(low + ratio * (255 - low))) for low in (49, 54, 149))
            else:
                ratio = (value - 0.5) * 2.0
                color = tuple(int(round(255 + ratio * (high - 255))) for high in (165, 0, 38))
            x0, y0 = col * cell, header + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color)
    draw.text((6, 7), f"{title}  min={minimum:.4g} max={maximum:.4g}", fill="black")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def evaluate_pairs(
    image_root: str | Path,
    training_manifest: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    dad_weights: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device: str = "auto",
    max_pairs: int = 9,
    support_stride: int = 8,
) -> list[dict[str, object]]:
    """一次加载冻结模型，完成两种 v0.3 模式的多组几何评测。"""
    if max_pairs <= 0 or support_stride <= 0:
        raise ValueError("图像对数量和几何支持步长必须为正数")
    output_root = Path(output)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("--output 必须是评测输出文件夹")
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = load_heldout_pairs(image_root)[:max_pairs]
    validate_heldout_against_training(pairs, training_manifest)
    selected_device = resolve_device(device)
    runner = load_runner(checkpoint, selected_device)
    layout = PatchLayout(runner.config)
    print(f"评测设备：{selected_device} | 图像对：{len(pairs)}", flush=True)

    states = _prepare_pasm_budgets(pairs, runner, dino_weights, selected_device)
    _prepare_dad_points(states, runner, layout, dad_weights, selected_device)
    rows = _match_and_measure(states, runner, layout, loma_weights, dad_weights, dino_weights, output_root, selected_device, support_stride)
    _save_csv(output_root / "summary.csv", rows, SUMMARY_FIELDS)
    _save_csv(output_root / "aggregate.csv", _aggregate_rows(rows), list(_aggregate_rows(rows)[0]) if rows else ["method"])
    _save_pair_index(output_root / "pairs.csv", states)
    print(f"评测完成：{output_root}", flush=True)
    return rows


def _prepare_pasm_budgets(
    pairs: Sequence[PairRecord], runner, dino_weights: str | Path, device: torch.device
) -> list[PairState]:
    """一次加载 DINOv2，为所有图像对计算静态 PASM Patch 预算。"""
    encoder = DINOv2PatchEncoder(dino_weights, runner.config.image_size).to(device).eval()
    states = []
    for index, pair in enumerate(pairs, start=1):
        print(f"DINO/PASM 预算 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
        image_a = load_resized_rgb(pair.image_a, runner.config.image_size)
        image_b = load_resized_rgb(pair.image_b, runner.config.image_size)
        images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(device)
        with torch.inference_mode():
            tokens = pool_dino_tokens(encoder(images), runner.config.patch_rows, runner.config.patch_cols)
            _, _, values_a, values_b = runner.semantic_encoder(tokens[:1], tokens[1:])
        states.append(PairState(
            index=index, pair=pair, image_a=image_a, image_b=image_b,
            budget_a=allocate_patch_budget(values_a[0].cpu().numpy(), runner.config, round_index=2),
            budget_b=allocate_patch_budget(values_b[0].cpu().numpy(), runner.config, round_index=2),
        ))
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states


def _prepare_dad_points(states: Sequence[PairState], runner, layout: PatchLayout, dad_weights: str | Path, device: torch.device) -> None:
    """一次加载 DaD，为所有图像对产生全局与 Patch 预算内的点集。"""
    detector = FrozenDaDDetector(dad_weights, runner.config.image_size, device)
    for state in states:
        print(f"DaD 选点 {state.index}/{len(states)}：{state.pair.pair_id}", flush=True)
        detections_a = detector.detect(state.image_a)
        detections_b = detector.detect(state.image_b)
        global_a, global_scores_a = select_global_dad_points(detections_a, runner.config.image_size, runner.config.total_points)
        global_b, global_scores_b = select_global_dad_points(detections_b, runner.config.image_size, runner.config.total_points)
        state.global_a = (global_a, layout.patch_ids(global_a), global_scores_a)
        state.global_b = (global_b, layout.patch_ids(global_b), global_scores_b)
        state.pasm_a = select_patch_budgeted_points(detections_a, layout, state.budget_a)
        state.pasm_b = select_patch_budgeted_points(detections_b, layout, state.budget_b)
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _match_and_measure(
    states: Sequence[PairState], runner, layout: PatchLayout, loma_weights: str | Path, dad_weights: str | Path,
    dino_weights: str | Path, output_root: Path, device: torch.device, support_stride: int,
) -> list[dict[str, object]]:
    """一次加载完整 LoMa，匹配所有点集并保存几何统计和原始输出。"""
    matcher = FrozenLoMaMatcher(loma_weights, dad_weights, dino_weights, device)
    rows: list[dict[str, object]] = []
    for state in states:
        if state.global_a is None or state.global_b is None or state.pasm_a is None or state.pasm_b is None:
            raise RuntimeError("DaD 点集尚未准备完成")
        print(f"LoMa 匹配与几何统计 {state.index}/{len(states)}：{state.pair.pair_id}", flush=True)
        geometry = load_pair_geometry(state.pair, runner.config.image_size)
        support_a, support_b = projected_patch_support_targets(
            geometry, layout, support_stride, runner.config.geometry_depth_tolerance
        )
        pair_root = output_root / f"{state.index:02d}_{_safe_name(state.pair.pair_id)}"
        for method, selected_a, selected_b, budget_a, budget_b in (
            ("dad_global", state.global_a, state.global_b, None, None),
            ("pasm_dad", state.pasm_a, state.pasm_b, state.budget_a, state.budget_b),
        ):
            points_a, patch_ids_a, scores_a = selected_a
            points_b, patch_ids_b, scores_b = selected_b
            indices_a, indices_b, confidence = matcher.match(state.image_a, points_a, state.image_b, points_b)
            method_output = static_experiment_output_paths(pair_root / method)
            _save_mode_result(method_output, method, state.image_a, state.image_b, points_a, points_b, patch_ids_a, patch_ids_b, scores_a, scores_b, budget_a, budget_b, indices_a, indices_b, confidence, runner.config)
            row = _metric_row(state.index, state.pair.pair_id, method, points_a, points_b, indices_a, indices_b, confidence, geometry, runner.config.geometry_depth_tolerance)
            if method == "pasm_dad":
                row.update(_alignment_columns(budget_a, budget_b, support_a, support_b))
                _save_patch_values(pair_root, budget_a, budget_b, support_a, support_b)
            rows.append(row)
    return rows


def _save_mode_result(paths, mode: str, image_a: np.ndarray, image_b: np.ndarray, points_a: np.ndarray, points_b: np.ndarray, patch_ids_a: np.ndarray, patch_ids_b: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, budget_a: np.ndarray | None, budget_b: np.ndarray | None, indices_a: np.ndarray, indices_b: np.ndarray, confidence: np.ndarray, config) -> None:
    """保存单种选点模式的点、匹配和完整数值结果。"""
    matched_a, matched_b = points_a[indices_a], points_b[indices_b]
    budget_a = np.bincount(patch_ids_a, minlength=config.patch_count).astype(np.int64) if budget_a is None else budget_a
    budget_b = np.bincount(patch_ids_b, minlength=config.patch_count).astype(np.int64) if budget_b is None else budget_b
    np.savez_compressed(
        paths.result, mode=np.asarray(mode), selected_points_a=points_a, selected_points_b=points_b,
        selected_patch_ids_a=patch_ids_a, selected_patch_ids_b=patch_ids_b,
        selected_dad_scores_a=scores_a, selected_dad_scores_b=scores_b,
        patch_budget_a=budget_a, patch_budget_b=budget_b,
        keypoints_a=matched_a, keypoints_b=matched_b, confidence=confidence,
        config=np.asarray(str(asdict(config))),
    )
    save_static_point_distribution(paths.points, image_a, image_b, points_a, points_b, mode)
    save_match_visualization(paths.matches, image_a, image_b, matched_a, matched_b)
    keep = confidence >= 0.6
    save_match_visualization(paths.high_confidence_matches, image_a, image_b, matched_a[keep], matched_b[keep])


def _metric_row(index: int, pair_id: str, method: str, points_a: np.ndarray, points_b: np.ndarray, indices_a: np.ndarray, indices_b: np.ndarray, confidence: np.ndarray, geometry: PairGeometry, depth_tolerance: float) -> dict[str, object]:
    """计算一组互为最强匹配的 3px、6px、覆盖和高置信几何指标。"""
    masks = {threshold: _match_inlier_mask(points_a, points_b, indices_a, indices_b, geometry, threshold, depth_tolerance) for threshold in (3.0, 6.0)}
    high = confidence >= 0.6
    matches = len(indices_a)
    row: dict[str, object] = {"pair_index": index, "pair_id": pair_id, "method": method, "mutual_matches": matches}
    for threshold, suffix in ((3.0, "3px"), (6.0, "6px")):
        mask = masks[threshold]
        row[f"inliers_{suffix}"] = int(mask.sum())
        row[f"inlier_rate_{suffix}"] = float(mask.mean()) if matches else 0.0
        row[f"coverage_{suffix}"] = 0.5 * (
            geometry_coverage_rate(points_a, points_b, geometry, "a_to_b", threshold, depth_tolerance)
            + geometry_coverage_rate(points_b, points_a, geometry, "b_to_a", threshold, depth_tolerance)
        )
        row[f"high_confidence_inliers_{suffix}"] = int(mask[high].sum())
        row[f"high_confidence_inlier_rate_{suffix}"] = float(mask[high].mean()) if high.any() else 0.0
    row["high_confidence_matches"] = int(high.sum())
    return row


def _match_inlier_mask(points_a: np.ndarray, points_b: np.ndarray, indices_a: np.ndarray, indices_b: np.ndarray, geometry: PairGeometry, threshold_px: float, depth_tolerance: float) -> np.ndarray:
    """按 A→B 深度可见投影判断每一条 LoMa 匹配边是否为几何内点。"""
    if len(indices_a) == 0:
        return np.empty((0,), dtype=bool)
    visible_indices, projected = visible_projected_points(points_a, geometry, "a_to_b", depth_tolerance)
    expected = np.full((len(points_a), 2), np.nan, dtype=np.float32)
    expected[visible_indices] = projected
    targets = expected[indices_a]
    valid = np.isfinite(targets).all(axis=1)
    distance_sq = np.sum((targets - points_b[indices_b]) ** 2, axis=1)
    return valid & (distance_sq <= threshold_px**2)


def _alignment_columns(budget_a: np.ndarray, budget_b: np.ndarray, support_a: np.ndarray, support_b: np.ndarray) -> dict[str, float]:
    """计算两张图的预算与共同可见 Patch 支持一致性。"""
    rho_a, iou_a = _patch_alignment(budget_a, support_a)
    rho_b, iou_b = _patch_alignment(budget_b, support_b)
    return {"budget_support_spearman_a": rho_a, "budget_support_spearman_b": rho_b, "budget_support_top26_iou_a": iou_a, "budget_support_top26_iou_b": iou_b}


def _patch_alignment(budget: np.ndarray, support: np.ndarray, top_count: int = 26) -> tuple[float, float]:
    """返回预算与几何支持的 Spearman 等级相关和 Top-K 集合交并比。"""
    first = np.asarray(budget, dtype=np.float64).reshape(-1)
    second = np.asarray(support, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Patch 预算和几何支持必须是同形有限数组")
    rank_a, rank_b = _average_ranks(first), _average_ranks(second)
    rho = 0.0 if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0 else float(np.corrcoef(rank_a, rank_b)[0, 1])
    count = min(top_count, len(first))
    top_a = set(np.argsort(-first, kind="stable")[:count])
    top_b = set(np.argsort(-second, kind="stable")[:count])
    return rho, float(len(top_a & top_b) / len(top_a | top_b)) if top_a | top_b else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """为含并列值的一维数组计算平均秩。"""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _save_patch_values(pair_root: Path, budget_a: np.ndarray, budget_b: np.ndarray, support_a: np.ndarray, support_b: np.ndarray) -> None:
    """保存热图对应的原始预算与共同可见支持数组。"""
    np.savez_compressed(pair_root / "patch_values.npz", patch_budget_a=budget_a, patch_budget_b=budget_b, geometry_support_a=support_a, geometry_support_b=support_b)


def _save_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    """将评测行保存为 UTF-8 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_pair_index(path: Path, states: Sequence[PairState]) -> None:
    """保存图像对编号与输出目录映射，供热图脚本读取。"""
    rows = [{"pair_index": state.index, "pair_id": state.pair.pair_id, "directory": f"{state.index:02d}_{_safe_name(state.pair.pair_id)}"} for state in states]
    _save_csv(path, rows, ["pair_index", "pair_id", "directory"])


def _aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """按方法汇总逐对宏平均和全部边的全局内点率。"""
    result = []
    for method in ("dad_global", "pasm_dad"):
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        total_matches = sum(int(row["mutual_matches"]) for row in group)
        total_high = sum(int(row["high_confidence_matches"]) for row in group)
        result.append({
            "method": method, "pairs": len(group), "mean_mutual_matches": float(np.mean([float(row["mutual_matches"]) for row in group])),
            "mean_inlier_rate_3px": float(np.mean([float(row["inlier_rate_3px"]) for row in group])),
            "mean_inlier_rate_6px": float(np.mean([float(row["inlier_rate_6px"]) for row in group])),
            "global_inlier_rate_3px": sum(int(row["inliers_3px"]) for row in group) / total_matches if total_matches else 0.0,
            "global_inlier_rate_6px": sum(int(row["inliers_6px"]) for row in group) / total_matches if total_matches else 0.0,
            "mean_coverage_3px": float(np.mean([float(row["coverage_3px"]) for row in group])),
            "mean_coverage_6px": float(np.mean([float(row["coverage_6px"]) for row in group])),
            "mean_high_confidence_inlier_rate_3px": float(np.mean([float(row["high_confidence_inlier_rate_3px"]) for row in group])),
            "mean_high_confidence_inlier_rate_6px": float(np.mean([float(row["high_confidence_inlier_rate_6px"]) for row in group])),
            "global_high_confidence_inlier_rate_3px": sum(int(row["high_confidence_inliers_3px"]) for row in group) / total_high if total_high else 0.0,
            "global_high_confidence_inlier_rate_6px": sum(int(row["high_confidence_inliers_6px"]) for row in group) / total_high if total_high else 0.0,
        })
    return result


def _safe_name(value: str) -> str:
    """将图像对标识转换为 Windows 可用的目录名。"""
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def main() -> None:
    """解析命令行参数并运行固定九组 v0.3 留出评测。"""
    parser = argparse.ArgumentParser(description="评测 PASM v0.3 在严格留出同场景图像对上的几何匹配质量")
    parser.add_argument("--image-root", type=Path, required=True, help="0285/0 图像、深度和相机文件所在目录")
    parser.add_argument("--training-manifest", type=Path, required=True, help="训练时使用的 demo_pairs.json，用于泄漏检查")
    parser.add_argument("--dino-weights", type=Path, required=True, help="本地 DINOv2 权重")
    parser.add_argument("--loma-weights", type=Path, required=True, help="本地 LoMa-B128 权重")
    parser.add_argument("--dad-weights", type=Path, required=True, help="本地 DaD 权重")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PASM v0.2 训练检查点")
    parser.add_argument("--output", type=Path, required=True, help="本次评测输出文件夹")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--max-pairs", type=int, default=9, help="最多运行固定留出集中的前几组")
    parser.add_argument("--support-stride", type=int, default=8, help="Patch 共同可见支持的几何网格步长")
    args = parser.parse_args()
    evaluate_pairs(args.image_root, args.training_manifest, args.dino_weights, args.loma_weights, args.dad_weights, args.checkpoint, args.output, args.device, args.max_pairs, args.support_stride)


if __name__ == "__main__":
    main()
