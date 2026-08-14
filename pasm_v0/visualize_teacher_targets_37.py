# 在 DINOv2 的 37×37 原始网格上对比两类训练真值。

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pasm_v0.backbones import FrozenLoMaDescriptor, load_resized_rgb, resolve_device
from pasm_v0.config import PasmV0Config
from pasm_v0.data import load_manifest
from pasm_v0.evaluate_v03 import save_patch_heatmap
from pasm_v0.geometry import load_pair_geometry, projected_patch_support_targets
from pasm_v0.quality_teacher import teacher_scores
from pasm_v0.sampling import PatchLayout


def grid_centers(image_size: int, rows: int, cols: int) -> np.ndarray:
    """返回规则网格中每个 Patch 的像素中心。"""
    axis_x = (np.arange(cols, dtype=np.float32) + 0.5) * image_size / cols
    axis_y = (np.arange(rows, dtype=np.float32) + 0.5) * image_size / rows
    grid_y, grid_x = np.meshgrid(axis_y, axis_x, indexing="ij")
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)


def grid_points(image_size: int, stride: int) -> np.ndarray:
    """返回按指定像素间隔排列的密集采样点。"""
    axis = np.arange(0, image_size, stride, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个 37×37 真值图的排序相关性。"""
    a = np.argsort(np.argsort(np.asarray(first).reshape(-1), kind="stable"), kind="stable")
    b = np.argsort(np.argsort(np.asarray(second).reshape(-1), kind="stable"), kind="stable")
    return float(np.corrcoef(a, b)[0, 1])


def top_overlap(first: np.ndarray, second: np.ndarray, fraction: float = 0.2) -> float:
    """计算两个真值图最高区域的交并比。"""
    count = max(1, int(round(first.size * fraction)))
    first_ids = set(np.argsort(first.reshape(-1))[-count:])
    second_ids = set(np.argsort(second.reshape(-1))[-count:])
    return len(first_ids & second_ids) / max(1, len(first_ids | second_ids))


def aggregate_dense_teacher(teacher: np.ndarray, image_size: int, rows: int, cols: int) -> np.ndarray:
    """将密集质量教师按像素位置聚合到指定 Patch 网格。"""
    values = np.asarray(teacher, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("密集质量教师必须是有限二维数组")
    height, width = values.shape
    x = (np.arange(width, dtype=np.float32) + 0.5) * image_size / width
    y = (np.arange(height, dtype=np.float32) + 0.5) * image_size / height
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    patch_x = np.minimum((grid_x / (image_size / cols)).astype(np.int64), cols - 1)
    patch_y = np.minimum((grid_y / (image_size / rows)).astype(np.int64), rows - 1)
    patch_ids = (patch_y * cols + patch_x).reshape(-1)
    sums = np.bincount(patch_ids, weights=values.reshape(-1), minlength=rows * cols)
    counts = np.bincount(patch_ids, minlength=rows * cols)
    return (sums / np.maximum(counts, 1)).reshape(rows, cols).astype(np.float32)


def aggregate_point_scores(
    points: np.ndarray, scores: np.ndarray, layout: PatchLayout
) -> np.ndarray:
    """将密集点质量按实际像素位置聚合到 Patch 网格。"""
    patch_ids = layout.patch_ids(points)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    sums = np.bincount(patch_ids, weights=values, minlength=layout.config.patch_count)
    counts = np.bincount(patch_ids, minlength=layout.config.patch_count)
    return (sums / np.maximum(counts, 1)).reshape(layout.config.patch_rows, layout.config.patch_cols).astype(np.float32)


def visualize(
    manifest: str | Path,
    loma_weights: str | Path | None,
    output: str | Path,
    max_pairs: int = 10,
    support_stride: int = 2,
    device: str = "auto",
    quality_cache: str | Path | None = None,
    quality_stride: int = 2,
) -> Path:
    """重新计算训练集前若干图像对的 37×37 两类真值。"""
    if max_pairs <= 0 or support_stride <= 0 or quality_stride <= 0:
        raise ValueError("图像对数量和几何支持步长必须为正数")
    config = PasmV0Config(
        patch_rows=37,
        patch_cols=37,
        total_points=1369,
        iterations=1,
        round_min_points=(1,),
        max_points_per_patch=4,
    )
    layout = PatchLayout(config)
    pairs = [pair for pair in load_manifest(manifest) if pair.label == 1][:max_pairs]
    if not pairs:
        raise ValueError("清单中没有正图像对")
    if quality_cache is None and loma_weights is None:
        raise ValueError("必须提供 LoMa 权重或质量教师缓存目录")
    selected_device = resolve_device(device)
    loma = FrozenLoMaDescriptor(loma_weights, device=str(selected_device)) if quality_cache is None else None
    points = grid_centers(config.image_size, config.patch_rows, config.patch_cols)
    dense_quality_points = grid_points(config.image_size, quality_stride)
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, pair in enumerate(pairs, start=1):
        image_a = load_resized_rgb(pair.image_a, config.image_size)
        image_b = load_resized_rgb(pair.image_b, config.image_size)
        geometry = load_pair_geometry(pair, config.image_size)
        support_a, support_b = projected_patch_support_targets(
            geometry, layout, support_stride, config.geometry_depth_tolerance
        )
        if quality_cache is not None:
            cache_path = Path(quality_cache) / f"{pair.pair_id}.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            with np.load(cache_path) as cached:
                quality_a = aggregate_dense_teacher(cached["teacher_a"], config.image_size, 37, 37)
                quality_b = aggregate_dense_teacher(cached["teacher_b"], config.image_size, 37, 37)
        else:
            descriptor_a = loma.describe_image(image_a)
            descriptor_b = loma.describe_image(image_b)
            quality_a = teacher_scores(
                dense_quality_points, dense_quality_points, descriptor_a, descriptor_b, geometry, "a_to_b",
                config.geometry_depth_tolerance, 0.1, 6.0, selected_device,
            )
            quality_b = teacher_scores(
                dense_quality_points, dense_quality_points, descriptor_b, descriptor_a, geometry, "b_to_a",
                config.geometry_depth_tolerance, 0.1, 6.0, selected_device,
            )
            quality_a = aggregate_point_scores(dense_quality_points, quality_a, layout)
            quality_b = aggregate_point_scores(dense_quality_points, quality_b, layout)
        pair_root = output_root / f"{index:02d}_{pair.pair_id}"
        pair_root.mkdir(parents=True, exist_ok=True)
        save_patch_heatmap(pair_root / "geometry_support_37_a.png", support_a, 37, 37, "A 几何共同可见性 37×37")
        save_patch_heatmap(pair_root / "geometry_support_37_b.png", support_b, 37, 37, "B 几何共同可见性 37×37")
        save_patch_heatmap(pair_root / "quality_teacher_37_a.png", quality_a, 37, 37, "A LoMa 质量教师 37×37")
        save_patch_heatmap(pair_root / "quality_teacher_37_b.png", quality_b, 37, 37, "B LoMa 质量教师 37×37")
        np.savez_compressed(
            pair_root / "targets_37.npz",
            geometry_support_a=support_a,
            geometry_support_b=support_b,
            quality_teacher_a=quality_a,
            quality_teacher_b=quality_b,
        )
        rows.extend((
            {"pair_id": pair.pair_id, "image": "A", "spearman": spearman(support_a, quality_a), "top20_iou": top_overlap(support_a, quality_a)},
            {"pair_id": pair.pair_id, "image": "B", "spearman": spearman(support_b, quality_b), "top20_iou": top_overlap(support_b, quality_b)},
        ))
        print(f"处理 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["pair_id", "image", "spearman", "top20_iou"])
        writer.writeheader()
        writer.writerows(rows)
    return output_root


def main() -> None:
    """解析命令行参数并生成 37×37 真值对比结果。"""
    parser = argparse.ArgumentParser(description="对比 DINO 37×37 网格上的几何真值与 LoMa 质量教师")
    parser.add_argument("--manifest", type=Path, required=True, help="训练图像对清单")
    parser.add_argument("--loma-weights", type=Path, help="LoMa-B128 权重路径；未提供缓存时必需")
    parser.add_argument("--quality-cache", type=Path, help="已有密集质量教师缓存目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--max-pairs", type=int, default=10, help="最多处理的正图像对数量")
    parser.add_argument("--support-stride", type=int, default=2, help="几何支持采样步长")
    parser.add_argument("--device", default="auto", help="计算设备，如 cuda 或 cpu")
    parser.add_argument("--quality-stride", type=int, default=2, help="重新计算 LoMa 质量教师的采样间隔")
    args = parser.parse_args()
    print(visualize(args.manifest, args.loma_weights, args.output, args.max_pairs, args.support_stride, args.device, args.quality_cache, args.quality_stride))


if __name__ == "__main__":
    main()
