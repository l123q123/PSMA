# 对比 Patch 价值真值与 LoMa 局部质量教师真值。

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pasm_v0.config import PasmV0Config
from pasm_v0.data import load_manifest
from pasm_v0.evaluate_v03 import save_patch_heatmap
from pasm_v0.geometry import load_pair_geometry, projected_patch_support_targets
from pasm_v0.sampling import PatchLayout


def minmax(values: np.ndarray) -> np.ndarray:
    """将二维分数图按自身范围缩放到 0 到 1。"""
    scores = np.asarray(values, dtype=np.float32)
    low, high = float(scores.min()), float(scores.max())
    if high <= low:
        return np.full(scores.shape, 0.5, dtype=np.float32)
    return (scores - low) / (high - low)


def save_heatmap(path: Path, values: np.ndarray, size: int = 512) -> None:
    """将分数图按统一蓝白红颜色保存为纯热力图。"""
    scores = minmax(values)
    colors = np.full(scores.shape + (3,), 255.0, dtype=np.float32)
    blue = np.array([49.0, 54.0, 149.0], dtype=np.float32)
    red = np.array([165.0, 0.0, 38.0], dtype=np.float32)
    low = scores <= 0.5
    high = scores > 0.5
    colors[low] = 255.0 + (scores[low] * 2.0)[:, None] * (blue - 255.0)
    colors[high] = 255.0 + ((scores[high] - 0.5) * 2.0)[:, None] * (red - 255.0)
    image = Image.fromarray(np.rint(np.clip(colors, 0, 255)).astype(np.uint8), mode="RGB")
    image.resize((size, size), Image.Resampling.NEAREST).save(path)


def patch_reduce(quality: np.ndarray, rows: int, cols: int, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """将密集质量教师按 Patch 计算均值和最大值。"""
    expected = (rows * cells, cols * cells)
    values = np.asarray(quality, dtype=np.float32)
    if values.shape != expected:
        raise ValueError(f"质量教师尺寸应为 {expected}，实际为 {values.shape}")
    blocks = values.reshape(rows, cells, cols, cells)
    return blocks.mean(axis=(1, 3)), blocks.max(axis=(1, 3))


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个 Patch 分数排序的 Spearman 相关系数。"""
    a = np.asarray(first).reshape(-1)
    b = np.asarray(second).reshape(-1)
    rank_a = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    rank_b = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def top_overlap(first: np.ndarray, second: np.ndarray, fraction: float = 0.2) -> float:
    """计算两个 Patch 真值最高区域的交并比。"""
    count = max(1, int(round(first.size * fraction)))
    first_ids = set(np.argsort(first.reshape(-1))[-count:])
    second_ids = set(np.argsort(second.reshape(-1))[-count:])
    return len(first_ids & second_ids) / max(1, len(first_ids | second_ids))


def compare_targets(
    manifest: str | Path,
    quality_cache: str | Path,
    output: str | Path,
    max_pairs: int = 10,
    support_stride: int = 4,
) -> Path:
    """生成训练图上的两类真值热图和相关性汇总。"""
    if max_pairs <= 0 or support_stride <= 0:
        raise ValueError("图像对数量和几何支持步长必须为正数")
    config = PasmV0Config()
    layout = PatchLayout(config)
    pairs = [pair for pair in load_manifest(manifest) if pair.label == 1][:max_pairs]
    if not pairs:
        raise ValueError("清单中没有正图像对")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, pair in enumerate(pairs, start=1):
        cache_path = Path(quality_cache) / f"{pair.pair_id}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path) as cached:
            teacher_a = np.asarray(cached["teacher_a"], dtype=np.float32)
            teacher_b = np.asarray(cached["teacher_b"], dtype=np.float32)
            cells = int(cached["quality_cells_per_axis"].item())
        geometry = load_pair_geometry(pair, config.image_size)
        support_a, support_b = projected_patch_support_targets(
            geometry, layout, support_stride, config.geometry_depth_tolerance
        )
        mean_a, max_a = patch_reduce(teacher_a, config.patch_rows, config.patch_cols, cells)
        mean_b, max_b = patch_reduce(teacher_b, config.patch_rows, config.patch_cols, cells)
        pair_root = root / f"{index:02d}_{pair.pair_id}"
        pair_root.mkdir(parents=True, exist_ok=True)
        save_patch_heatmap(pair_root / "geometry_support_a.png", support_a, config.patch_rows, config.patch_cols, "A 几何共同可见支持")
        save_patch_heatmap(pair_root / "geometry_support_b.png", support_b, config.patch_rows, config.patch_cols, "B 几何共同可见支持")
        save_heatmap(pair_root / "quality_teacher_dense_a.png", teacher_a)
        save_heatmap(pair_root / "quality_teacher_dense_b.png", teacher_b)
        save_patch_heatmap(pair_root / "quality_teacher_mean_a.png", mean_a, config.patch_rows, config.patch_cols, "A LoMa 教师 Patch 均值")
        save_patch_heatmap(pair_root / "quality_teacher_mean_b.png", mean_b, config.patch_rows, config.patch_cols, "B LoMa 教师 Patch 均值")
        save_patch_heatmap(pair_root / "quality_teacher_max_a.png", max_a, config.patch_rows, config.patch_cols, "A LoMa 教师 Patch 最大值")
        save_patch_heatmap(pair_root / "quality_teacher_max_b.png", max_b, config.patch_rows, config.patch_cols, "B LoMa 教师 Patch 最大值")
        rows.extend((
            {"pair_id": pair.pair_id, "image": "A", "support_teacher_mean_spearman": spearman(support_a, mean_a), "support_teacher_max_spearman": spearman(support_a, max_a), "top20_iou_mean": top_overlap(support_a, mean_a), "top20_iou_max": top_overlap(support_a, max_a)},
            {"pair_id": pair.pair_id, "image": "B", "support_teacher_mean_spearman": spearman(support_b, mean_b), "support_teacher_max_spearman": spearman(support_b, max_b), "top20_iou_mean": top_overlap(support_b, mean_b), "top20_iou_max": top_overlap(support_b, max_b)},
        ))
        print(f"处理 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def main() -> None:
    """解析命令行参数并生成两类真值对比图。"""
    parser = argparse.ArgumentParser(description="对比 Patch 价值几何真值与 LoMa 质量教师真值")
    parser.add_argument("--manifest", type=Path, required=True, help="训练图像对清单")
    parser.add_argument("--quality-cache", type=Path, required=True, help="v0.2 质量教师缓存目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--max-pairs", type=int, default=10, help="最多处理的正图像对数量")
    parser.add_argument("--support-stride", type=int, default=4, help="几何支持采样步长")
    args = parser.parse_args()
    print(compare_targets(args.manifest, args.quality_cache, args.output, args.max_pairs, args.support_stride))


if __name__ == "__main__":
    main()
