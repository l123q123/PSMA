# 为 v0.3 批量评测结果绘制 PASM Patch 预算与共同可见几何真值热力图。

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pasm_v0.config import PasmV0Config
    from pasm_v0.evaluate_v03 import load_heldout_pairs, save_patch_heatmap
    from pasm_v0.geometry import load_pair_geometry, projected_patch_support_targets
    from pasm_v0.sampling import PatchLayout
else:
    from .config import PasmV0Config
    from .evaluate_v03 import load_heldout_pairs, save_patch_heatmap
    from .geometry import load_pair_geometry, projected_patch_support_targets
    from .sampling import PatchLayout


def visualize_patch_values(image_root: str | Path, evaluation_output: str | Path, max_pairs: int = 9, support_stride: int = 8) -> None:
    """读取评测输出，为每组图像保存预算和共同可见几何支持热图。"""
    if max_pairs <= 0 or support_stride <= 0:
        raise ValueError("图像对数量和几何支持步长必须为正数")
    root = Path(evaluation_output)
    pairs = {pair.pair_id: pair for pair in load_heldout_pairs(image_root)}
    with (root / "pairs.csv").open("r", encoding="utf-8", newline="") as file:
        index_rows = list(csv.DictReader(file))[:max_pairs]
    if not index_rows:
        raise ValueError("评测输出中没有 pairs.csv 图像对索引")
    config = PasmV0Config()
    layout = PatchLayout(config)
    for row in index_rows:
        pair = pairs.get(row["pair_id"])
        if pair is None:
            raise ValueError(f"固定留出集未找到图像对：{row['pair_id']}")
        pair_root = root / row["directory"]
        result_path = pair_root / "pasm_dad" / "result.npz"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        with np.load(result_path) as result:
            budget_a = np.asarray(result["patch_budget_a"], dtype=np.float32)
            budget_b = np.asarray(result["patch_budget_b"], dtype=np.float32)
        if budget_a.shape != (config.patch_count,) or budget_b.shape != (config.patch_count,):
            raise ValueError("PASM Patch 预算形状不正确")
        geometry = load_pair_geometry(pair, config.image_size)
        support_a, support_b = projected_patch_support_targets(geometry, layout, support_stride, config.geometry_depth_tolerance)
        save_patch_heatmap(pair_root / "budget_heatmap_a.png", budget_a, config.patch_rows, config.patch_cols, "A 图 PASM Patch 预算")
        save_patch_heatmap(pair_root / "budget_heatmap_b.png", budget_b, config.patch_rows, config.patch_cols, "B 图 PASM Patch 预算")
        save_patch_heatmap(pair_root / "geometry_support_heatmap_a.png", support_a, config.patch_rows, config.patch_cols, "A 图共同可见几何支持")
        save_patch_heatmap(pair_root / "geometry_support_heatmap_b.png", support_b, config.patch_rows, config.patch_cols, "B 图共同可见几何支持")
        np.savez_compressed(pair_root / "patch_values.npz", patch_budget_a=budget_a, patch_budget_b=budget_b, geometry_support_a=support_a, geometry_support_b=support_b)
        print(f"Patch 热图 {row['pair_index']}/{len(index_rows)}：{row['pair_id']}", flush=True)


def main() -> None:
    """解析命令行参数并绘制已有评测结果的 Patch 热力图。"""
    parser = argparse.ArgumentParser(description="可视化 v0.3 评测中每张图的 Patch 预算与共同可见几何真值")
    parser.add_argument("--image-root", type=Path, required=True, help="0285/0 图像、深度和相机文件所在目录")
    parser.add_argument("--evaluation-output", type=Path, required=True, help="evaluate_v03 的输出文件夹")
    parser.add_argument("--max-pairs", type=int, default=9, help="最多处理 pairs.csv 中的前几组")
    parser.add_argument("--support-stride", type=int, default=8, help="Patch 共同可见支持的几何网格步长")
    args = parser.parse_args()
    visualize_patch_values(args.image_root, args.evaluation_output, args.max_pairs, args.support_stride)


if __name__ == "__main__":
    main()
