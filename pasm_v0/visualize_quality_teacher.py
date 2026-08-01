# 生成软质量教师与均匀随机选点的 MegaDepth 可视化和快速量化对照。

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pasm_v0.backbones import FrozenLoMaDescriptor, load_resized_rgb, resolve_device
from pasm_v0.config import PasmV0Config
from pasm_v0.data import load_manifest
from pasm_v0.geometry import load_pair_geometry
from pasm_v0.quality_teacher import (
    cosine_mutual_matches,
    geometry_coverage_rate,
    mutual_geometry_inlier_rate,
    patch_candidate_points,
    sampled_descriptors,
    select_patch_teacher_points,
    teacher_scores,
)
from pasm_v0.sampling import PatchLayout


SUMMARY_FIELDS = [
    "pair_id",
    "random_coverage_3px",
    "random_coverage_6px",
    "random_mutual_matches",
    "random_inlier_3px",
    "random_inlier_6px",
    "teacher_coverage_3px",
    "teacher_coverage_6px",
    "teacher_mutual_matches",
    "teacher_inlier_3px",
    "teacher_inlier_6px",
]


def build_argument_parser() -> argparse.ArgumentParser:
    """创建软质量教师可视化命令的参数解析器。"""
    parser = argparse.ArgumentParser(description="比较随机点与软质量教师点的 LoMa 匹配质量")
    parser.add_argument("--manifest", required=True, help="带深度和相机路径的 MegaDepth 图像对清单")
    parser.add_argument("--loma-weights", required=True, help="LoMa-B128 权重路径")
    parser.add_argument("--output", required=True, help="十组图片和 summary.csv 的输出目录")
    parser.add_argument("--max-pairs", type=int, default=10, help="最多处理的正图像对数量")
    parser.add_argument("--device", default="auto", help="LoMa 和教师分数计算设备，如 cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=0, help="随机基线的固定随机种子")
    parser.add_argument("--temperature", type=float, default=0.1, help="教师分数 sigmoid 的温度")
    parser.add_argument("--exclusion-radius", type=float, default=6.0, help="真实投影附近不作为错误候选的半径")
    parser.add_argument("--nms-radius", type=float, default=4.0, help="Patch 内教师点的非极大值抑制半径")
    parser.add_argument("--display-matches", type=int, default=100, help="每张匹配图最多绘制的互为最近邻边数")
    return parser


def save_point_comparison(
    path: str | Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    scores_a: Sequence[float] | None = None,
    scores_b: Sequence[float] | None = None,
) -> None:
    """保存左右图拼接后的两图点分布图。"""
    canvas, width = _side_by_side_image(image_a, image_b)
    first = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    first_scores = _optional_scores(scores_a, len(first))
    second_scores = _optional_scores(scores_b, len(second))
    draw = ImageDraw.Draw(canvas)
    for index, point in enumerate(first):
        _draw_point(draw, point, _point_color(None if first_scores is None else first_scores[index]))
    for index, point in enumerate(second):
        _draw_point(draw, point + np.array([width, 0.0]), _point_color(None if second_scores is None else second_scores[index]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_match_comparison(
    path: str | Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    similarities: Sequence[float],
    limit: int = 100,
) -> None:
    """保存左右图拼接后的高相似度互为最近邻连线图。"""
    first = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    target = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(similarities, dtype=np.float32).reshape(-1)
    if len(source) != len(target) or len(source) != len(values) or limit <= 0:
        raise ValueError("匹配索引、相似度或显示边数无效")
    if np.any(source < 0) or np.any(source >= len(first)) or np.any(target < 0) or np.any(target >= len(second)):
        raise ValueError("匹配索引超出点集范围")
    canvas, width = _side_by_side_image(image_a, image_b)
    draw = ImageDraw.Draw(canvas)
    order = np.argsort(-values, kind="stable")[:limit]
    for edge in order:
        point_a = first[source[edge]]
        point_b = second[target[edge]] + np.array([width, 0.0])
        draw.line((tuple(point_a), tuple(point_b)), fill=(255, 170, 0), width=1)
        _draw_point(draw, point_a, (0, 230, 255))
        _draw_point(draw, point_b, (0, 230, 255))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_summary(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    """将每个图像对的随机与教师指标保存为固定列顺序的 CSV。"""
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """运行十对 MegaDepth 的软质量教师可视化与量化对照。"""
    arguments = build_argument_parser().parse_args()
    if arguments.max_pairs <= 0 or arguments.display_matches <= 0:
        raise ValueError("图像对数量和显示边数必须为正数")
    config = PasmV0Config()
    layout = PatchLayout(config)
    output_root = Path(arguments.output)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(arguments.device)
    pairs = [pair for pair in load_manifest(arguments.manifest) if pair.label == 1]
    selected_pairs = pairs[: arguments.max_pairs]
    if not selected_pairs:
        raise ValueError("清单中没有可用的正图像对")
    loma = FrozenLoMaDescriptor(arguments.loma_weights, device=str(device))
    candidates, candidate_patch_ids = patch_candidate_points(layout)
    generator = np.random.default_rng(arguments.seed)
    rows = []
    print(f"软质量教师设备：{device}")
    for index, pair in enumerate(selected_pairs, start=1):
        print(f"处理图像对 {index}/{len(selected_pairs)}：{pair.pair_id}")
        image_a = load_resized_rgb(pair.image_a, config.image_size)
        image_b = load_resized_rgb(pair.image_b, config.image_size)
        geometry = load_pair_geometry(pair, config.image_size)
        descriptor_a = loma.describe_image(image_a)
        descriptor_b = loma.describe_image(image_b)
        random_a, _ = layout.initial_points(seed=int(generator.integers(2**31)))
        random_b, _ = layout.initial_points(seed=int(generator.integers(2**31)))
        scores_a = teacher_scores(
            candidates,
            candidates,
            descriptor_a,
            descriptor_b,
            geometry,
            "a_to_b",
            config.geometry_depth_tolerance,
            arguments.temperature,
            arguments.exclusion_radius,
            device,
        )
        scores_b = teacher_scores(
            candidates,
            candidates,
            descriptor_b,
            descriptor_a,
            geometry,
            "b_to_a",
            config.geometry_depth_tolerance,
            arguments.temperature,
            arguments.exclusion_radius,
            device,
        )
        teacher_a, _, selected_scores_a = select_patch_teacher_points(
            candidates, candidate_patch_ids, scores_a, layout, points_per_patch=4, nms_radius=arguments.nms_radius
        )
        teacher_b, _, selected_scores_b = select_patch_teacher_points(
            candidates, candidate_patch_ids, scores_b, layout, points_per_patch=4, nms_radius=arguments.nms_radius
        )
        pair_name = pair.pair_id.replace("/", "_").replace("\\", "_")
        pair_output = output_root / f"{index:02d}_{pair_name}"
        row: dict[str, object] = {"pair_id": pair.pair_id}
        for name, points_a, points_b, point_scores_a, point_scores_b in (
            ("random", random_a, random_b, None, None),
            ("teacher", teacher_a, teacher_b, selected_scores_a, selected_scores_b),
        ):
            descriptors_a = sampled_descriptors(descriptor_a, points_a, config.image_size, device)
            descriptors_b = sampled_descriptors(descriptor_b, points_b, config.image_size, device)
            source, target, similarities = cosine_mutual_matches(descriptors_a, descriptors_b)
            coverage_3 = 0.5 * (
                geometry_coverage_rate(points_a, points_b, geometry, "a_to_b", 3.0, config.geometry_depth_tolerance)
                + geometry_coverage_rate(points_b, points_a, geometry, "b_to_a", 3.0, config.geometry_depth_tolerance)
            )
            coverage_6 = 0.5 * (
                geometry_coverage_rate(points_a, points_b, geometry, "a_to_b", 6.0, config.geometry_depth_tolerance)
                + geometry_coverage_rate(points_b, points_a, geometry, "b_to_a", 6.0, config.geometry_depth_tolerance)
            )
            row.update({
                f"{name}_coverage_3px": coverage_3,
                f"{name}_coverage_6px": coverage_6,
                f"{name}_mutual_matches": len(source),
                f"{name}_inlier_3px": mutual_geometry_inlier_rate(
                    points_a, points_b, source, target, geometry, 3.0, config.geometry_depth_tolerance
                ),
                f"{name}_inlier_6px": mutual_geometry_inlier_rate(
                    points_a, points_b, source, target, geometry, 6.0, config.geometry_depth_tolerance
                ),
            })
            save_point_comparison(
                pair_output / f"{name}_points.png",
                image_a,
                image_b,
                points_a,
                points_b,
                point_scores_a,
                point_scores_b,
            )
            save_match_comparison(
                pair_output / f"{name}_matches.png",
                image_a,
                image_b,
                points_a,
                points_b,
                source,
                target,
                similarities,
                arguments.display_matches,
            )
        rows.append(row)
    save_summary(output_root / "summary.csv", rows)
    print(f"已保存 {len(rows)} 个图像对到：{output_root}")


def _side_by_side_image(image_a: np.ndarray, image_b: np.ndarray) -> tuple[Image.Image, int]:
    """将两张 CHW 图像转为等高左右拼接的 Pillow 画布。"""
    first = _pillow_image(image_a)
    second = _pillow_image(image_b)
    if first.height != second.height:
        raise ValueError("两张图必须具有相同高度")
    canvas = Image.new("RGB", (first.width + second.width, first.height))
    canvas.paste(first, (0, 0))
    canvas.paste(second, (first.width, 0))
    return canvas, first.width


def _pillow_image(image: np.ndarray) -> Image.Image:
    """将归一化 CHW RGB 数组转换为可绘图的 Pillow 图像。"""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError("图像必须是 [3, H, W] 数组")
    pixels = np.clip(array.transpose(1, 2, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _optional_scores(scores: Sequence[float] | None, point_count: int) -> np.ndarray | None:
    """验证可选点分数长度并转换为浮点数组。"""
    if scores is None:
        return None
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(values) != point_count:
        raise ValueError("点分数数量必须与点数一致")
    return values


def _draw_point(draw: ImageDraw.ImageDraw, point: np.ndarray, color: tuple[int, int, int]) -> None:
    """在指定坐标绘制一个小圆形特征点。"""
    x, y = float(point[0]), float(point[1])
    draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), fill=color, outline=(0, 0, 0))


def _point_color(score: float | None) -> tuple[int, int, int]:
    """将教师分数映射为低分黄、高分红的可视化颜色。"""
    if score is None:
        return 0, 230, 255
    value = float(np.clip(score, 0.0, 1.0))
    return 255, int(round(220.0 * (1.0 - value))), 0


if __name__ == "__main__":
    main()
