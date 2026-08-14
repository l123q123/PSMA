# PASM v0.4 的 Patch 监督目标生成与缓存读取。

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .backbones import FrozenLoMaDescriptor, load_resized_rgb, resolve_device
from .config import PasmV0Config, dino_token_static_config
from .data import load_manifest
from .geometry import (
    load_pair_geometry,
    projected_patch_support_fractions,
    projected_patch_support_targets,
)
from .quality_teacher import teacher_scores
from .sampling import PatchLayout


TargetKind = Literal["geometry", "quality", "geometry_calibrated"]

TARGET_STRIDE = 2


def grid_points(image_size: int, stride: int = TARGET_STRIDE) -> np.ndarray:
    """返回覆盖整张正方形图像的规则像素采样点。"""
    if image_size <= 0 or stride <= 0:
        raise ValueError("图像尺寸和采样步长必须为正数")
    axis = np.arange(0, image_size, stride, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)


def normalize_patch_target(values: np.ndarray, patch_count: int) -> np.ndarray:
    """将非负 Patch 权重转换为严格归一化的概率分布。"""
    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if patch_count <= 0 or target.size != patch_count or not np.isfinite(target).all():
        raise ValueError("Patch 目标必须是指定长度的有限数组")
    if np.any(target < 0.0):
        raise ValueError("Patch 目标不能为负数")
    total = float(target.sum())
    if total <= 0.0:
        raise ValueError("Patch 目标必须包含至少一个正支持区域")
    return (target / total).astype(np.float32)


def aggregate_point_scores(points: np.ndarray, scores: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """将规则像素点的质量分数按所属 Patch 取平均。"""
    coordinates = np.asarray(points, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,) or len(coordinates) != len(values):
        raise ValueError("点坐标和质量分数的形状必须一致")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("质量分数必须是有限非负数")
    patch_ids = layout.patch_ids(coordinates)
    sums = np.bincount(patch_ids, weights=values, minlength=layout.config.patch_count)
    counts = np.bincount(patch_ids, minlength=layout.config.patch_count)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def load_patch_targets(
    cache_directory: str | Path,
    pair_id: str,
    config: PasmV0Config,
    target_kind: TargetKind,
) -> tuple[np.ndarray, np.ndarray]:
    """读取并严格验证一对图像对应的两张 Patch 价值监督图。"""
    _validate_target_kind(target_kind)
    if target_kind == "geometry_calibrated":
        raise ValueError("geometry_calibrated 必须使用 load_calibrated_patch_targets 读取")
    path = Path(cache_directory) / f"{pair_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        _validate_metadata(payload, pair_id, config, target_kind)
        target_a = _validated_probability_target(payload["target_a"], config.patch_count)
        target_b = _validated_probability_target(payload["target_b"], config.patch_count)
    return target_a, target_b


def load_calibrated_patch_targets(
    cache_directory: str | Path,
    pair_id: str,
    config: PasmV0Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 raw geometry support 与其归一化诊断分布。"""
    path = Path(cache_directory) / f"{pair_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        _validate_calibrated_metadata(payload, pair_id, config)
        raw_a = _validated_raw_target(payload["raw_target_a"], config.patch_count)
        raw_b = _validated_raw_target(payload["raw_target_b"], config.patch_count)
        distribution_a = _validated_probability_target(payload["distribution_target_a"], config.patch_count)
        distribution_b = _validated_probability_target(payload["distribution_target_b"], config.patch_count)
    return raw_a, raw_b, distribution_a, distribution_b


def cache_targets(
    manifest_path: str | Path,
    output_directory: str | Path,
    target_kind: TargetKind,
    loma_weights: str | Path | None = None,
    max_pairs: int | None = None,
    device: str = "auto",
    patch_grid: int = 37,
) -> Path:
    """为训练清单中的正图像对离线生成指定网格的目标缓存。"""
    _validate_target_kind(target_kind)
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("最多处理图像对数必须为正数")
    if target_kind == "quality" and loma_weights is None:
        raise ValueError("生成质量教师缓存必须提供 LoMa 权重")
    config = dino_token_static_config(patch_grid)
    layout = PatchLayout(config)
    pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("清单中没有正图像对")
    output = Path(output_directory)
    if output.exists() and not output.is_dir():
        raise ValueError("目标缓存输出必须是文件夹")
    output.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    descriptor = (
        FrozenLoMaDescriptor(loma_weights, device=str(selected_device))
        if target_kind == "quality"
        else None
    )
    dense_points = grid_points(config.image_size) if target_kind == "quality" else None
    for index, pair in enumerate(pairs, start=1):
        path = output / f"{pair.pair_id}.npz"
        if path.is_file():
            if target_kind == "geometry_calibrated":
                load_calibrated_patch_targets(output, pair.pair_id, config)
            else:
                load_patch_targets(output, pair.pair_id, config, target_kind)
            print(f"跳过 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
            continue
        geometry = load_pair_geometry(pair, config.image_size)
        if target_kind == "geometry":
            target_a, target_b = projected_patch_support_targets(
                geometry, layout, TARGET_STRIDE, config.geometry_depth_tolerance
            )
        elif target_kind == "geometry_calibrated":
            raw_a, raw_b = projected_patch_support_fractions(
                geometry, layout, TARGET_STRIDE, config.geometry_depth_tolerance
            )
            _save_calibrated_patch_targets(path, pair.pair_id, config, raw_a, raw_b)
            print(f"生成 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
            continue
        else:
            if descriptor is None or dense_points is None:
                raise RuntimeError("质量教师描述子未初始化")
            image_a = load_resized_rgb(pair.image_a, config.image_size)
            image_b = load_resized_rgb(pair.image_b, config.image_size)
            descriptor_a = descriptor.describe_image(image_a)
            descriptor_b = descriptor.describe_image(image_b)
            scores_a = teacher_scores(
                dense_points,
                dense_points,
                descriptor_a,
                descriptor_b,
                geometry,
                "a_to_b",
                config.geometry_depth_tolerance,
                temperature=0.1,
                exclusion_radius=6.0,
                device=selected_device,
            )
            scores_b = teacher_scores(
                dense_points,
                dense_points,
                descriptor_b,
                descriptor_a,
                geometry,
                "b_to_a",
                config.geometry_depth_tolerance,
                temperature=0.1,
                exclusion_radius=6.0,
                device=selected_device,
            )
            target_a = aggregate_point_scores(dense_points, scores_a, layout)
            target_b = aggregate_point_scores(dense_points, scores_b, layout)
        _save_patch_targets(path, pair.pair_id, config, target_kind, target_a, target_b)
        print(f"生成 {index}/{len(pairs)}：{pair.pair_id}", flush=True)
    return output


def _save_patch_targets(
    path: Path,
    pair_id: str,
    config: PasmV0Config,
    target_kind: TargetKind,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> None:
    """将双图 Patch 概率目标与验证元数据写入压缩缓存。"""
    np.savez_compressed(
        path,
        target_a=normalize_patch_target(target_a, config.patch_count),
        target_b=normalize_patch_target(target_b, config.patch_count),
        target_kind=np.asarray(target_kind),
        pair_id=np.asarray(pair_id),
        image_size=np.asarray(config.image_size),
        patch_rows=np.asarray(config.patch_rows),
        patch_cols=np.asarray(config.patch_cols),
        stride=np.asarray(TARGET_STRIDE),
    )


def _save_calibrated_patch_targets(
    path: Path,
    pair_id: str,
    config: PasmV0Config,
    raw_target_a: np.ndarray,
    raw_target_b: np.ndarray,
) -> None:
    """写入连续 geometry support 及其独立的归一化诊断分布。"""
    raw_a = _validated_raw_target(raw_target_a, config.patch_count)
    raw_b = _validated_raw_target(raw_target_b, config.patch_count)
    np.savez_compressed(
        path,
        raw_target_a=raw_a,
        raw_target_b=raw_b,
        distribution_target_a=normalize_patch_target(raw_a, config.patch_count),
        distribution_target_b=normalize_patch_target(raw_b, config.patch_count),
        target_kind=np.asarray("geometry_calibrated"),
        target_schema=np.asarray(2),
        pair_id=np.asarray(pair_id),
        image_size=np.asarray(config.image_size),
        patch_rows=np.asarray(config.patch_rows),
        patch_cols=np.asarray(config.patch_cols),
        stride=np.asarray(TARGET_STRIDE),
    )


def _validate_target_kind(target_kind: str) -> None:
    """验证目标类型属于本次 A/B 消融的两个端点。"""
    if target_kind not in {"geometry", "quality", "geometry_calibrated"}:
        raise ValueError("目标类型必须是 geometry、quality 或 geometry_calibrated")


def _validated_probability_target(values: np.ndarray, patch_count: int) -> np.ndarray:
    """验证缓存中的 Patch 目标已是合法概率分布且不作隐式修正。"""
    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if patch_count <= 0 or target.size != patch_count or not np.isfinite(target).all():
        raise ValueError("缓存中的 Patch 目标必须是指定长度的有限数组")
    if np.any(target < 0.0) or not np.isclose(float(target.sum()), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("缓存中的 Patch 目标必须是和为一的非负概率分布")
    return target


def _validated_raw_target(values: np.ndarray, patch_count: int) -> np.ndarray:
    """验证逐 Patch 的连续 geometry support。"""
    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if patch_count <= 0 or target.size != patch_count or not np.isfinite(target).all():
        raise ValueError("缓存中的原始 Patch 支持必须是指定长度的有限数组")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("缓存中的原始 Patch 支持必须位于 [0, 1]")
    if not np.any(target > 0.0):
        raise ValueError("缓存中的原始 Patch 支持必须包含至少一个可见区域")
    return target


def _validate_metadata(payload: np.lib.npyio.NpzFile, pair_id: str, config: PasmV0Config, target_kind: TargetKind) -> None:
    """检查缓存身份、网格和监督类型与当前实验完全一致。"""
    required = {"target_a", "target_b", "target_kind", "pair_id", "image_size", "patch_rows", "patch_cols", "stride"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"Patch 目标缓存缺少字段：{sorted(missing)}")
    metadata = {
        "target_kind": str(payload["target_kind"].item()),
        "pair_id": str(payload["pair_id"].item()),
        "image_size": int(payload["image_size"].item()),
        "patch_rows": int(payload["patch_rows"].item()),
        "patch_cols": int(payload["patch_cols"].item()),
        "stride": int(payload["stride"].item()),
    }
    if metadata["target_kind"] != target_kind or metadata["pair_id"] != pair_id:
        raise ValueError("Patch 目标缓存的图像对或监督类型不匹配")
    if (
        metadata["image_size"] != config.image_size
        or metadata["patch_rows"] != config.patch_rows
        or metadata["patch_cols"] != config.patch_cols
        or metadata["stride"] != TARGET_STRIDE
    ):
        raise ValueError("Patch 目标缓存的网格或采样步长不匹配")


def _validate_calibrated_metadata(payload: np.lib.npyio.NpzFile, pair_id: str, config: PasmV0Config) -> None:
    """检查 calibrated cache 的身份、网格及 schema。"""
    required = {
        "raw_target_a", "raw_target_b", "distribution_target_a", "distribution_target_b",
        "target_kind", "target_schema", "pair_id", "image_size", "patch_rows", "patch_cols", "stride",
    }
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"校准 Patch 目标缓存缺少字段：{sorted(missing)}")
    if str(payload["target_kind"].item()) != "geometry_calibrated" or int(payload["target_schema"].item()) != 2:
        raise ValueError("校准 Patch 目标缓存类型或 schema 不匹配")
    _validate_metadata_fields(payload, pair_id, config)


def _validate_metadata_fields(payload: np.lib.npyio.NpzFile, pair_id: str, config: PasmV0Config) -> None:
    """校验两种缓存共用的图像对和网格元数据。"""
    if str(payload["pair_id"].item()) != pair_id:
        raise ValueError("Patch 目标缓存的图像对不匹配")
    if (
        int(payload["image_size"].item()) != config.image_size
        or int(payload["patch_rows"].item()) != config.patch_rows
        or int(payload["patch_cols"].item()) != config.patch_cols
        or int(payload["stride"].item()) != TARGET_STRIDE
    ):
        raise ValueError("Patch 目标缓存的网格或采样步长不匹配")
