# PASM v0.1 的 MegaDepth 几何读取、直接投影和监督目标。

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from .data import PairRecord
from .sampling import PatchLayout


@dataclass(frozen=True)
class PairGeometry:
    """保存缩放到模型坐标系的双图深度、内参与位姿。"""

    depth_a: np.ndarray
    depth_b: np.ndarray
    intrinsics_a: np.ndarray
    intrinsics_b: np.ndarray
    cam2world_a: np.ndarray
    cam2world_b: np.ndarray
    original_size_a: tuple[int, int]
    original_size_b: tuple[int, int]
    image_size: int

    def __post_init__(self) -> None:
        """验证几何数组、原图尺寸和模型输入尺寸。"""
        depth_a = _depth_array(self.depth_a, "A 图深度")
        depth_b = _depth_array(self.depth_b, "B 图深度")
        size_a = _image_size(self.original_size_a, "A 图原始尺寸")
        size_b = _image_size(self.original_size_b, "B 图原始尺寸")
        if depth_a.shape != (size_a[1], size_a[0]) or depth_b.shape != (size_b[1], size_b[0]):
            raise ValueError("深度图尺寸必须与对应 RGB 图像一致")
        if self.image_size <= 0:
            raise ValueError("模型输入尺寸必须为正数")
        object.__setattr__(self, "depth_a", depth_a)
        object.__setattr__(self, "depth_b", depth_b)
        object.__setattr__(self, "intrinsics_a", _matrix(self.intrinsics_a, (3, 3), "A 图内参"))
        object.__setattr__(self, "intrinsics_b", _matrix(self.intrinsics_b, (3, 3), "B 图内参"))
        object.__setattr__(self, "cam2world_a", _matrix(self.cam2world_a, (4, 4), "A 图位姿"))
        object.__setattr__(self, "cam2world_b", _matrix(self.cam2world_b, (4, 4), "B 图位姿"))
        object.__setattr__(self, "original_size_a", size_a)
        object.__setattr__(self, "original_size_b", size_b)


def load_pair_geometry(pair: PairRecord, image_size: int) -> PairGeometry:
    """读取一条正图像对的深度和相机信息并缩放内参。"""
    if image_size <= 0:
        raise ValueError("模型输入尺寸必须为正数")
    required = (pair.depth_a, pair.depth_b, pair.camera_a, pair.camera_b)
    if any(path is None for path in required):
        raise ValueError(f"图像对 {pair.pair_id} 缺少深度或相机文件")
    with Image.open(pair.image_a) as image_a, Image.open(pair.image_b) as image_b:
        size_a = image_a.size
        size_b = image_b.size
    cam2world_a, intrinsics_a = read_camera_metadata(pair.camera_a)
    cam2world_b, intrinsics_b = read_camera_metadata(pair.camera_b)
    return PairGeometry(
        depth_a=read_exr_depth(pair.depth_a),
        depth_b=read_exr_depth(pair.depth_b),
        intrinsics_a=_scale_intrinsics(intrinsics_a, size_a, image_size),
        intrinsics_b=_scale_intrinsics(intrinsics_b, size_b, image_size),
        cam2world_a=cam2world_a,
        cam2world_b=cam2world_b,
        original_size_a=size_a,
        original_size_b=size_b,
        image_size=image_size,
    )


def visible_projected_points(
    points: np.ndarray,
    geometry: PairGeometry,
    direction: Literal["a_to_b", "b_to_a"],
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回指定方向通过深度可见性检查的源点索引和目标投影坐标。"""
    source = _points(points)
    if depth_tolerance < 0.0:
        raise ValueError("深度误差阈值不能为负数")
    if direction == "a_to_b":
        return _project_visible_points(
            source,
            geometry.depth_a,
            geometry.intrinsics_a,
            geometry.cam2world_a,
            geometry.depth_b,
            geometry.intrinsics_b,
            geometry.cam2world_b,
            geometry.original_size_a,
            geometry.original_size_b,
            geometry.image_size,
            depth_tolerance,
        )
    if direction == "b_to_a":
        return _project_visible_points(
            source,
            geometry.depth_b,
            geometry.intrinsics_b,
            geometry.cam2world_b,
            geometry.depth_a,
            geometry.intrinsics_a,
            geometry.cam2world_a,
            geometry.original_size_b,
            geometry.original_size_a,
            geometry.image_size,
            depth_tolerance,
        )
    raise ValueError("投影方向必须为 a_to_b 或 b_to_a")


def confirmed_invisible_projected_points(
    points: np.ndarray,
    geometry: PairGeometry,
    direction: Literal["a_to_b", "b_to_a"],
    depth_tolerance: float,
) -> np.ndarray:
    """标记有明确几何证据不可见的源点，无法判定的点保留为 False。"""
    source = _points(points)
    if depth_tolerance < 0.0:
        raise ValueError("深度误差阈值不能为负数")
    if direction == "a_to_b":
        return _confirmed_invisible_points(
            source,
            geometry.depth_a,
            geometry.intrinsics_a,
            geometry.cam2world_a,
            geometry.depth_b,
            geometry.intrinsics_b,
            geometry.cam2world_b,
            geometry.original_size_a,
            geometry.original_size_b,
            geometry.image_size,
            depth_tolerance,
        )
    if direction == "b_to_a":
        return _confirmed_invisible_points(
            source,
            geometry.depth_b,
            geometry.intrinsics_b,
            geometry.cam2world_b,
            geometry.depth_a,
            geometry.intrinsics_a,
            geometry.cam2world_a,
            geometry.original_size_b,
            geometry.original_size_a,
            geometry.image_size,
            depth_tolerance,
        )
    raise ValueError("投影方向必须为 a_to_b 或 b_to_a")


def projected_match_targets(
    points_a: np.ndarray,
    points_b: np.ndarray,
    geometry: PairGeometry,
    threshold_px: float,
    depth_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用双向可见投影和互相最近邻生成当前点级匹配标签。"""
    source_a = _points(points_a)
    source_b = _points(points_b)
    if threshold_px <= 0.0 or depth_tolerance < 0.0:
        raise ValueError("匹配阈值和深度误差阈值无效")
    candidate_a = _projected_nearest_targets(
        source_a,
        source_b,
        geometry.depth_a,
        geometry.intrinsics_a,
        geometry.cam2world_a,
        geometry.depth_b,
        geometry.intrinsics_b,
        geometry.cam2world_b,
        geometry.original_size_a,
        geometry.original_size_b,
        geometry.image_size,
        threshold_px,
        depth_tolerance,
    )
    candidate_b = _projected_nearest_targets(
        source_b,
        source_a,
        geometry.depth_b,
        geometry.intrinsics_b,
        geometry.cam2world_b,
        geometry.depth_a,
        geometry.intrinsics_a,
        geometry.cam2world_a,
        geometry.original_size_b,
        geometry.original_size_a,
        geometry.image_size,
        threshold_px,
        depth_tolerance,
    )
    targets_a = np.full(len(source_a), -1, dtype=np.int64)
    targets_b = np.full(len(source_b), -1, dtype=np.int64)
    for index_a, index_b in enumerate(candidate_a):
        if index_b >= 0 and candidate_b[index_b] == index_a:
            targets_a[index_a] = index_b
            targets_b[index_b] = index_a
    return (
        torch.from_numpy(targets_a),
        torch.from_numpy(targets_a >= 0),
        torch.from_numpy(targets_b),
        torch.from_numpy(targets_b >= 0),
    )


def projected_patch_support_targets(
    geometry: PairGeometry,
    layout: PatchLayout,
    stride: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """用固定网格的双向可见投影生成两图 Patch 支持分布。"""
    if stride <= 0 or depth_tolerance < 0.0 or layout.config.image_size != geometry.image_size:
        raise ValueError("Patch 网格、投影步长或深度误差阈值无效")
    axis = np.arange(0, geometry.image_size, stride, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    visible_a = _visible_projection_indices(
        points,
        geometry.depth_a,
        geometry.intrinsics_a,
        geometry.cam2world_a,
        geometry.depth_b,
        geometry.intrinsics_b,
        geometry.cam2world_b,
        geometry.original_size_a,
        geometry.original_size_b,
        geometry.image_size,
        depth_tolerance,
    )
    visible_b = _visible_projection_indices(
        points,
        geometry.depth_b,
        geometry.intrinsics_b,
        geometry.cam2world_b,
        geometry.depth_a,
        geometry.intrinsics_a,
        geometry.cam2world_a,
        geometry.original_size_b,
        geometry.original_size_a,
        geometry.image_size,
        depth_tolerance,
    )
    return _patch_support(points[visible_a], layout), _patch_support(points[visible_b], layout)


def projected_patch_support_fractions(
    geometry: PairGeometry,
    layout: PatchLayout,
    stride: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回每个 Patch 内通过双图几何可见性检查的采样比例。"""
    if stride <= 0 or depth_tolerance < 0.0 or layout.config.image_size != geometry.image_size:
        raise ValueError("Patch 网格、投影步长或深度误差阈值无效")
    axis = np.arange(0, geometry.image_size, stride, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    visible_a = _visible_projection_indices(
        points,
        geometry.depth_a,
        geometry.intrinsics_a,
        geometry.cam2world_a,
        geometry.depth_b,
        geometry.intrinsics_b,
        geometry.cam2world_b,
        geometry.original_size_a,
        geometry.original_size_b,
        geometry.image_size,
        depth_tolerance,
    )
    visible_b = _visible_projection_indices(
        points,
        geometry.depth_b,
        geometry.intrinsics_b,
        geometry.cam2world_b,
        geometry.depth_a,
        geometry.intrinsics_a,
        geometry.cam2world_a,
        geometry.original_size_b,
        geometry.original_size_a,
        geometry.image_size,
        depth_tolerance,
    )
    samples_per_patch = _patch_support_counts(points, layout)
    support_a = _patch_support_counts(points[visible_a], layout)
    support_b = _patch_support_counts(points[visible_b], layout)
    return (
        (support_a / np.maximum(samples_per_patch, 1.0)).astype(np.float32),
        (support_b / np.maximum(samples_per_patch, 1.0)).astype(np.float32),
    )


def read_camera_metadata(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """从 MegaDepth Safetensors 文件读取相机外参与内参。"""
    camera_path = Path(path)
    payload = camera_path.read_bytes()
    if len(payload) < 8:
        raise ValueError(f"相机文件过短：{camera_path}")
    header_length = struct.unpack("<Q", payload[:8])[0]
    header_end = 8 + header_length
    if header_end > len(payload):
        raise ValueError(f"相机文件头超出范围：{camera_path}")
    try:
        metadata = json.loads(payload[8:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"相机文件头不是有效 JSON：{camera_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"相机文件头格式无效：{camera_path}")
    data = payload[header_end:]
    return (
        _camera_tensor(metadata, "cam2world", (4, 4), data, camera_path),
        _camera_tensor(metadata, "intrinsics", (3, 3), data, camera_path),
    )


def read_exr_depth(path: str | Path) -> np.ndarray:
    """读取 MegaDepth EXR 文件中的 Y 或 R 深度通道。"""
    import Imath
    import OpenEXR

    depth_path = Path(path)
    exr_file = OpenEXR.InputFile(str(depth_path))
    try:
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        channels = header["channels"]
        channel = "Y" if "Y" in channels else "R" if "R" in channels else None
        if channel is None:
            raise ValueError(f"深度图不含 Y 或 R 通道：{depth_path}")
        raw = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(raw, dtype=np.float32).copy()
        if depth.size != width * height:
            raise ValueError(f"深度图通道尺寸无效：{depth_path}")
        return depth.reshape(height, width)
    finally:
        exr_file.close()


def _projected_nearest_targets(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    source_cam2world: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_cam2world: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    image_size: int,
    threshold_px: float,
    depth_tolerance: float,
) -> np.ndarray:
    """将源点投影后映射为目标当前点集中的阈值最近邻。"""
    targets = np.full(len(source_points), -1, dtype=np.int64)
    if len(source_points) == 0 or len(target_points) == 0:
        return targets
    source_indices, projected = _project_visible_points(
        source_points,
        source_depth,
        source_intrinsics,
        source_cam2world,
        target_depth,
        target_intrinsics,
        target_cam2world,
        source_size,
        target_size,
        image_size,
        depth_tolerance,
    )
    if len(source_indices) == 0:
        return targets
    distances = np.sum((projected[:, None, :] - target_points[None, :, :]) ** 2, axis=2)
    nearest = distances.argmin(axis=1)
    valid = distances[np.arange(len(nearest)), nearest] <= threshold_px**2
    targets[source_indices[valid]] = nearest[valid]
    return targets


def _visible_projection_indices(
    points: np.ndarray,
    source_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    source_cam2world: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_cam2world: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    image_size: int,
    depth_tolerance: float,
) -> np.ndarray:
    """返回通过目标深度可见性检查的源点索引。"""
    indices, _ = _project_visible_points(
        points,
        source_depth,
        source_intrinsics,
        source_cam2world,
        target_depth,
        target_intrinsics,
        target_cam2world,
        source_size,
        target_size,
        image_size,
        depth_tolerance,
    )
    return indices


def _project_visible_points(
    points: np.ndarray,
    source_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    source_cam2world: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_cam2world: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    image_size: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """投影源点并过滤无效、出界和被遮挡的位置。"""
    source_indices = np.arange(len(points), dtype=np.int64)
    in_bounds = _in_model_bounds(points, image_size)
    source_indices = source_indices[in_bounds]
    source_pixels = points[in_bounds]
    if len(source_pixels) == 0:
        return source_indices, np.empty((0, 2), dtype=np.float32)
    source_values = _nearest_depth(source_depth, source_pixels, source_size, image_size)
    valid_depth = np.isfinite(source_values) & (source_values > 0.0)
    source_indices = source_indices[valid_depth]
    source_pixels = source_pixels[valid_depth]
    source_values = source_values[valid_depth]
    if len(source_pixels) == 0:
        return source_indices, np.empty((0, 2), dtype=np.float32)
    homogeneous = np.column_stack((source_pixels, np.ones(len(source_pixels), dtype=np.float64)))
    source_xyz = (np.linalg.inv(source_intrinsics) @ homogeneous.T).T * source_values[:, None]
    target_from_source = np.linalg.inv(target_cam2world) @ source_cam2world
    target_xyz = source_xyz @ target_from_source[:3, :3].T + target_from_source[:3, 3]
    projected_depth = target_xyz[:, 2]
    valid_camera = np.isfinite(target_xyz).all(axis=1) & (projected_depth > 0.0)
    source_indices = source_indices[valid_camera]
    target_xyz = target_xyz[valid_camera]
    projected_depth = projected_depth[valid_camera]
    if len(target_xyz) == 0:
        return source_indices, np.empty((0, 2), dtype=np.float32)
    projected = (target_intrinsics @ target_xyz.T).T
    valid_projection = np.isfinite(projected).all(axis=1) & (projected[:, 2] != 0.0)
    source_indices = source_indices[valid_projection]
    projected_depth = projected_depth[valid_projection]
    projected = projected[valid_projection]
    target_pixels = projected[:, :2] / projected[:, 2:3]
    in_target = _in_model_bounds(target_pixels, image_size)
    source_indices = source_indices[in_target]
    projected_depth = projected_depth[in_target]
    target_pixels = target_pixels[in_target]
    if len(target_pixels) == 0:
        return source_indices, np.empty((0, 2), dtype=np.float32)
    observed_depth = _bilinear_depth(target_depth, target_pixels, target_size, image_size)
    visible = np.isfinite(observed_depth) & (observed_depth > 0.0)
    visible &= np.abs(projected_depth - observed_depth) <= depth_tolerance * np.maximum(observed_depth, 1e-6)
    return source_indices[visible], target_pixels[visible].astype(np.float32)


def _confirmed_invisible_points(
    points: np.ndarray,
    source_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    source_cam2world: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_cam2world: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    image_size: int,
    depth_tolerance: float,
) -> np.ndarray:
    """按投影失败原因区分明确不可见和无法判定的点。"""
    confirmed = np.zeros(len(points), dtype=bool)
    source_indices = np.arange(len(points), dtype=np.int64)
    in_bounds = _in_model_bounds(points, image_size)
    source_indices = source_indices[in_bounds]
    source_pixels = points[in_bounds]
    if len(source_pixels) == 0:
        return confirmed
    source_values = _nearest_depth(source_depth, source_pixels, source_size, image_size)
    valid_depth = np.isfinite(source_values) & (source_values > 0.0)
    source_indices = source_indices[valid_depth]
    source_pixels = source_pixels[valid_depth]
    source_values = source_values[valid_depth]
    if len(source_pixels) == 0:
        return confirmed
    homogeneous = np.column_stack((source_pixels, np.ones(len(source_pixels), dtype=np.float64)))
    source_xyz = (np.linalg.inv(source_intrinsics) @ homogeneous.T).T * source_values[:, None]
    target_from_source = np.linalg.inv(target_cam2world) @ source_cam2world
    target_xyz = source_xyz @ target_from_source[:3, :3].T + target_from_source[:3, 3]
    projected_depth = target_xyz[:, 2]
    valid_camera = np.isfinite(target_xyz).all(axis=1) & (projected_depth > 0.0)
    confirmed[source_indices[~valid_camera]] = True
    source_indices = source_indices[valid_camera]
    target_xyz = target_xyz[valid_camera]
    projected_depth = projected_depth[valid_camera]
    if len(target_xyz) == 0:
        return confirmed
    projected = (target_intrinsics @ target_xyz.T).T
    valid_projection = np.isfinite(projected).all(axis=1) & (projected[:, 2] != 0.0)
    confirmed[source_indices[~valid_projection]] = True
    source_indices = source_indices[valid_projection]
    projected_depth = projected_depth[valid_projection]
    projected = projected[valid_projection]
    if len(projected) == 0:
        return confirmed
    target_pixels = projected[:, :2] / projected[:, 2:3]
    in_target = _in_model_bounds(target_pixels, image_size)
    confirmed[source_indices[~in_target]] = True
    source_indices = source_indices[in_target]
    projected_depth = projected_depth[in_target]
    target_pixels = target_pixels[in_target]
    if len(target_pixels) == 0:
        return confirmed
    observed_depth = _bilinear_depth(target_depth, target_pixels, target_size, image_size)
    known_depth = np.isfinite(observed_depth) & (observed_depth > 0.0)
    mismatch = known_depth & (
        np.abs(projected_depth - observed_depth)
        > depth_tolerance * np.maximum(observed_depth, 1e-6)
    )
    confirmed[source_indices[mismatch]] = True
    return confirmed


def _patch_support(points: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """将可见源点的空间覆盖度转换为归一化 Patch 分布。"""
    support = _patch_support_counts(points, layout)
    if not np.any(support):
        return support
    return support / support.sum()


def _patch_support_counts(points: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """统计规则点在各 Patch 内的数量，不做全图归一化。"""
    support = np.zeros(layout.config.patch_count, dtype=np.float32)
    if len(points) == 0:
        return support
    patch_ids = layout.patch_ids(points)
    support += np.bincount(patch_ids, minlength=layout.config.patch_count).astype(np.float32)
    return support


def _nearest_depth(
    depth: np.ndarray, points: np.ndarray, original_size: tuple[int, int], image_size: int
) -> np.ndarray:
    """在原始深度图中读取模型坐标对应的最近像素深度。"""
    coordinates = _depth_coordinates(points, original_size, image_size)
    x = np.rint(coordinates[:, 0]).astype(np.int64)
    y = np.rint(coordinates[:, 1]).astype(np.int64)
    x = np.clip(x, 0, depth.shape[1] - 1)
    y = np.clip(y, 0, depth.shape[0] - 1)
    return depth[y, x].astype(np.float64, copy=False)


def _bilinear_depth(
    depth: np.ndarray, points: np.ndarray, original_size: tuple[int, int], image_size: int
) -> np.ndarray:
    """在原始深度图中双线性读取模型坐标对应的深度。"""
    coordinates = _depth_coordinates(points, original_size, image_size)
    x = np.clip(coordinates[:, 0], 0.0, depth.shape[1] - 1.0)
    y = np.clip(coordinates[:, 1], 0.0, depth.shape[0] - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, depth.shape[1] - 1)
    y1 = np.minimum(y0 + 1, depth.shape[0] - 1)
    dx = x - x0
    dy = y - y0
    return (
        (1.0 - dx) * (1.0 - dy) * depth[y0, x0]
        + dx * (1.0 - dy) * depth[y0, x1]
        + (1.0 - dx) * dy * depth[y1, x0]
        + dx * dy * depth[y1, x1]
    )


def _depth_coordinates(points: np.ndarray, original_size: tuple[int, int], image_size: int) -> np.ndarray:
    """将模型输入坐标转换为原始深度图坐标。"""
    width, height = original_size
    scale = np.array((width / image_size, height / image_size), dtype=np.float64)
    return points * scale


def _scale_intrinsics(intrinsics: np.ndarray, original_size: tuple[int, int], image_size: int) -> np.ndarray:
    """按 RGB 缩放比例将原始内参转换到模型输入坐标。"""
    width, height = _image_size(original_size, "原始图像尺寸")
    scaled = _matrix(intrinsics, (3, 3), "相机内参").copy()
    scaled[0, :] *= image_size / width
    scaled[1, :] *= image_size / height
    return scaled


def _camera_tensor(
    metadata: dict[str, Any], name: str, shape: tuple[int, int], payload: bytes, path: Path
) -> np.ndarray:
    """从 Safetensors 元数据读取指定的浮点矩阵。"""
    descriptor = metadata.get(name)
    if not isinstance(descriptor, dict) or descriptor.get("dtype") != "F64":
        raise ValueError(f"相机文件字段无效：{path} 的 {name}")
    if tuple(descriptor.get("shape", ())) != shape:
        raise ValueError(f"相机文件矩阵形状无效：{path} 的 {name}")
    offsets = descriptor.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(value, int) for value in offsets):
        raise ValueError(f"相机文件偏移无效：{path} 的 {name}")
    start, end = offsets
    expected = int(np.prod(shape)) * np.dtype(np.float64).itemsize
    if start < 0 or end - start != expected or end > len(payload):
        raise ValueError(f"相机文件数据范围无效：{path} 的 {name}")
    return np.frombuffer(payload, dtype="<f8", count=int(np.prod(shape)), offset=start).reshape(shape).copy()


def _points(points: np.ndarray) -> np.ndarray:
    """验证并转换有限的二维模型坐标数组。"""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("点坐标必须是有限的 [N, 2] 数组")
    return array


def _depth_array(depth: np.ndarray, name: str) -> np.ndarray:
    """验证并转换二维深度数组。"""
    array = np.asarray(depth, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name}必须是二维数组")
    return array


def _matrix(matrix: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    """验证有限且形状正确的相机矩阵。"""
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}形状或数值无效")
    return array


def _image_size(size: tuple[int, int], name: str) -> tuple[int, int]:
    """验证并返回正整数的图像宽高。"""
    if len(size) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in size):
        raise ValueError(f"{name}必须是正整数宽高")
    return int(size[0]), int(size[1])


def _in_model_bounds(points: np.ndarray, image_size: int) -> np.ndarray:
    """返回落在模型输入图像边界内的点掩码。"""
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < image_size)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < image_size)
    )
