# PASM v0 训练所需的图像对清单和几何缓存读取工具。

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PairRecord:
    """保存一条训练图像对及其可选几何缓存路径。"""

    pair_id: str
    image_a: Path
    image_b: Path
    label: int
    geometry_cache: Path | None
    depth_a: Path | None
    depth_b: Path | None
    camera_a: Path | None
    camera_b: Path | None


def load_manifest(path: str | Path, require_files: bool = True) -> list[PairRecord]:
    """读取 MegaDepth 图像对清单并验证必要字段。"""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("清单必须包含非空的 pairs 列表")

    records = []
    for raw_pair in raw_pairs:
        try:
            record = PairRecord(
                pair_id=str(raw_pair["pair_id"]),
                image_a=Path(raw_pair["image_A"]),
                image_b=Path(raw_pair["image_B"]),
                label=int(raw_pair["label"]),
                geometry_cache=Path(raw_pair["geometry_cache"])
                if raw_pair.get("geometry_cache")
                else None,
                depth_a=Path(raw_pair["depth_A"]) if raw_pair.get("depth_A") else None,
                depth_b=Path(raw_pair["depth_B"]) if raw_pair.get("depth_B") else None,
                camera_a=Path(raw_pair["camera_A"]) if raw_pair.get("camera_A") else None,
                camera_b=Path(raw_pair["camera_B"]) if raw_pair.get("camera_B") else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("每个图像对必须包含 pair_id、image_A、image_B 和 label") from error
        if record.label not in (0, 1):
            raise ValueError("图像对标签必须为 0 或 1")
        if record.label == 1 and any(
            path is None for path in (record.depth_a, record.depth_b, record.camera_a, record.camera_b)
        ):
            raise ValueError(f"正图像对 {record.pair_id} 缺少深度或相机文件")
        if require_files:
            for source in (
                record.image_a,
                record.image_b,
                record.geometry_cache,
                record.depth_a,
                record.depth_b,
                record.camera_a,
                record.camera_b,
            ):
                if source is not None and not source.is_file():
                    raise FileNotFoundError(source)
        records.append(record)
    return records


def load_geometry(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取同序的两图正对应点坐标。"""
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path) as payload:
        if "points_A" not in payload or "points_B" not in payload:
            raise ValueError("几何缓存必须包含 points_A 和 points_B")
        points_a = np.asarray(payload["points_A"], dtype=np.float32).copy()
        points_b = np.asarray(payload["points_B"], dtype=np.float32).copy()
    if points_a.ndim != 2 or points_a.shape[1] != 2 or points_b.shape != points_a.shape:
        raise ValueError("几何点必须是形状相同的 [N, 2] 数组")
    if not np.isfinite(points_a).all() or not np.isfinite(points_b).all():
        raise ValueError("几何点必须为有限数值")
    return points_a, points_b


def scale_geometry(
    points_a: np.ndarray,
    points_b: np.ndarray,
    scale_a: tuple[float, float],
    scale_b: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """按两图各自缩放比例变换对应点坐标。"""
    scaled_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2).copy()
    scaled_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2).copy()
    if len(scaled_a) != len(scaled_b):
        raise ValueError("两图几何点数量必须一致")
    scaled_a *= np.asarray(scale_a, dtype=np.float32)
    scaled_b *= np.asarray(scale_b, dtype=np.float32)
    return scaled_a, scaled_b


def scaled_pair_geometry(
    pair: PairRecord, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """读取图像对几何点并缩放到模型输入尺寸。"""
    if pair.geometry_cache is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()
    points_a, points_b = load_geometry(pair.geometry_cache)
    with Image.open(pair.image_a) as image_a, Image.open(pair.image_b) as image_b:
        width_a, height_a = image_a.size
        width_b, height_b = image_b.size
    height, width = image_size
    return scale_geometry(
        points_a,
        points_b,
        (width / width_a, height / height_a),
        (width / width_b, height / height_b),
    )
