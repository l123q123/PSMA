# PASM v0.2 的离线软质量教师缓存与质量图构造。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import PasmV0Config
from .geometry import PairGeometry
from .quality_teacher import teacher_scores
from .sampling import PatchLayout


@dataclass(frozen=True)
class QualityTeacherCache:
    """保存一对图像的双向密集质量教师图及其必要元数据。"""

    pair_id: str
    teacher_a: np.ndarray
    teacher_b: np.ndarray
    image_size: int
    quality_cells_per_axis: int

    def __post_init__(self) -> None:
        """验证教师图范围、尺寸和质量格密度。"""
        if not self.pair_id or self.image_size <= 0 or self.quality_cells_per_axis <= 0:
            raise ValueError("质量教师缓存元数据无效")
        for teacher in (self.teacher_a, self.teacher_b):
            array = np.asarray(teacher, dtype=np.float32)
            if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError("质量教师图必须是范围位于 [0, 1] 的有限二维数组")
            if array.shape[0] != array.shape[1] or array.shape[0] <= 0:
                raise ValueError("质量教师图尺寸无效")
        object.__setattr__(self, "teacher_a", np.asarray(self.teacher_a, dtype=np.float32))
        object.__setattr__(self, "teacher_b", np.asarray(self.teacher_b, dtype=np.float32))


def build_quality_teacher(
    pair_id: str,
    layout: PatchLayout,
    descriptor_grid_a: np.ndarray,
    descriptor_grid_b: np.ndarray,
    geometry: PairGeometry,
    config: PasmV0Config,
    device: str | torch.device,
    temperature: float = 0.1,
    exclusion_radius: float = 6.0,
) -> QualityTeacherCache:
    """根据几何可见性和冻结 LoMa 描述子构造双图软质量教师图。"""
    if layout.config != config or geometry.image_size != config.image_size:
        raise ValueError("质量教师的布局、几何和配置必须使用同一输入尺寸")
    candidates, patch_ids = layout.quality_cell_centers()
    scores_a = teacher_scores(
        candidates, candidates, descriptor_grid_a, descriptor_grid_b, geometry, "a_to_b",
        config.geometry_depth_tolerance, temperature, exclusion_radius, device,
    )
    scores_b = teacher_scores(
        candidates, candidates, descriptor_grid_b, descriptor_grid_a, geometry, "b_to_a",
        config.geometry_depth_tolerance, temperature, exclusion_radius, device,
    )
    return QualityTeacherCache(
        pair_id,
        _quality_map(scores_a, patch_ids, layout),
        _quality_map(scores_b, patch_ids, layout),
        config.image_size,
        config.quality_cells_per_axis,
    )


def save_quality_cache(path: str | Path, cache: QualityTeacherCache) -> None:
    """将一对图像的质量教师图和元数据保存为压缩 NPZ。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        pair_id=np.asarray(cache.pair_id),
        teacher_a=cache.teacher_a,
        teacher_b=cache.teacher_b,
        image_size=np.asarray(cache.image_size, dtype=np.int64),
        quality_cells_per_axis=np.asarray(cache.quality_cells_per_axis, dtype=np.int64),
    )


def load_quality_cache(path: str | Path, config: PasmV0Config, pair_id: str) -> QualityTeacherCache:
    """读取并严格验证当前图像对和质量格配置对应的教师缓存。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source) as payload:
        required = {"pair_id", "teacher_a", "teacher_b", "image_size", "quality_cells_per_axis"}
        if not required.issubset(payload.files):
            raise ValueError("质量教师缓存缺少必要字段")
        cache = QualityTeacherCache(
            str(payload["pair_id"].item()),
            np.asarray(payload["teacher_a"], dtype=np.float32),
            np.asarray(payload["teacher_b"], dtype=np.float32),
            int(payload["image_size"].item()),
            int(payload["quality_cells_per_axis"].item()),
        )
    expected_shape = PatchLayout(config).quality_grid_shape
    if (
        cache.pair_id != pair_id
        or cache.image_size != config.image_size
        or cache.quality_cells_per_axis != config.quality_cells_per_axis
        or cache.teacher_a.shape != expected_shape
        or cache.teacher_b.shape != expected_shape
    ):
        raise ValueError("质量教师缓存与当前图像对或质量格配置不一致")
    return cache


def _quality_map(scores: np.ndarray, patch_ids: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """将按 Patch 顺序的局部质量分数重排为全图二维质量图。"""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    ids = np.asarray(patch_ids, dtype=np.int64).reshape(-1)
    cells = layout.config.quality_cells_per_axis
    if len(values) != len(ids) or len(values) != layout.config.patch_count * cells**2:
        raise ValueError("质量教师候选分数数量与局部质量格不一致")
    rows, cols = layout.quality_grid_shape
    result = np.empty((rows, cols), dtype=np.float32)
    for patch_id in range(layout.config.patch_count):
        local = values[ids == patch_id]
        if len(local) != cells**2:
            raise ValueError("每个 Patch 必须恰好拥有全部局部质量格")
        row, col = divmod(patch_id, layout.config.patch_cols)
        result[row * cells:(row + 1) * cells, col * cells:(col + 1) * cells] = local.reshape(cells, cells)
    return result
