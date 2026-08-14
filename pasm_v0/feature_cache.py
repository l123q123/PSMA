"""冻结 DINO/DaD 特征的磁盘缓存。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np


def cache_key(image_path: str | Path) -> str:
    """用规范化路径生成跨清单稳定的缓存键。"""
    return hashlib.sha256(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()[:24]




def pair_cache_key(pair_id: str) -> str:
    """为图像对标识生成稳定的缓存键。"""
    return hashlib.sha256(str(pair_id).encode("utf-8")).hexdigest()[:24]


class FeatureCache:
    """保存并读取单图冻结特征，避免训练时重复运行大模型。"""

    names = ("dense_logits", "feature_map", "tokens", "nms_points", "nms_scores")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def save(self, key: str, values: Mapping[str, np.ndarray]) -> Path:
        if set(values) != set(self.names):
            raise ValueError(f"缓存必须包含 {self.names}")
        arrays = {name: np.asarray(values[name], dtype=np.float32) for name in self.names}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("缓存特征必须全部有限")
        target = self.path(key)
        np.savez_compressed(target, **arrays)
        return target

    def load(self, key: str) -> dict[str, np.ndarray]:
        target = self.path(key)
        if not target.is_file():
            raise FileNotFoundError(target)
        with np.load(target) as payload:
            if set(payload.files) != set(self.names):
                raise ValueError(f"缓存字段不完整: {target}")
            return {name: np.asarray(payload[name], dtype=np.float32).copy() for name in self.names}
