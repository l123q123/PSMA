# 测试 v0.4 两类 37×37 Patch 监督缓存的格式和读取校验。

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pasm_v0.config import dino_token_static_config
from pasm_v0.v04_targets import (
    load_calibrated_patch_targets,
    load_patch_targets,
    normalize_patch_target,
)


class PasmV04TargetTests(unittest.TestCase):
    """验证 A/B Patch 目标都以明确且可复现的概率缓存保存。"""

    def test_normalize_patch_target_returns_probability_distribution(self) -> None:
        """非负 Patch 权重必须规范化为和为一的概率分布。"""
        target = normalize_patch_target(np.array([0.0, 2.0, 2.0], dtype=np.float32), 3)

        np.testing.assert_allclose(target, np.array([0.0, 0.5, 0.5], dtype=np.float32))

    def test_load_patch_targets_rejects_wrong_kind_and_invalid_distribution(self) -> None:
        """读取缓存必须同时核对监督类型和概率分布有效性。"""
        config = dino_token_static_config()
        values = np.zeros(config.patch_count, dtype=np.float32)
        values[5] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "demo.npz"
            np.savez_compressed(
                path,
                target_a=values,
                target_b=values,
                target_kind=np.asarray("geometry"),
                pair_id=np.asarray("demo"),
                image_size=np.asarray(config.image_size),
                patch_rows=np.asarray(config.patch_rows),
                patch_cols=np.asarray(config.patch_cols),
                stride=np.asarray(2),
            )

            target_a, target_b = load_patch_targets(root, "demo", config, "geometry")
            np.testing.assert_allclose(target_a, values)
            np.testing.assert_allclose(target_b, values)
            with self.assertRaises(ValueError):
                load_patch_targets(root, "demo", config, "quality")

            np.savez_compressed(
                path,
                target_a=np.full(config.patch_count, 1.0, dtype=np.float32),
                target_b=values,
                target_kind=np.asarray("geometry"),
                pair_id=np.asarray("demo"),
                image_size=np.asarray(config.image_size),
                patch_rows=np.asarray(config.patch_rows),
                patch_cols=np.asarray(config.patch_cols),
                stride=np.asarray(2),
            )
            with self.assertRaises(ValueError):
                load_patch_targets(root, "demo", config, "geometry")

    def test_load_calibrated_patch_targets_preserves_raw_support_and_distribution(self) -> None:
        """校准教师必须分别保留逐 Patch 原始支持和诊断分布。"""
        config = dino_token_static_config(16)
        raw = np.zeros(config.patch_count, dtype=np.float32)
        raw[[0, 1]] = (0.25, 0.75)
        distribution = raw / raw.sum()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(
                root / "demo.npz",
                raw_target_a=raw,
                raw_target_b=raw,
                distribution_target_a=distribution,
                distribution_target_b=distribution,
                target_kind=np.asarray("geometry_calibrated"),
                target_schema=np.asarray(2),
                pair_id=np.asarray("demo"),
                image_size=np.asarray(config.image_size),
                patch_rows=np.asarray(config.patch_rows),
                patch_cols=np.asarray(config.patch_cols),
                stride=np.asarray(2),
            )

            raw_a, raw_b, distribution_a, distribution_b = load_calibrated_patch_targets(
                root, "demo", config
            )

        np.testing.assert_allclose(raw_a, raw)
        np.testing.assert_allclose(raw_b, raw)
        np.testing.assert_allclose(distribution_a, distribution)
        np.testing.assert_allclose(distribution_b, distribution)


if __name__ == "__main__":
    unittest.main()
