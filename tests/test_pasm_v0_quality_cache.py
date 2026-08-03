# 测试 PASM v0.2 离线软质量教师缓存。

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from pasm_v0.config import PasmV0Config
from pasm_v0.cache_quality_teacher import build_argument_parser
from pasm_v0.quality_cache import QualityTeacherCache, load_quality_cache, save_quality_cache


class PasmV0QualityCacheTests(unittest.TestCase):
    """验证质量教师缓存的元数据和数值读取。"""

    def setUp(self) -> None:
        """创建小尺寸质量格配置。"""
        self.config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            quality_cells_per_axis=2,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
        )

    def test_quality_cache_round_trip_preserves_both_teacher_maps(self) -> None:
        """保存后读取必须保留双图教师图和元数据。"""
        teacher_a = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
        teacher_b = 1.0 - teacher_a
        cache = QualityTeacherCache("pair-1", teacher_a, teacher_b, 32, 2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair-1.npz"
            save_quality_cache(path, cache)
            loaded = load_quality_cache(path, self.config, "pair-1")

        np.testing.assert_array_equal(loaded.teacher_a, teacher_a)
        np.testing.assert_array_equal(loaded.teacher_b, teacher_b)
        self.assertEqual(loaded.image_size, 32)
        self.assertEqual(loaded.quality_cells_per_axis, 2)

    def test_quality_cache_rejects_mismatched_pair_or_grid(self) -> None:
        """错误图像对或质量格配置的缓存不得被训练误用。"""
        cache = QualityTeacherCache("pair-1", np.ones((8, 8), np.float32), np.zeros((8, 8), np.float32), 32, 2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair-1.npz"
            save_quality_cache(path, cache)
            with self.assertRaises(ValueError):
                load_quality_cache(path, self.config, "pair-2")
            with self.assertRaises(ValueError):
                load_quality_cache(path, replace(self.config, quality_cells_per_axis=4), "pair-1")

    def test_cache_command_exposes_required_inputs(self) -> None:
        """缓存命令必须暴露数据、LoMa 权重和输出路径。"""
        arguments = build_argument_parser().parse_args([
            "--manifest", "pairs.json", "--loma-weights", "loma.pth", "--output", "cache"
        ])

        self.assertEqual(arguments.max_pairs, None)
        self.assertEqual(arguments.temperature, 0.1)


if __name__ == "__main__":
    unittest.main()
