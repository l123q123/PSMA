# 测试 PASM v0.2 的密集质量监督损失。

import unittest

import numpy as np
import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.geometry import PairGeometry
from pasm_v0.matching import PairLocalQualityEncoder, PairSemanticEncoder
from pasm_v0.pipeline import PasmV0
from pasm_v0.sampling import PatchLayout
from pasm_v0.training import pasm_v02_loss


class PasmV02TrainingTests(unittest.TestCase):
    """验证质量损失能直接训练双图局部质量网络。"""

    def test_v02_loss_backpropagates_into_dense_quality_head(self) -> None:
        """密集教师 BCE 必须向质量头和局部投影产生梯度。"""
        config = PasmV0Config(
            image_size=32,
            patch_rows=4,
            patch_cols=4,
            quality_cells_per_axis=2,
            total_points=32,
            round_min_points=(2, 1, 0),
            max_points_per_patch=4,
        )
        torch.manual_seed(3)
        local = PairLocalQualityEncoder(local_dim=4, semantic_dim=8, heads=2)
        runner = PasmV0(PairSemanticEncoder(8, 8, 2), local, config)
        result = runner.run(
            torch.randn(1, 16, 8),
            torch.randn(1, 16, 8),
            torch.randn(4, 8, 8),
            torch.randn(4, 8, 8),
            seed=4,
        )
        depth = np.full((32, 32), 2.0, dtype=np.float32)
        geometry = PairGeometry(depth, depth.copy(), np.eye(3), np.eye(3), np.eye(4), np.eye(4), (32, 32), (32, 32), 32)
        teacher_a = np.ones((8, 8), dtype=np.float32)
        teacher_b = np.zeros((8, 8), dtype=np.float32)

        total, base, quality = pasm_v02_loss(
            result.rounds, geometry, PatchLayout(config), config, teacher_a, teacher_b
        )
        total.backward()

        self.assertGreater(float(base.detach()), 0.0)
        self.assertGreater(float(quality.detach()), 0.0)
        self.assertIsNotNone(local.quality_head.weight.grad)
        self.assertIsNotNone(local.local_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
