# 测试 PASM v0 冻结特征接口的设备与张量边界行为。

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pasm_v0.backbones import FrozenFeatureExtractor, clone_frozen_tokens, resolve_device
from pasm_v0.config import PasmV0Config


class PasmV0BackboneTests(unittest.TestCase):
    """验证冻结特征可安全进入可训练语义头。"""

    def test_auto_device_uses_cpu_when_cuda_is_unavailable(self) -> None:
        """自动设备在无 CUDA 环境下必须回退到 CPU。"""
        with patch("pasm_v0.backbones.torch.cuda.is_available", return_value=False):
            device = resolve_device("auto")

        self.assertEqual(device.type, "cpu")

    def test_cloned_inference_tokens_support_backward(self) -> None:
        """冻结推理张量复制后必须可以被可训练层用于反传。"""
        with torch.inference_mode():
            frozen = torch.ones(2, 3)

        tokens = clone_frozen_tokens(frozen)
        head = torch.nn.Linear(3, 1)
        head(tokens).sum().backward()

        self.assertFalse(tokens.is_inference())
        self.assertIsNotNone(head.weight.grad)

    def test_extractor_returns_trainable_copy_of_frozen_dino_tokens(self) -> None:
        """特征提取器必须在推理上下文之外复制 DINO token。"""
        config = PasmV0Config(
            image_size=28, patch_rows=2, patch_cols=2, total_points=8, round_min_points=(2, 1, 0)
        )
        extractor = FrozenFeatureExtractor.__new__(FrozenFeatureExtractor)
        extractor.config = config
        extractor.device = torch.device("cpu")
        extractor.dino = Mock(return_value=torch.ones(2, 4, 3))
        extractor.loma = Mock()
        extractor.loma.describe_image.return_value = np.ones((4, 7, 7), dtype=np.float32)

        features = extractor.extract(
            np.ones((3, 28, 28), dtype=np.float32), np.ones((3, 28, 28), dtype=np.float32)
        )

        self.assertFalse(features.tokens_a.is_inference())
        torch.nn.Linear(3, 1)(features.tokens_a).sum().backward()


if __name__ == "__main__":
    unittest.main()
