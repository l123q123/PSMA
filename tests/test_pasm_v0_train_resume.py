# 测试 PASM v0 训练检查点的恢复与旧格式兼容性。

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.matching import PairSemanticEncoder
from pasm_v0.train import ensure_best_checkpoint, load_training_checkpoint, save_training_checkpoint


class PasmV0TrainResumeTests(unittest.TestCase):
    """验证训练状态可精确恢复，并兼容旧模型检查点。"""

    def test_checkpoint_restores_epoch_history_best_loss_and_optimizer(self) -> None:
        """完整检查点必须恢复模型、优化器和续训进度。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        _optimizer_step(encoder, optimizer)
        history = [{"epoch": 1, "loss": 2.0, "pairs": 3}]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_training_checkpoint(
                path, encoder, optimizer, PasmV0Config(), 4, 2, history, best_loss=2.0
            )
            restored = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)

            epoch, restored_history, best_loss = load_training_checkpoint(
                path, restored, restored_optimizer, torch.device("cpu")
            )

        self.assertEqual(epoch, 1)
        self.assertEqual(restored_history, history)
        self.assertEqual(best_loss, 2.0)
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
        for expected, actual in zip(encoder.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)

    def test_legacy_checkpoint_uses_neighboring_history_for_warm_start(self) -> None:
        """旧检查点应从同名历史文件确定已完成轮数并只恢复模型权重。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        history = [{"epoch": 1, "loss": 2.5, "pairs": 3}]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "semantic_encoder": encoder.state_dict(),
                    "config": {},
                    "token_dim": 4,
                    "hidden_dim": 4,
                    "heads": 2,
                },
                path,
            )
            path.with_suffix(".history.json").write_text(json.dumps(history), encoding="utf-8")
            restored = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
            optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)

            epoch, restored_history, best_loss = load_training_checkpoint(
                path, restored, optimizer, torch.device("cpu")
            )

        self.assertEqual(epoch, 1)
        self.assertEqual(restored_history, history)
        self.assertEqual(best_loss, 2.5)
        self.assertFalse(optimizer.state)

    def test_resume_initializes_missing_best_checkpoint_from_loaded_state(self) -> None:
        """恢复到新输出路径时必须先保存当前模型作为初始最佳模型。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        history = [{"epoch": 1, "loss": 2.0, "pairs": 3}]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            ensure_best_checkpoint(
                path, encoder, optimizer, PasmV0Config(), 4, 2, history, best_loss=2.0
            )

            self.assertTrue(path.is_file())


def _optimizer_step(encoder: PairSemanticEncoder, optimizer: torch.optim.Optimizer) -> None:
    """执行一次最小反向传播以创建 Adam 状态。"""
    loss = sum(parameter.square().sum() for parameter in encoder.parameters())
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    unittest.main()
