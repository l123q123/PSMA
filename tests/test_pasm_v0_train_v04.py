# 测试 PASM v0.4 单轮训练的损失拆分与检查点元数据。

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from inspect import signature
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pasm_v0.config import dino_token_static_config
from pasm_v0.cache_v04_targets import main as cache_v04_targets_main
from pasm_v0.diagnose_v04_distribution import calibrated_support_metrics
from pasm_v0.matching import PairSemanticEncoder
from pasm_v0.train_v04 import (
    initialize_encoder_from_checkpoint,
    load_v04_checkpoint,
    main as train_v04_main,
    point_match_loss,
    prepare_teacher_validation_pairs,
    save_v04_checkpoint,
    set_training_seed,
    train_model,
    update_validation_early_stopping,
    v04_calibrated_shape_loss,
    v04_loss_terms,
)
from pasm_v0.evaluate_v04 import evaluate_ablation
from pasm_v0.v04_targets import cache_targets


class PasmV04TrainTests(unittest.TestCase):
    """验证 A/B 训练只改变价值教师且可恢复其身份。"""

    def test_static_config_supports_16_patch_grid(self) -> None:
        """静态单轮配置应支持切换到 16×16 Patch 网格。"""
        config = dino_token_static_config(16)

        self.assertEqual((config.patch_rows, config.patch_cols), (16, 16))
        self.assertEqual(config.iterations, 1)
        self.assertEqual(config.total_points, 1024)
        self.assertEqual(config.max_points_per_patch, 20)

    def test_static_config_rejects_unknown_patch_grid(self) -> None:
        """静态单轮配置只接受 16 或 37 两种网格。"""
        with self.assertRaises(ValueError):
            dino_token_static_config(20)

    def test_v04_entrypoints_accept_patch_grid(self) -> None:
        """缓存、训练和评测入口都必须显式接收同一个 Patch 网格参数。"""
        self.assertIn("patch_grid", signature(cache_targets).parameters)
        self.assertIn("patch_grid", signature(train_model).parameters)
        self.assertIn("patch_grid", signature(evaluate_ablation).parameters)

    def test_calibrated_target_kind_is_available_from_cache_cli(self) -> None:
        """缓存命令行帮助必须暴露 geometry_calibrated 教师类型。"""
        stdout = StringIO()
        with patch("sys.argv", ["cache_v04_targets", "--help"]), redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as error:
                cache_v04_targets_main()

        self.assertEqual(error.exception.code, 0)
        self.assertIn("geometry_calibrated", stdout.getvalue())

    def test_v04_checkpoint_records_selected_target_kind(self) -> None:
        """v0.4 检查点必须明确记录所用 Patch 监督类型。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.pt"
            save_v04_checkpoint(
                path,
                encoder,
                optimizer,
                dino_token_static_config(),
                hidden_dim=4,
                heads=2,
                history=[],
                best_loss=float("inf"),
                target_kind="geometry",
                target_cache=Path(directory) / "targets",
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["target_kind"], "geometry")
        self.assertIn("target_cache", payload)

    def test_v04_checkpoint_records_cross_attention_layers(self) -> None:
        """v0.4 检查点必须记录语义编码器的 Cross-Attention 层数。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2, cross_attention_layers=2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry_deep.pt"
            save_v04_checkpoint(
                path,
                encoder,
                optimizer,
                dino_token_static_config(),
                hidden_dim=4,
                heads=2,
                history=[],
                best_loss=float("inf"),
                target_kind="geometry",
                target_cache=Path(directory) / "targets",
                cross_attention_layers=2,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["cross_attention_layers"], 2)

    def test_v04_checkpoint_records_calibrated_match_settings(self) -> None:
        """点级微调检查点应记录固定候选池和匹配损失权重。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrated_match.pt"
            save_v04_checkpoint(
                path,
                encoder,
                optimizer,
                dino_token_static_config(),
                hidden_dim=4,
                heads=2,
                history=[],
                best_loss=float("inf"),
                target_kind="geometry_calibrated",
                target_cache=Path(directory) / "targets",
                loss_mode="calibrated_match",
                match_candidate_count=1536,
                match_weight=0.2,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["match_candidate_count"], 1536)
        self.assertEqual(payload["match_weight"], 0.2)

    def test_resume_rejects_checkpoint_from_other_target_kind(self) -> None:
        """geometry 训练不能恢复 quality 的优化器和模型状态。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.pt"
            save_v04_checkpoint(
                path,
                encoder,
                optimizer,
                dino_token_static_config(),
                hidden_dim=4,
                heads=2,
                history=[],
                best_loss=float("inf"),
                target_kind="quality",
                target_cache=Path(directory) / "targets",
            )
            restored = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)

            with self.assertRaises(ValueError):
                load_v04_checkpoint(
                    path,
                    restored,
                    restored_optimizer,
                    torch.device("cpu"),
                    expected_target_kind="geometry",
                )

    def test_v04_loss_terms_use_supplied_patch_target_and_keep_gradients(self) -> None:
        """不同价值教师应只改变价值损失，且匹配边和价值均可反传。"""
        edge_weights = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32, requires_grad=True
        )
        patch_values = torch.tensor([0.9, 0.1], dtype=torch.float32, requires_grad=True)
        indices = torch.tensor([0, 1], dtype=torch.long)
        valid = torch.tensor([True, True])

        match_a, value_a, total_a = v04_loss_terms(
            edge_weights, indices, valid, patch_values, np.array([1.0, 0.0], dtype=np.float32)
        )
        match_b, value_b, total_b = v04_loss_terms(
            edge_weights, indices, valid, patch_values, np.array([0.0, 1.0], dtype=np.float32)
        )
        total_a.backward(retain_graph=True)

        self.assertAlmostEqual(float(match_a.detach()), float(match_b.detach()))
        self.assertLess(float(value_a.detach()), float(value_b.detach()))
        self.assertGreater(float(total_b.detach()), float(total_a.detach()))
        self.assertIsNotNone(edge_weights.grad)
        self.assertIsNotNone(patch_values.grad)

    def test_v04_value_only_uses_only_value_gradient(self) -> None:
        """纯价值模式只让 Patch 价值参与反传。"""
        edge_weights = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32, requires_grad=True
        )
        patch_values = torch.tensor([0.9, 0.1], dtype=torch.float32, requires_grad=True)

        match_loss, value_loss, total_loss = v04_loss_terms(
            edge_weights,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([True, True]),
            patch_values,
            np.array([1.0, 0.0], dtype=np.float32),
            loss_mode="value_only",
        )
        total_loss.backward()

        self.assertAlmostEqual(float(total_loss.detach()), float(value_loss.detach()))
        self.assertIsNone(edge_weights.grad)
        self.assertIsNotNone(patch_values.grad)

    def test_calibrated_shape_loss_optimizes_raw_support_and_keeps_gradients(self) -> None:
        """校准模式必须逐 Patch 拟合 raw support，并保留弱 shape 约束。"""
        patch_values = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
        raw_target = np.array([0.2, 0.8], dtype=np.float32)
        distribution_target = np.array([0.2, 0.8], dtype=np.float32)

        local, regression, shape, total = v04_calibrated_shape_loss(
            patch_values, raw_target, distribution_target
        )
        total.backward()

        self.assertGreater(float(local.detach()), 0.0)
        self.assertGreater(float(regression.detach()), 0.0)
        self.assertGreater(float(shape.detach()), 0.0)
        self.assertGreater(float(total.detach()), float(local.detach()))
        self.assertIsNotNone(patch_values.grad)

    def test_point_match_loss_rewards_geometric_true_edges_and_backpropagates(self) -> None:
        correct = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32, requires_grad=True
        )
        incorrect = torch.tensor(
            [[0.2, 0.8], [0.8, 0.2]], dtype=torch.float32, requires_grad=True
        )
        targets = torch.tensor([0, 1], dtype=torch.long)
        valid = torch.tensor([True, True])

        correct_loss = point_match_loss(correct, targets, valid)
        incorrect_loss = point_match_loss(incorrect, targets, valid)
        correct_loss.backward()

        self.assertLess(float(correct_loss.detach()), float(incorrect_loss.detach()))
        self.assertIsNotNone(correct.grad)

    def test_initialize_encoder_from_checkpoint_loads_only_semantic_weights(self) -> None:
        """微调初始化应载入语义权重，但不要求恢复旧优化器或训练历史。"""
        source = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            save_v04_checkpoint(
                path,
                source,
                optimizer,
                dino_token_static_config(),
                hidden_dim=4,
                heads=2,
                history=[{"epoch": 1, "total_loss": 0.1}],
                best_loss=0.1,
                target_kind="geometry_calibrated",
                target_cache=Path(directory) / "targets",
                loss_mode="calibrated_shape",
            )
            restored = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
            initialize_encoder_from_checkpoint(path, restored, torch.device("cpu"))

        for expected, actual in zip(source.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)

    def test_calibrated_support_metrics_report_raw_map_error(self) -> None:
        """校准诊断必须报告 raw support 的绝对误差与 Brier 分数。"""
        metrics = calibrated_support_metrics(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.25, 0.75], dtype=np.float32),
        )

        self.assertAlmostEqual(metrics["raw_mae"], 0.25)
        self.assertAlmostEqual(metrics["brier"], 0.0625)

    def test_training_seed_reproduces_encoder_initialization(self) -> None:
        """相同种子必须让 A/B 从完全相同的可训练初始化开始。"""
        set_training_seed(13)
        first = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)
        set_training_seed(13)
        second = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)

        for expected, actual in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)

    def test_validation_early_stopping_resets_on_lower_teacher_loss(self) -> None:
        """验证教师损失下降时应刷新最佳值并清零早停计数。"""
        improved, best, count, stop = update_validation_early_stopping(
            metric=0.29,
            best_metric=0.30,
            no_improvement_count=2,
            patience=5,
        )

        self.assertEqual((improved, best, count, stop), (True, 0.29, 0, False))

    def test_validation_early_stopping_stops_after_patience(self) -> None:
        """验证教师损失连续未下降达到 patience 次后应请求停止训练。"""
        improved, best, count, stop = update_validation_early_stopping(
            metric=0.30,
            best_metric=0.30,
            no_improvement_count=4,
            patience=5,
        )

        self.assertEqual((improved, best, count, stop), (False, 0.30, 5, True))

    def test_validation_early_stopping_can_be_disabled(self) -> None:
        """patience 为零时应继续记录未提升次数，但永不提前停止。"""
        improved, best, count, stop = update_validation_early_stopping(
            metric=0.31,
            best_metric=0.30,
            no_improvement_count=4,
            patience=0,
        )

        self.assertEqual((improved, best, count, stop), (False, 0.30, 5, False))

    def test_v04_cli_exposes_teacher_validation_options(self) -> None:
        """训练入口应暴露基于 held-out 几何教师的验证与早停参数。"""
        stdout = StringIO()
        with patch("sys.argv", ["train_v04", "--help"]), redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as error:
                train_v04_main()

        self.assertEqual(error.exception.code, 0)
        self.assertIn("--validation-image-root", stdout.getvalue())
        self.assertIn("--validation-manifest", stdout.getvalue())
        self.assertIn("--early-stop-patience", stdout.getvalue())

    def test_teacher_validation_accepts_custom_manifest(self) -> None:
        """验证教师应能从自定义 manifest 读取指定数量的正图像对。"""
        from pasm_v0.data import PairRecord

        records = [
            PairRecord(
                pair_id=f"pair-{index}",
                image_a=Path(f"a-{index}.jpg"),
                image_b=Path(f"b-{index}.jpg"),
                label=1,
                geometry_cache=Path(f"geometry-{index}.npz"),
                depth_a=Path(f"depth-a-{index}.exr"),
                depth_b=Path(f"depth-b-{index}.exr"),
                camera_a=Path(f"camera-a-{index}.safetensor"),
                camera_b=Path(f"camera-b-{index}.safetensor"),
            )
            for index in range(3)
        ]
        config = dino_token_static_config(16)
        with patch("pasm_v0.train_v04.load_manifest", return_value=records) as load_manifest_mock, \
            patch("pasm_v0.train_v04.load_pair_geometry", return_value=object()), \
            patch(
                "pasm_v0.train_v04.projected_patch_support_fractions",
                return_value=(np.ones(config.patch_count), np.ones(config.patch_count)),
            ), \
            patch("pasm_v0.evaluate_v03.load_heldout_pairs", side_effect=AssertionError("旧验证集加载器不应被调用")), \
            patch("pasm_v0.evaluate_v03.validate_heldout_against_training"):
            result = prepare_teacher_validation_pairs(
                image_root=Path("unused"),
                training_manifest=Path("training.json"),
                config=config,
                max_pairs=2,
                validation_manifest=Path("validation.json"),
            )

        self.assertEqual(len(result), 2)
        self.assertEqual([item.pair.pair_id for item in result], ["pair-0", "pair-1"])
        load_manifest_mock.assert_called_once_with(Path("validation.json"))


if __name__ == "__main__":
    unittest.main()
