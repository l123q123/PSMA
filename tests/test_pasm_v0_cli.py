# 测试 PASM v0 命令行入口的基本可用性。

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pasm_v0.config import PasmV0Config
from pasm_v0.infer import (
    experiment_output_paths,
    load_runner,
    round_anchor_counts,
    save_match_visualization,
    save_quality_map,
    save_round_point_distribution,
)
from pasm_v0.matching import PairLocalQualityEncoder, PairSemanticEncoder
from pasm_v0.pipeline import RoundRecord
from pasm_v0.train import format_training_device, format_training_progress
from pasm_v0.visualize_quality_teacher import (
    build_argument_parser,
    save_match_comparison,
    save_point_comparison,
    save_summary,
)


class PasmV0CliTests(unittest.TestCase):
    """验证命令行帮助无需加载权重即可运行。"""

    def test_inference_command_exposes_required_arguments(self) -> None:
        """推理命令应显示两张图、权重和输出参数。"""
        script = Path(__file__).parents[1] / "pasm_v0" / "infer.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASM v0", result.stdout)
        self.assertIn("--image-a", result.stdout)
        self.assertIn("--dino-weights", result.stdout)
        self.assertIn("--checkpoint", result.stdout)

    def test_quality_teacher_defaults_keep_ten_pairs_and_hundred_display_edges(self) -> None:
        """软质量教师命令应默认处理十对图并限制每张连线图的边数。"""
        arguments = build_argument_parser().parse_args([
            "--manifest", "pairs.json", "--loma-weights", "loma.pth", "--output", "results"
        ])

        self.assertEqual(arguments.max_pairs, 10)
        self.assertEqual(arguments.display_matches, 100)
        self.assertEqual(arguments.temperature, 0.1)

    def test_quality_teacher_command_exposes_data_and_teacher_arguments(self) -> None:
        """软质量教师命令帮助应列出数据、权重和教师分数参数。"""
        script = Path(__file__).parents[1] / "pasm_v0" / "visualize_quality_teacher.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--loma-weights", result.stdout)
        self.assertIn("--exclusion-radius", result.stdout)

    def test_quality_teacher_visualizations_and_summary_are_written(self) -> None:
        """软质量教师辅助输出应保存两类图片和可读取的 CSV 汇总。"""
        image_a = np.zeros((3, 16, 16), dtype=np.float32)
        image_b = np.ones((3, 16, 16), dtype=np.float32)
        points_a = np.array([[2.0, 3.0], [12.0, 10.0]], dtype=np.float32)
        points_b = np.array([[4.0, 5.0], [11.0, 9.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points_path = root / "teacher_points.png"
            matches_path = root / "teacher_matches.png"
            summary_path = root / "summary.csv"
            save_point_comparison(points_path, image_a, image_b, points_a, points_b, [0.2, 0.8], [0.3, 0.9])
            save_match_comparison(matches_path, image_a, image_b, points_a, points_b, [0, 1], [0, 1], [0.8, 0.7])
            save_summary(summary_path, [{"pair_id": "sample"}])

            self.assertTrue(points_path.is_file())
            self.assertTrue(matches_path.is_file())
            self.assertIn("sample", summary_path.read_text(encoding="utf-8"))

    def test_training_progress_includes_total_base_and_quality_losses(self) -> None:
        """训练进度文字应同时显示 total、base 和 quality 三种损失。"""
        text = format_training_progress(3, 10, 47, 96, 0.625, 0.5, 0.4, 0.225)

        self.assertEqual(
            text,
            "第 3/10 轮 | 图像对 47/96 | 当前 total=0.6250 | 平均 total=0.5000 | "
            "base=0.4000 | quality=0.2250",
        )

    def test_training_device_text_identifies_selected_device(self) -> None:
        """训练启动信息应明确显示实际使用的设备。"""
        self.assertEqual(format_training_device("cuda"), "训练设备：cuda")

    def test_training_command_exposes_quality_cache_and_resume_arguments(self) -> None:
        """v0.2 训练命令必须提供教师缓存和续训参数。"""
        script = Path(__file__).parents[1] / "pasm_v0" / "train.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", result.stdout)
        self.assertIn("--quality-cache", result.stdout)

    def test_inference_visualizations_save_point_and_match_images(self) -> None:
        """推理可视化应保存一张轮次点图和一张最终连线图。"""
        image_a = np.zeros((3, 16, 16), dtype=np.float32)
        image_b = np.ones((3, 16, 16), dtype=np.float32)
        points_a = np.array([[2.0, 3.0], [12.0, 10.0]], dtype=np.float32)
        points_b = np.array([[4.0, 5.0], [11.0, 9.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            point_image = Path(directory) / "round_1_points.png"
            match_image = Path(directory) / "matches.png"
            quality_image = Path(directory) / "quality_a.png"
            save_round_point_distribution(point_image, image_a, image_b, points_a, points_b, 1)
            save_match_visualization(match_image, image_a, image_b, points_a, points_b)
            save_quality_map(quality_image, image_a, np.zeros((4, 4), dtype=np.float32))

            self.assertTrue(point_image.is_file())
            self.assertTrue(match_image.is_file())
            self.assertTrue(quality_image.is_file())

    def test_experiment_output_directory_uses_fixed_result_names(self) -> None:
        """一次推理的所有结果必须放入指定实验目录，并使用固定文件名。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = experiment_output_paths(Path(directory) / "实验一")

            self.assertEqual(paths.result.name, "result.npz")
            self.assertEqual([path.name for path in paths.round_points], [
                "round_1_points.png",
                "round_2_points.png",
                "round_3_points.png",
            ])
            self.assertEqual(paths.matches.name, "matches.png")
            self.assertEqual(paths.high_confidence_matches.name, "high_confidence_matches.png")
            self.assertEqual(paths.quality_maps[0].name, "quality_a.png")
            self.assertEqual(paths.quality_maps[1].name, "quality_b.png")
            self.assertTrue(paths.result.parent.is_dir())

    def test_experiment_output_directory_rejects_npz_file_path(self) -> None:
        """推理输出必须是目录，不能再传入旧版 NPZ 文件路径。"""
        with self.assertRaises(ValueError):
            experiment_output_paths("outputs/result.npz")

    def test_round_anchor_counts_follow_recorded_patch_ids(self) -> None:
        """推理输出统计应按锚点所在 Patch 记录每轮数量。"""
        record = RoundRecord(
            points_a=np.array([[2.0, 2.0], [12.0, 2.0]], dtype=np.float32),
            points_b=np.array([[3.0, 3.0], [13.0, 3.0]], dtype=np.float32),
            patch_ids_a=np.array([0, 1], dtype=np.int64),
            patch_ids_b=np.array([2, 3], dtype=np.int64),
            budget_a=np.array([1, 1, 0, 0], dtype=np.int64),
            budget_b=np.array([0, 0, 1, 1], dtype=np.int64),
            edge_weights=torch.eye(2),
            patch_values_a=torch.ones(4),
            patch_values_b=torch.ones(4),
            anchor_indices_a=np.array([1], dtype=np.int64),
            anchor_indices_b=np.array([0], dtype=np.int64),
            anchor_scores=np.array([0.9], dtype=np.float32),
        )

        counts_a, counts_b = round_anchor_counts([record], patch_count=4)

        np.testing.assert_array_equal(counts_a, np.array([[0, 1, 0, 0]], dtype=np.int64))
        np.testing.assert_array_equal(counts_b, np.array([[0, 0, 1, 0]], dtype=np.int64))

    def test_v02_inference_rejects_checkpoint_without_local_quality_encoder(self) -> None:
        """v0.2 推理必须拒绝不含局部质量网络的旧检查点。"""
        encoder = PairSemanticEncoder(token_dim=4, hidden_dim=4, heads=2)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "semantic_encoder": encoder.state_dict(),
                    "config": {},
                    "token_dim": 4,
                    "hidden_dim": 4,
                    "heads": 2,
                },
                checkpoint,
            )
            with self.assertRaises(ValueError):
                load_runner(checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
