# PASM v0 的两图推理命令行入口。

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pasm_v0.backbones import FrozenFeatureExtractor, load_resized_rgb, resolve_device
    from pasm_v0.config import PasmV0Config
    from pasm_v0.matching import PairSemanticEncoder, mutual_match_indices
    from pasm_v0.pipeline import PasmV0, RoundRecord
else:
    from .backbones import FrozenFeatureExtractor, load_resized_rgb, resolve_device
    from .config import PasmV0Config
    from .matching import PairSemanticEncoder, mutual_match_indices
    from .pipeline import PasmV0, RoundRecord


@dataclass(frozen=True)
class ExperimentOutputPaths:
    """保存一次推理实验的固定输出文件路径。"""

    result: Path
    round_points: tuple[Path, ...]
    matches: Path
    high_confidence_matches: Path


def load_runner(
    checkpoint_path: str | Path, device: torch.device, strengthened_sampling: bool = False
) -> PasmV0:
    """从训练检查点恢复 v0 配置和双图语义编码器。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_data = dict(checkpoint["config"])
    config_data.setdefault("budget_score_power", 1.0)
    if strengthened_sampling:
        strengthened = PasmV0Config()
        config_data.update(
            anchor_threshold=strengthened.anchor_threshold,
            budget_score_power=strengthened.budget_score_power,
            jitter_ratio=strengthened.jitter_ratio,
        )
    config_data["round_min_points"] = tuple(config_data["round_min_points"])
    config = PasmV0Config(**config_data)
    encoder = PairSemanticEncoder(
        int(checkpoint["token_dim"]), int(checkpoint["hidden_dim"]), int(checkpoint["heads"])
    ).to(device)
    encoder.load_state_dict(checkpoint["semantic_encoder"], strict=True)
    return PasmV0(encoder.eval(), config)


def run_inference(
    image_a_path: str | Path,
    image_b_path: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    checkpoint: str | Path,
    output_directory: str | Path,
    device: str = "auto",
    seed: int | None = 0,
    strengthened_sampling: bool = False,
) -> Path:
    """运行 PASM v0 并将所有结果保存到指定实验目录。"""
    selected_device = resolve_device(device)
    runner = load_runner(checkpoint, selected_device, strengthened_sampling)
    extractor = FrozenFeatureExtractor(runner.config, dino_weights, loma_weights, device=str(selected_device))
    image_a = load_resized_rgb(image_a_path, runner.config.image_size)
    image_b = load_resized_rgb(image_b_path, runner.config.image_size)
    features = extractor.extract(image_a, image_b)
    with torch.inference_mode():
        result = runner.run(
            features.tokens_a,
            features.tokens_b,
            features.descriptor_grid_a,
            features.descriptor_grid_b,
            seed=seed,
        )
        indices_a, indices_b = mutual_match_indices(result.edge_weights, runner.config.match_threshold)
    indices_a = indices_a.cpu().numpy()
    indices_b = indices_b.cpu().numpy()
    confidence = result.edge_weights[indices_a, indices_b].detach().cpu().numpy()
    output = experiment_output_paths(output_directory)
    anchor_count_a, anchor_count_b = round_anchor_counts(result.rounds, runner.config.patch_count)
    np.savez_compressed(
        output.result,
        keypoints_a=result.points_a[indices_a],
        keypoints_b=result.points_b[indices_b],
        confidence=confidence,
        patch_ids_a=result.patch_ids_a[indices_a],
        patch_ids_b=result.patch_ids_b[indices_b],
        round_budget_a=np.stack([record.budget_a for record in result.rounds]),
        round_budget_b=np.stack([record.budget_b for record in result.rounds]),
        round_points_a=np.stack([record.points_a for record in result.rounds]),
        round_points_b=np.stack([record.points_b for record in result.rounds]),
        round_anchor_count_a=anchor_count_a,
        round_anchor_count_b=anchor_count_b,
        config=np.asarray(str(asdict(runner.config))),
    )
    for round_index, (record, point_path) in enumerate(zip(result.rounds, output.round_points, strict=True), start=1):
        save_round_point_distribution(
            point_path,
            image_a,
            image_b,
            record.points_a,
            record.points_b,
            round_index,
        )
    save_match_visualization(
        output.matches,
        image_a,
        image_b,
        result.points_a[indices_a],
        result.points_b[indices_b],
    )
    high_confidence = confidence >= 0.6
    save_match_visualization(
        output.high_confidence_matches,
        image_a,
        image_b,
        result.points_a[indices_a][high_confidence],
        result.points_b[indices_b][high_confidence],
    )
    return output.result


def experiment_output_paths(directory: str | Path) -> ExperimentOutputPaths:
    """创建实验目录并返回固定名称的全部输出路径。"""
    output = Path(directory)
    if output.suffix:
        raise ValueError("--output 必须是实验输出文件夹，不能是文件路径")
    output.mkdir(parents=True, exist_ok=True)
    return ExperimentOutputPaths(
        result=output / "result.npz",
        round_points=tuple(output / f"round_{index}_points.png" for index in range(1, 4)),
        matches=output / "matches.png",
        high_confidence_matches=output / "high_confidence_matches.png",
    )


def round_anchor_counts(rounds: list[RoundRecord], patch_count: int) -> tuple[np.ndarray, np.ndarray]:
    """按每轮记录的锚点索引统计两图各 Patch 的锚点数量。"""
    if patch_count <= 0:
        raise ValueError("Patch 总数必须为正数")
    counts_a = []
    counts_b = []
    for record in rounds:
        counts_a.append(
            np.bincount(record.patch_ids_a[record.anchor_indices_a], minlength=patch_count).astype(np.int64)
        )
        counts_b.append(
            np.bincount(record.patch_ids_b[record.anchor_indices_b], minlength=patch_count).astype(np.int64)
        )
    return np.stack(counts_a), np.stack(counts_b)


def save_round_point_distribution(
    output_path: str | Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    round_index: int,
) -> None:
    """将指定轮次的两图采样点绘制为左右对照图。"""
    from PIL import ImageDraw

    canvas, offset = _side_by_side_image(image_a, image_b)
    painter = ImageDraw.Draw(canvas)
    for points, x_offset in ((points_a, 0), (points_b, offset)):
        for x_value, y_value in np.asarray(points, dtype=np.float32):
            painter.ellipse((x_value + x_offset - 2, y_value - 2, x_value + x_offset + 2, y_value + 2), fill="red")
    painter.text((8, 8), f"第 {round_index} 轮采样点", fill="yellow")
    canvas.save(output_path)


def save_match_visualization(
    output_path: str | Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> None:
    """将最终互为最强的点级匹配绘制为左右图之间的连线。"""
    from PIL import ImageDraw

    canvas, offset = _side_by_side_image(image_a, image_b)
    painter = ImageDraw.Draw(canvas)
    for (x_a, y_a), (x_b, y_b) in zip(np.asarray(points_a), np.asarray(points_b), strict=True):
        painter.line((x_a, y_a, x_b + offset, y_b), fill="lime", width=1)
        painter.ellipse((x_a - 2, y_a - 2, x_a + 2, y_a + 2), fill="red")
        painter.ellipse((x_b + offset - 2, y_b - 2, x_b + offset + 2, y_b + 2), fill="red")
    painter.text((8, 8), f"最终匹配：{len(points_a)} 对", fill="yellow")
    canvas.save(output_path)


def _side_by_side_image(image_a: np.ndarray, image_b: np.ndarray):
    """将两张 CHW RGB 图像转换为同高的左右拼接 Pillow 图像。"""
    from PIL import Image

    first = _to_pillow_image(image_a)
    second = _to_pillow_image(image_b)
    if first.height != second.height:
        raise ValueError("两张可视化图像高度必须一致")
    canvas = Image.new("RGB", (first.width + second.width, first.height))
    canvas.paste(first, (0, 0))
    canvas.paste(second, (first.width, 0))
    return canvas, first.width


def _to_pillow_image(image: np.ndarray):
    """将归一化 CHW RGB 数组转换为可绘制的 Pillow 图像。"""
    from PIL import Image

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError("可视化图像必须具有 [3, H, W] 形状")
    rgb = np.clip(array.transpose(1, 2, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def main() -> None:
    """解析命令行参数并执行 PASM v0 两图推理。"""
    parser = argparse.ArgumentParser(description="运行 PASM v0 的三轮点级联合重采样")
    parser.add_argument("--image-a", type=Path, required=True, help="第一张输入图像")
    parser.add_argument("--image-b", type=Path, required=True, help="第二张输入图像")
    parser.add_argument("--dino-weights", type=Path, required=True, help="DINOv2 权重路径")
    parser.add_argument("--loma-weights", type=Path, required=True, help="LoMa 描述子权重路径")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练得到的 PASM v0 检查点")
    parser.add_argument("--output", type=Path, required=True, help="本次实验的输出文件夹")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子")
    parser.add_argument(
        "--strengthened-sampling", action="store_true", help="使用当前强化采样参数，不修改检查点权重"
    )
    args = parser.parse_args()
    output = run_inference(
        args.image_a,
        args.image_b,
        args.dino_weights,
        args.loma_weights,
        args.checkpoint,
        args.output,
        args.device,
        args.seed,
        args.strengthened_sampling,
    )
    print(output)


if __name__ == "__main__":
    main()
