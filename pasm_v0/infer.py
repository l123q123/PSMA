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
    from pasm_v0.backbones import (
        DINOv2PatchEncoder,
        FrozenFeatureExtractor,
        load_resized_rgb,
        pool_dino_tokens,
        resolve_device,
    )
    from pasm_v0.config import PasmV0Config
    from pasm_v0.matching import PairSemanticEncoder, mutual_match_indices
    from pasm_v0.pipeline import PasmV0, RoundRecord
    from pasm_v0.sampling import allocate_patch_budget
    from pasm_v0.dad import FrozenDaDDetector, select_global_dad_points, select_patch_budgeted_points
    from pasm_v0.loma_matcher import FrozenLoMaMatcher
else:
    from .backbones import (
        DINOv2PatchEncoder,
        FrozenFeatureExtractor,
        load_resized_rgb,
        pool_dino_tokens,
        resolve_device,
    )
    from .config import PasmV0Config
    from .matching import PairSemanticEncoder, mutual_match_indices
    from .pipeline import PasmV0, RoundRecord
    from .sampling import allocate_patch_budget
    from .dad import FrozenDaDDetector, select_global_dad_points, select_patch_budgeted_points
    from .loma_matcher import FrozenLoMaMatcher


@dataclass(frozen=True)
class ExperimentOutputPaths:
    """保存一次推理实验的固定输出文件路径。"""

    result: Path
    round_points: tuple[Path, ...]
    matches: Path
    high_confidence_matches: Path


@dataclass(frozen=True)
class StaticExperimentOutputPaths:
    """保存一次 v0.3 静态推理实验的固定输出文件路径。"""

    result: Path
    points: Path
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
        int(checkpoint["token_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["heads"]),
        int(checkpoint.get("cross_attention_layers", 1)),
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
    mode: str = "iterative_loma",
    dad_weights: str | Path | None = None,
) -> Path:
    """运行指定 PASM 推理模式并将结果保存到对应实验目录。"""
    if mode in {"dad_global", "pasm_dad"}:
        return run_static_inference(
            image_a_path,
            image_b_path,
            dino_weights,
            loma_weights,
            checkpoint,
            output_directory,
            mode,
            dad_weights,
            device,
        )
    if mode not in {"iterative_loma", "static_semantic"}:
        raise ValueError(f"不支持的推理模式：{mode}")
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


def static_experiment_output_paths(directory: str | Path) -> StaticExperimentOutputPaths:
    """创建 v0.3 静态实验目录并返回固定输出路径。"""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    return StaticExperimentOutputPaths(
        result=output / "result.npz",
        points=output / "points.png",
        matches=output / "matches.png",
        high_confidence_matches=output / "high_confidence_matches.png",
    )


def run_static_inference(
    image_a_path: str | Path,
    image_b_path: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    checkpoint: str | Path,
    output_directory: str | Path,
    mode: str,
    dad_weights: str | Path | None,
    device: str,
) -> Path:
    """运行单轮 dad_global 或 pasm_dad 推理并保存全部静态结果。"""
    if mode not in {"dad_global", "pasm_dad"}:
        raise ValueError(f"静态推理模式无效：{mode}")
    if dad_weights is None:
        raise ValueError("dad_global 和 pasm_dad 必须提供 --dad-weights")
    selected_device = resolve_device(device)
    runner = load_runner(checkpoint, selected_device)
    image_a = load_resized_rgb(image_a_path, runner.config.image_size)
    image_b = load_resized_rgb(image_b_path, runner.config.image_size)
    if mode == "pasm_dad":
        tokens_a, tokens_b = _extract_patch_tokens(
            image_a, image_b, dino_weights, runner.config, selected_device
        )
        with torch.inference_mode():
            _, _, values_a, values_b = runner.semantic_encoder(tokens_a, tokens_b)
        budget_a = allocate_patch_budget(
            values_a[0].detach().cpu().numpy(), runner.config, round_index=0
        )
        budget_b = allocate_patch_budget(
            values_b[0].detach().cpu().numpy(), runner.config, round_index=0
        )
    detector = FrozenDaDDetector(dad_weights, runner.config.image_size, selected_device)
    detections_a = detector.detect(image_a)
    detections_b = detector.detect(image_b)
    if mode == "pasm_dad":
        points_a, patch_ids_a, dad_scores_a = select_patch_budgeted_points(
            detections_a, runner.layout, budget_a
        )
        points_b, patch_ids_b, dad_scores_b = select_patch_budgeted_points(
            detections_b, runner.layout, budget_b
        )
    else:
        points_a, dad_scores_a = select_global_dad_points(
            detections_a, runner.config.image_size, runner.config.total_points
        )
        points_b, dad_scores_b = select_global_dad_points(
            detections_b, runner.config.image_size, runner.config.total_points
        )
        patch_ids_a = runner.layout.patch_ids(points_a)
        patch_ids_b = runner.layout.patch_ids(points_b)
        budget_a = np.bincount(patch_ids_a, minlength=runner.config.patch_count).astype(np.int64)
        budget_b = np.bincount(patch_ids_b, minlength=runner.config.patch_count).astype(np.int64)
    del detector, detections_a, detections_b
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    matcher = FrozenLoMaMatcher(loma_weights, dad_weights, dino_weights, selected_device)
    indices_a, indices_b, confidence = matcher.match(image_a, points_a, image_b, points_b)
    output = static_experiment_output_paths(output_directory)
    matched_a = points_a[indices_a]
    matched_b = points_b[indices_b]
    np.savez_compressed(
        output.result,
        mode=np.asarray(mode),
        selected_points_a=points_a,
        selected_points_b=points_b,
        selected_patch_ids_a=patch_ids_a,
        selected_patch_ids_b=patch_ids_b,
        selected_dad_scores_a=dad_scores_a,
        selected_dad_scores_b=dad_scores_b,
        patch_budget_a=budget_a,
        patch_budget_b=budget_b,
        keypoints_a=matched_a,
        keypoints_b=matched_b,
        confidence=confidence,
        config=np.asarray(str(asdict(runner.config))),
    )
    save_static_point_distribution(output.points, image_a, image_b, points_a, points_b, mode)
    save_match_visualization(output.matches, image_a, image_b, matched_a, matched_b)
    high_confidence = confidence >= 0.6
    save_match_visualization(
        output.high_confidence_matches,
        image_a,
        image_b,
        matched_a[high_confidence],
        matched_b[high_confidence],
    )
    return output.result


def _extract_patch_tokens(
    image_a: np.ndarray,
    image_b: np.ndarray,
    dino_weights: str | Path,
    config: PasmV0Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只加载 DINOv2 提取静态 Patch 预算所需的双图 token。"""
    encoder = DINOv2PatchEncoder(dino_weights, image_size=config.image_size).to(device).eval()
    images = torch.from_numpy(np.stack((image_a, image_b), axis=0)).to(device)
    with torch.inference_mode():
        tokens = pool_dino_tokens(
            encoder(images), config.patch_rows, config.patch_cols
        )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokens[:1], tokens[1:]


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


def save_static_point_distribution(
    output_path: str | Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    mode: str,
) -> None:
    """将 v0.3 单轮静态选点绘制为左右图对照图。"""
    from PIL import ImageDraw

    canvas, offset = _side_by_side_image(image_a, image_b)
    painter = ImageDraw.Draw(canvas)
    for points, x_offset in ((points_a, 0), (points_b, offset)):
        for x_value, y_value in np.asarray(points, dtype=np.float32):
            painter.ellipse(
                (x_value + x_offset - 2, y_value - 2, x_value + x_offset + 2, y_value + 2),
                fill="red",
            )
    painter.text((8, 8), f"{mode} 静态选点：{len(points_a)} + {len(points_b)}", fill="yellow")
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
    parser = argparse.ArgumentParser(description="运行 PASM v0 的两图匹配推理")
    parser.add_argument("--image-a", type=Path, required=True, help="第一张输入图像")
    parser.add_argument("--image-b", type=Path, required=True, help="第二张输入图像")
    parser.add_argument("--dino-weights", type=Path, required=True, help="DINOv2 权重路径")
    parser.add_argument("--loma-weights", type=Path, required=True, help="LoMa 描述子权重路径")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练得到的 PASM v0 检查点")
    parser.add_argument("--output", type=Path, required=True, help="本次实验的输出文件夹")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子")
    parser.add_argument(
        "--mode",
        choices=("iterative_loma", "static_semantic", "dad_global", "pasm_dad"),
        default="iterative_loma",
        help="推理模式：旧版三轮、旧版静态语义、DaD 全局或 PASM 静态预算",
    )
    parser.add_argument("--dad-weights", type=Path, help="DaD 本地权重路径，v0.3 模式必需")
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
        args.mode,
        args.dad_weights,
    )
    print(output)


if __name__ == "__main__":
    main()
