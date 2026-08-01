# PASM v0 的 MegaDepth 单阶段训练命令行入口。

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pasm_v0.backbones import FrozenFeatureExtractor, load_resized_rgb, resolve_device
    from pasm_v0.config import PasmV0Config
    from pasm_v0.data import load_manifest
    from pasm_v0.geometry import PairGeometry, load_pair_geometry, projected_match_targets, projected_patch_support_targets
    from pasm_v0.matching import PairSemanticEncoder
    from pasm_v0.pipeline import PasmV0, RoundRecord
    from pasm_v0.sampling import PatchLayout
    from pasm_v0.training import pasm_v0_loss
else:
    from .backbones import FrozenFeatureExtractor, load_resized_rgb, resolve_device
    from .config import PasmV0Config
    from .data import load_manifest
    from .geometry import PairGeometry, load_pair_geometry, projected_match_targets, projected_patch_support_targets
    from .matching import PairSemanticEncoder
    from .pipeline import PasmV0, RoundRecord
    from .sampling import PatchLayout
    from .training import pasm_v0_loss


def train_model(
    manifest_path: str | Path,
    dino_weights: str | Path,
    loma_weights: str | Path,
    output_path: str | Path,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    hidden_dim: int = 128,
    heads: int = 4,
    device: str = "auto",
    max_pairs: int | None = None,
    seed: int = 0,
    resume_path: str | Path | None = None,
) -> Path:
    """利用 MegaDepth 深度和位姿训练 v0.1 的双图语义编码器和价值头。"""
    if epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("训练轮数和学习率必须为正数")
    selected_device = resolve_device(device)
    print(format_training_device(selected_device), flush=True)
    config = PasmV0Config()
    pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("清单中没有带深度和相机信息的正图像对")
    extractor = FrozenFeatureExtractor(config, dino_weights, loma_weights, str(selected_device))
    encoder = PairSemanticEncoder(token_dim=1024, hidden_dim=hidden_dim, heads=heads).to(selected_device)
    runner = PasmV0(encoder, config)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    layout = PatchLayout(config)
    output = Path(output_path)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    start_epoch = 0
    if resume_path is not None:
        start_epoch, history, best_loss = load_training_checkpoint(
            resume_path, encoder, optimizer, selected_device
        )
        ensure_best_checkpoint(
            output, encoder, optimizer, config, hidden_dim, heads, history, best_loss
        )
    if start_epoch >= epochs:
        raise ValueError("目标训练轮数必须大于检查点已完成轮数")
    for epoch in range(start_epoch, epochs):
        encoder.train()
        loss_sum = 0.0
        used_pairs = 0
        pair_progress = tqdm(pairs, total=len(pairs), unit="对", dynamic_ncols=True)
        pair_progress.set_description(f"第 {epoch + 1}/{epochs} 轮")
        for pair_index, pair in enumerate(pair_progress, start=1):
            geometry = load_pair_geometry(pair, config.image_size)
            image_a = load_resized_rgb(pair.image_a, config.image_size)
            image_b = load_resized_rgb(pair.image_b, config.image_size)
            features = extractor.extract(image_a, image_b)
            result = runner.run(
                features.tokens_a,
                features.tokens_b,
                features.descriptor_grid_a,
                features.descriptor_grid_b,
                seed=seed + epoch * len(pairs) + pair_index,
            )
            loss = _result_loss(result.rounds, geometry, layout, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            used_pairs += 1
            pair_progress.set_description(
                format_training_progress(
                    epoch + 1,
                    epochs,
                    pair_index,
                    len(pairs),
                    float(loss.detach()),
                    loss_sum / used_pairs,
                )
            )
        epoch_loss = loss_sum / max(used_pairs, 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "pairs": used_pairs})
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
        save_training_checkpoint(
            _last_checkpoint_path(output),
            encoder,
            optimizer,
            config,
            hidden_dim,
            heads,
            history,
            best_loss,
        )
        if is_best:
            save_training_checkpoint(
                output, encoder, optimizer, config, hidden_dim, heads, history, best_loss
            )
    output.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def save_training_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    optimizer: torch.optim.Optimizer,
    config: PasmV0Config,
    hidden_dim: int,
    heads: int,
    history: list[dict[str, float | int]],
    best_loss: float,
) -> None:
    """保存模型、优化器和已完成轮次的完整训练状态。"""
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "semantic_encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": len(history),
            "history": history,
            "best_loss": best_loss,
            "config": asdict(config),
            "token_dim": 1024,
            "hidden_dim": hidden_dim,
            "heads": heads,
        },
        checkpoint,
    )


def load_training_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> tuple[int, list[dict[str, float | int]], float]:
    """恢复检查点模型和可用优化器状态，并返回训练进度。"""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "semantic_encoder" not in payload:
        raise ValueError("续训检查点不含语义编码器权重")
    encoder.load_state_dict(payload["semantic_encoder"])
    history = _checkpoint_history(payload, checkpoint)
    epoch = int(payload.get("epoch", history[-1]["epoch"] if history else 0))
    best_loss = float(payload.get("best_loss", min((item["loss"] for item in history), default=float("inf"))))
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return epoch, history, best_loss


def ensure_best_checkpoint(
    path: str | Path,
    encoder: PairSemanticEncoder,
    optimizer: torch.optim.Optimizer,
    config: PasmV0Config,
    hidden_dim: int,
    heads: int,
    history: list[dict[str, float | int]],
    best_loss: float,
) -> None:
    """在新输出路径不存在时保存恢复模型作为初始最佳模型。"""
    if not Path(path).is_file():
        save_training_checkpoint(path, encoder, optimizer, config, hidden_dim, heads, history, best_loss)


def _last_checkpoint_path(output: Path) -> Path:
    """返回与最佳检查点同目录的最后训练状态路径。"""
    return output.with_name(f"{output.stem}.last{output.suffix}")


def _checkpoint_history(payload: dict[object, object], path: Path) -> list[dict[str, float | int]]:
    """优先读取检查点历史，并兼容旧检查点的并列 JSON 历史文件。"""
    saved_history = payload.get("history")
    if isinstance(saved_history, list):
        return [dict(item) for item in saved_history if isinstance(item, dict)]
    history_path = path.with_suffix(".history.json")
    if not history_path.is_file():
        return []
    saved_history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(saved_history, list) or not all(isinstance(item, dict) for item in saved_history):
        raise ValueError("续训历史文件格式无效")
    return [dict(item) for item in saved_history]


def format_training_progress(
    epoch: int,
    total_epochs: int,
    pair_index: int,
    pair_count: int,
    loss: float,
    average_loss: float,
) -> str:
    """生成包含轮次、图像对和损失的训练进度文字。"""
    return (
        f"第 {epoch}/{total_epochs} 轮 | 图像对 {pair_index}/{pair_count} | "
        f"当前 loss={loss:.4f} | 平均 loss={average_loss:.4f}"
    )


def format_training_device(device: torch.device | str) -> str:
    """生成训练启动时的设备提示文字。"""
    return f"训练设备：{device}"


def _result_loss(
    rounds: list[RoundRecord],
    geometry: PairGeometry,
    layout: PatchLayout,
    config: PasmV0Config,
) -> torch.Tensor:
    """汇总由直接投影生成的三轮双向点级和 Patch 价值损失。"""
    support_a, support_b = projected_patch_support_targets(
        geometry, layout, config.geometry_support_stride, config.geometry_depth_tolerance
    )
    losses = []
    for record in rounds:
        target_a, valid_a, target_b, valid_b = projected_match_targets(
            record.points_a,
            record.points_b,
            geometry,
            config.geometry_match_threshold_px,
            config.geometry_depth_tolerance,
        )
        losses.append(
            pasm_v0_loss(record.edge_weights, target_a, valid_a, record.patch_values_a, support_a)
        )
        losses.append(
            pasm_v0_loss(
                record.edge_weights.transpose(0, 1),
                target_b,
                valid_b,
                record.patch_values_b,
                support_b,
            )
        )
    return torch.stack(losses).mean()


def main() -> None:
    """解析训练参数并保存 PASM v0 语义编码器检查点。"""
    parser = argparse.ArgumentParser(description="使用 MegaDepth 深度和位姿训练 PASM v0.1")
    parser.add_argument("--manifest", type=Path, required=True, help="MegaDepth 图像对清单")
    parser.add_argument("--dino-weights", type=Path, required=True, help="DINOv2 权重路径")
    parser.add_argument("--loma-weights", type=Path, required=True, help="LoMa 描述子权重路径")
    parser.add_argument("--output", type=Path, required=True, help="输出检查点路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam 学习率")
    parser.add_argument("--hidden-dim", type=int, default=128, help="语义隐藏维度")
    parser.add_argument("--heads", type=int, default=4, help="Cross-Attention 头数")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--max-pairs", type=int, help="最多使用的正图像对数量")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子")
    parser.add_argument("--resume", type=Path, help="恢复指定训练检查点并训练到目标总轮数")
    args = parser.parse_args()
    output = train_model(
        args.manifest,
        args.dino_weights,
        args.loma_weights,
        args.output,
        args.epochs,
        args.learning_rate,
        args.hidden_dim,
        args.heads,
        args.device,
        args.max_pairs,
        args.seed,
        args.resume,
    )
    print(output)


if __name__ == "__main__":
    main()
