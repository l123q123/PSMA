# 生成 PASM v0.2 离线双图软质量教师缓存的命令行入口。

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from .backbones import FrozenLoMaDescriptor, load_resized_rgb, resolve_device
from .config import PasmV0Config
from .data import load_manifest
from .geometry import load_pair_geometry
from .quality_cache import build_quality_teacher, save_quality_cache
from .sampling import PatchLayout


def build_argument_parser() -> argparse.ArgumentParser:
    """创建离线质量教师缓存命令的参数解析器。"""
    parser = argparse.ArgumentParser(description="生成 PASM v0.2 的 MegaDepth 软质量教师缓存")
    parser.add_argument("--manifest", type=Path, required=True, help="带深度和相机路径的 MegaDepth 图像对清单")
    parser.add_argument("--loma-weights", type=Path, required=True, help="LoMa-B128 描述子权重路径")
    parser.add_argument("--output", type=Path, required=True, help="质量教师缓存输出文件夹")
    parser.add_argument("--device", default="auto", help="LoMa 和教师计算设备，如 cuda 或 cpu")
    parser.add_argument("--max-pairs", type=int, help="最多处理的正图像对数量")
    parser.add_argument("--temperature", type=float, default=0.1, help="教师正负描述子间隔的 sigmoid 温度")
    parser.add_argument("--exclusion-radius", type=float, default=6.0, help="真实投影附近不作为错误候选的半径")
    return parser


def build_quality_caches(
    manifest_path: str | Path,
    loma_weights: str | Path,
    output_directory: str | Path,
    device: str = "auto",
    max_pairs: int | None = None,
    temperature: float = 0.1,
    exclusion_radius: float = 6.0,
) -> Path:
    """为清单中的所有正图像对生成可复用的质量教师缓存。"""
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("最多图像对数量必须为正数")
    config = PasmV0Config()
    selected_device = resolve_device(device)
    pairs = [pair for pair in load_manifest(manifest_path) if pair.label == 1]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("清单中没有可生成教师的正图像对")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    layout = PatchLayout(config)
    loma = FrozenLoMaDescriptor(loma_weights, device=str(selected_device))
    for pair in tqdm(pairs, unit="对", dynamic_ncols=True, desc="生成质量教师"):
        geometry = load_pair_geometry(pair, config.image_size)
        image_a = load_resized_rgb(pair.image_a, config.image_size)
        image_b = load_resized_rgb(pair.image_b, config.image_size)
        cache = build_quality_teacher(
            pair.pair_id,
            layout,
            loma.describe_image(image_a),
            loma.describe_image(image_b),
            geometry,
            config,
            selected_device,
            temperature,
            exclusion_radius,
        )
        save_quality_cache(output / f"{pair.pair_id}.npz", cache)
    return output


def main() -> None:
    """解析命令行参数并生成离线质量教师缓存。"""
    args = build_argument_parser().parse_args()
    print(
        build_quality_caches(
            args.manifest,
            args.loma_weights,
            args.output,
            args.device,
            args.max_pairs,
            args.temperature,
            args.exclusion_radius,
        )
    )


if __name__ == "__main__":
    main()
