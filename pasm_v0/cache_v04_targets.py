# 生成 PASM v0.4 几何或质量教师 Patch 目标缓存的命令行入口。

from __future__ import annotations

import argparse
from pathlib import Path

from .v04_targets import cache_targets


def main() -> None:
    """解析命令行参数并生成指定监督类型的目标缓存。"""
    parser = argparse.ArgumentParser(description="生成 PASM v0.4 的 Patch 监督缓存")
    parser.add_argument("--manifest", type=Path, required=True, help="MegaDepth 训练图像对清单")
    parser.add_argument("--output", type=Path, required=True, help="目标缓存输出文件夹")
    parser.add_argument(
        "--target-kind",
        choices=("geometry", "quality", "geometry_calibrated"),
        required=True,
        help="Patch 监督类型",
    )
    parser.add_argument("--loma-weights", type=Path, help="LoMa-B128 权重；quality 时必需")
    parser.add_argument("--max-pairs", type=int, help="最多处理的正图像对数量")
    parser.add_argument("--device", default="auto", help="计算设备，如 cuda 或 cpu")
    parser.add_argument("--patch-grid", type=int, default=37, choices=(16, 37), help="Patch 网格边长")
    args = parser.parse_args()
    print(
        cache_targets(
            args.manifest,
            args.output,
            args.target_kind,
            args.loma_weights,
            args.max_pairs,
            args.device,
            args.patch_grid,
        )
    )


if __name__ == "__main__":
    main()
