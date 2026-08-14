"""预计算训练清单中每张图的冻结 DINO/DaD 特征。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .backbones import DINOv2PatchEncoder, load_resized_rgb, pool_dino_tokens, resolve_device
from .dad import FrozenDaDDetector
from .data import load_manifest
from .feature_cache import FeatureCache, cache_key
from .infer import load_runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--dad-weights", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    source = load_runner(args.semantic_checkpoint, device)
    dino = DINOv2PatchEncoder(args.dino_weights, source.config.image_size).to(device).eval()
    detector = FrozenDaDDetector(args.dad_weights, source.config.image_size, device)
    for module in (dino, source.semantic_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    cache = FeatureCache(args.output)
    images = sorted({path for pair in load_manifest(args.manifest) if pair.label == 1 for path in (pair.image_a, pair.image_b)})
    for path in tqdm(images, desc="预计算特征", unit="图"):
        key = cache_key(path)
        if cache.path(key).is_file():
            continue
        image = load_resized_rgb(path, source.config.image_size)
        detection = detector.detect(image)
        with torch.inference_mode():
            tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            tokens = pool_dino_tokens(dino(tensor), source.config.patch_rows, source.config.patch_cols)
        cache.save(key, {
            "dense_logits": detection.dense_logits,
            "feature_map": detection.feature_map,
            "tokens": tokens[0].detach().cpu().numpy(),
            "nms_points": detection.nms_points,
            "nms_scores": detection.nms_scores,
        })
    print(f"缓存完成: {len(images)} 张图 -> {args.output}")


if __name__ == "__main__":
    main()
