"""Shape-safe partial loading for official LVSM checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class LoadReport:
    loaded: int
    skipped: int
    missing: int


def load_partial_checkpoint(module: nn.Module, path: str | Path) -> LoadReport:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dict")
    current = module.state_dict()
    compatible = {}
    skipped = 0
    for key, value in state.items():
        normalized = key[7:] if key.startswith("module.") else key
        normalized = _remap_lvsm_key(normalized, current)
        if normalized == "latent_tokens" and value.ndim == 2 and value.shape[0] >= current[normalized].shape[0]:
            value = value[: current[normalized].shape[0]]
        if normalized in current and current[normalized].shape == value.shape:
            compatible[normalized] = value
            if normalized.startswith("image_tokenizer."):
                query_key = "query_tokenizer." + normalized[len("image_tokenizer."):]
                if query_key in current and current[query_key].shape == value.shape:
                    compatible[query_key] = value
        else:
            skipped += 1
    result = module.load_state_dict(compatible, strict=False)
    return LoadReport(loaded=len(compatible), skipped=skipped, missing=len(result.missing_keys))


def load_full_ray_initialization(
    module: nn.Module,
    official_lvsm_path: str | Path,
    ray_checkpoint_path: str | Path,
) -> tuple[LoadReport, LoadReport]:
    """Initialize a full ray-only model from official LVSM then a small NeuLoc ray model.

    Official LVSM supplies the full latent array and all twelve Transformer
    blocks.  Compatible learned localization weights then overwrite the shared
    tokenizer, query path, ray heads, and the first compact-model blocks.
    A smaller ray checkpoint cannot overwrite a larger latent array because
    ``load_partial_checkpoint`` only accepts matching tensor shapes.
    """
    official_report = load_partial_checkpoint(module, official_lvsm_path)
    ray_report = load_ray_branch_initialization(module, ray_checkpoint_path)
    return official_report, ray_report


def load_ray_branch_initialization(module: nn.Module, path: str | Path) -> LoadReport:
    """Load query-ray parameters without overwriting the official LVSM scene encoder."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dict")
    current = module.state_dict()
    prefixes = (
        "query_tokenizer.", "decoder.", "decoder_norm.",
        "direction_head.", "moment_head.", "view_token_pools.",
        "token_type_embedding.",
    )
    compatible = {}
    skipped = 0
    for raw_key, value in state.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key.startswith(prefixes):
            continue
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped += 1
    result = module.load_state_dict(compatible, strict=False)
    return LoadReport(loaded=len(compatible), skipped=skipped, missing=len(result.missing_keys))


def load_grloc_from_ray_checkpoint(
    module: nn.Module,
    path: str | Path,
    initialize_point_from_ray: bool = True,
) -> LoadReport:
    """Warm-start GRLoc from a trained ray-only NeuLoc model.

    The shared scene encoder is loaded directly.  Existing query tokens and
    decoder blocks always initialize the ray query path.  By default they also
    initialize the independent point path for backwards compatibility.  Set
    ``initialize_point_from_ray=False`` to retain the point path's constructor
    initialization while keeping the shared encoder and ray path pretrained.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dict")
    current = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped = 0
    for raw_key, value in state.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        targets = [key]
        if key.startswith("query_tokenizer."):
            suffix = key[len("query_tokenizer."):]
            targets = [f"ray_query_tokenizer.{suffix}"]
            if initialize_point_from_ray:
                targets.append(f"point_query_tokenizer.{suffix}")
        elif key.startswith("decoder."):
            suffix = key[len("decoder."):]
            targets = [f"ray_decoder.{suffix}"]
            if initialize_point_from_ray:
                targets.append(f"point_decoder.{suffix}")
        elif key.startswith("decoder_norm."):
            suffix = key[len("decoder_norm."):]
            targets = [f"ray_decoder_norm.{suffix}"]
            if initialize_point_from_ray:
                targets.append(f"point_decoder_norm.{suffix}")
        elif key.startswith("direction_head."):
            suffix = key[len("direction_head."):]
            targets = [f"direction_head.{suffix}"]
            if initialize_point_from_ray:
                targets.append(f"point_head.{suffix}")
        loaded_any = False
        for target in targets:
            if target in current and current[target].shape == value.shape:
                compatible[target] = value
                loaded_any = True
        if not loaded_any:
            skipped += 1
    result = module.load_state_dict(compatible, strict=False)
    return LoadReport(loaded=len(compatible), skipped=skipped, missing=len(result.missing_keys))


def _remap_lvsm_key(key: str, current: dict[str, torch.Tensor]) -> str:
    """Translate official LVSM names to the compact module names."""
    candidates = [key]
    if key == "n_light_field_latent":
        candidates.append("latent_tokens")
    if key.startswith("image_tokenizer.1."):
        candidates.append("image_tokenizer." + key[len("image_tokenizer.1."):])
        candidates.append("query_tokenizer." + key[len("image_tokenizer.1."):])
    if key.startswith("target_pose_tokenizer.1."):
        candidates.append("query_tokenizer." + key[len("target_pose_tokenizer.1."):])
    for prefix, replacement in (("transformer_encoder.", "encoder."), ("transformer_decoder.", "decoder.")):
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            suffix = suffix.replace(".attn.fc.", ".attn.proj.")
            suffix = suffix.replace(".mlp.mlp.0.", ".mlp.0.").replace(".mlp.mlp.2.", ".mlp.2.")
            candidates.append(replacement + suffix)
    for candidate in candidates:
        if candidate in current:
            return candidate
    return key
