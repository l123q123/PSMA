"""Evaluation helpers matching the initial LVSM-NeuLoc training objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import EpisodeDataset
from .model import LVSMNeuLoc
from .model_diffusion import RayDiffusionNeuLoc
from .solver import CarPose, solve_car_pose


def _batch_scalar(batch: dict[str, torch.Tensor], key: str) -> int | None:
    value = batch.get(key)
    if value is None:
        return None
    values = value.reshape(-1).unique()
    if values.numel() == 0 or int(values[0]) <= 0:
        return None
    if values.numel() != 1:
        raise ValueError(f"mixed {key} values in one batch are not supported")
    return int(values[0])

def query_camera_rays(intrinsics: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    """Return query patch-center rays in the model's [u, v] pixel convention."""
    ys = (torch.arange(image_size // patch_size, device=intrinsics.device, dtype=intrinsics.dtype) + 0.5) * patch_size
    xs = (torch.arange(image_size // patch_size, device=intrinsics.device, dtype=intrinsics.dtype) + 0.5) * patch_size
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx, gy), -1).reshape(-1, 2)
    homogeneous = torch.cat((grid, torch.ones_like(grid[:, :1])), -1).expand(intrinsics.shape[0], -1, -1)
    return F.normalize(torch.linalg.solve(intrinsics[:, None], homogeneous[..., None]).squeeze(-1), dim=-1)


def pose_metrics(rotation: torch.Tensor, center: torch.Tensor, target_pose: torch.Tensor) -> dict[str, float]:
    target_rotation = target_pose[..., :3, :3]
    target_center = target_pose[..., :3, 3]
    relative = rotation.transpose(-1, -2) @ target_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    target_translation = -(target_rotation.transpose(-1, -2) @ target_center.unsqueeze(-1)).squeeze(-1)
    translation = -(rotation.transpose(-1, -2) @ center.unsqueeze(-1)).squeeze(-1)
    return {
        "rotation_deg": float(torch.rad2deg(torch.acos(cosine)).mean()),
        "center_error": float(torch.linalg.vector_norm(center - target_center, dim=-1).mean()),
        "translation_error": float(torch.linalg.vector_norm(translation - target_translation, dim=-1).mean()),
    }


def load_model(checkpoint: str | Path, device: torch.device, image_size: int) -> LVSMNeuLoc:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"]
    dim = state["latent_tokens"].shape[1]
    latent_count = state["latent_tokens"].shape[0]
    encoder_layers = 1 + max(int(key.split(".")[1]) for key in state if key.startswith("encoder."))
    decoder_layers = 1 + max(int(key.split(".")[1]) for key in state if key.startswith("decoder."))
    head_dim = state["encoder.0.attn.q_norm.weight"].numel()
    if dim % head_dim:
        raise ValueError("checkpoint attention head dimension does not divide model dimension")
    config = payload.get("model_config", {})
    no_z = bool(config.get("no_z", False))
    relative_pose = bool(config.get("relative_pose", False) or "relative_pose_head.weight" in state)
    local_refinement = bool(config.get("local_refinement", False) or "refinement_stage_embedding.weight" in state)
    relative_robust_center_scale = float(config.get("relative_robust_center_scale", 0.25))
    if "diffusion_head.weight" in state:
        diffusion_layers = 1 + max(
            int(key.split(".")[1]) for key in state if key.startswith("diffusion_blocks.")
        )
        model = RayDiffusionNeuLoc(
            image_size=image_size, patch_size=8, dim=dim, latent_count=latent_count,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, heads=dim // head_dim,
            diffusion_layers=diffusion_layers,
            diffusion_steps=int(config.get("diffusion_train_steps", 50)),
            moment_scale=float(config.get("moment_scale", 1.0)),
            relative_pose=relative_pose,
            relative_robust_center_scale=relative_robust_center_scale,
            local_refinement=local_refinement,
            no_z=no_z,
        )
    else:
        model = LVSMNeuLoc(
            image_size=image_size, patch_size=8, dim=dim, latent_count=latent_count,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, heads=dim // head_dim,
            relative_pose=relative_pose,
            relative_robust_center_scale=relative_robust_center_scale,
            local_refinement=local_refinement,
            no_z=no_z,
        )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(
    model: LVSMNeuLoc, batch: dict[str, torch.Tensor], image_size: int, diffusion_steps: int = 0,
):
    stage_id = batch.get("stage_id")
    model_kwargs = dict(
        support_type_ids=batch.get("support_type_ids"),
        support_mask=batch.get("support_mask"),
        view_token_count=_batch_scalar(batch, "view_token_count"),
        support_token_budget=_batch_scalar(batch, "support_token_budget"),
    )
    if isinstance(model, RayDiffusionNeuLoc):
        if diffusion_steps < 2:
            raise ValueError("diffusion evaluation requires at least two DDIM steps")
        output = model.sample(
            batch["support_images"], batch["support_intrinsics"], batch["support_poses"],
            batch["query_image"], batch["query_intrinsics"], sample_steps=diffusion_steps, **model_kwargs,
        )
    else:
        output = model(
            batch["support_images"], batch["support_intrinsics"], batch["support_poses"],
            batch["query_image"], batch["query_intrinsics"], **model_kwargs,
        )
    local = query_camera_rays(batch["query_intrinsics"], image_size, model.patch_size)
    ray_pose = solve_car_pose(output.directions, output.moments, local)
    pose = ray_pose
    relative = output.relative_pose
    if relative is not None and stage_id is not None:
        use_relative = stage_id.reshape(-1) > 0
        pose = CarPose(
            rotation=torch.where(use_relative[:, None, None], relative.fused.rotation, ray_pose.rotation),
            center=torch.where(use_relative[:, None], relative.fused.center, ray_pose.center),
            translation=torch.where(use_relative[:, None], relative.fused.translation, ray_pose.translation),
        )
    target_rotation = batch["query_pose"][:, :3, :3]
    target_center = batch["query_pose"][:, :3, 3]
    target_directions = F.normalize((target_rotation[:, None] @ local[..., None]).squeeze(-1), dim=-1)
    target_moments = torch.cross(target_center[:, None].expand_as(target_directions), target_directions, dim=-1)
    metrics = pose_metrics(pose.rotation, pose.center, batch["query_pose"])
    metrics["ray_mse"] = float(F.mse_loss(output.directions, target_directions))
    metrics["moment_mse"] = float(F.mse_loss(output.moments, target_moments))
    return output, pose, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--support-views", type=int, default=4)
    parser.add_argument("--max-support-views", type=int, default=0,
                        help="pad support tensors to this many slots; 0 disables padding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diffusion-steps", type=int, default=16, help="DDIM steps for diffusion checkpoints")
    args = parser.parse_args()
    device = torch.device(args.device)
    max_support_views = args.max_support_views if args.max_support_views > 0 else None
    dataset = EpisodeDataset(
        args.manifest,
        args.image_size,
        args.support_views,
        max_support_views=max_support_views,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(args.checkpoint, device, args.image_size)
    records = []
    for index, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        _, _, metrics = predict(model, batch, args.image_size, diffusion_steps=args.diffusion_steps)
        records.append({"index": index, **metrics})
        print(f"[{index + 1}/{len(dataset)}] rot={metrics['rotation_deg']:.3f} center={metrics['center_error']:.4f}", flush=True)
    means = {key: sum(record[key] for record in records) / len(records) for key in records[0] if key != "index"}
    payload = {"split": "training_manifest", "episodes": len(records), "mean": means, "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
