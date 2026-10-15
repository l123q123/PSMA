"""Minimal single-node DDP fine-tuning entry point for LVSM-NeuLoc."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from neuloc_lvsm.checkpoint import load_full_ray_initialization, load_partial_checkpoint
from neuloc_lvsm.data import EpisodeDataset
from neuloc_lvsm.evaluate import predict, pose_metrics
from neuloc_lvsm.model import LVSMNeuLoc
from neuloc_lvsm.model_diffusion import RayDiffusionNeuLoc
from neuloc_lvsm.solver import solve_car_pose


def _batch_scalar(batch: dict[str, torch.Tensor], key: str, fallback: int | None) -> int | None:
    value = batch.get(key)
    if value is None:
        return fallback
    values = value.reshape(-1).unique()
    if values.numel() == 0 or int(values[0]) <= 0:
        return fallback
    if values.numel() != 1:
        raise ValueError(f"mixed {key} values in one batch are not supported yet")
    return int(values[0])

def _progress_postfix_train(loss: float, avg_loss: float, *, rotation_deg=None, center_error=None) -> dict[str, str]:
    result = {"loss": f"{loss:.4f}", "avg_loss": f"{avg_loss:.4f}"}
    if rotation_deg is not None:
        result["denoise_rot"] = f"{rotation_deg:.3f}deg"
    if center_error is not None:
        result["denoise_center"] = f"{center_error:.4f}"
    return result


def _progress_postfix_val(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "rot": f"{metrics['rotation_deg']:.3f}deg",
        "center": f"{metrics['center_error']:.4f}",
    }


def train_loader_kwargs(num_workers: int) -> dict:
    """Return loader settings that improve input throughput without changing samples."""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    kwargs = {"num_workers": num_workers, "pin_memory": True}
    if num_workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return kwargs


def configure_cuda_performance() -> None:
    """Enable safe kernel selection flags; this does not alter model/data semantics."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def apply_full_lvsm_ray_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the exact full-capacity LVSM geometry used by ray-only NeuLoc."""
    if not args.full_lvsm_ray:
        return args
    args.image_size = 256
    if args.support_views != 0:
        args.support_views = 8
    args.dim = 768
    args.latent_count = 3072
    args.encoder_layers = 12
    args.decoder_layers = 12
    args.heads = 12
    args.amp_dtype = "bf16"
    # Preserve the episode support count. Token-budget manifests use
    # 100/50/20/8 views with 80/160/400/1024 tokens per view, respectively;
    # truncating them to eight views silently changes the training protocol.
    args.gradient_checkpointing = False
    args.support_views = 0
    args.max_support_views = 0
    return args


def validate_training_args(args: argparse.Namespace) -> None:
    if args.val_every <= 0 or args.val_max_episodes < 0 or args.unseen_val_max_episodes < 0:
        raise ValueError("validation frequency and episode limits must be valid")
    if args.amp_dtype == "bf16" and not torch.cuda.is_available():
        raise RuntimeError("BF16 training requires CUDA")
    if args.relative_pose_loss_weight < 0 or args.relative_robust_center_scale <= 0:
        raise ValueError("relative pose loss weight and robust center scale must be valid")
    if args.full_lvsm_ray:
        if not args.official_lvsm_checkpoint or not args.ray_checkpoint:
            raise ValueError("full LVSM ray mode requires both official and ray checkpoints")
        if args.resume or args.lvsm_checkpoint:
            raise ValueError("full LVSM ray initialization cannot be combined with resume or --lvsm-checkpoint")
    elif args.ray_checkpoint or args.official_lvsm_checkpoint:
        raise ValueError("--official-lvsm-checkpoint and --ray-checkpoint require --full-lvsm-ray")
    if args.ray_diffusion:
        if args.diffusion_train_steps < 2:
            raise ValueError("diffusion train steps must be at least two")
        if args.diffusion_sample_steps < 2 or args.diffusion_sample_steps > args.diffusion_train_steps:
            raise ValueError("diffusion sample steps must be between two and diffusion train steps")
        if args.moment_scale <= 0:
            raise ValueError("moment scale must be positive")


def validate_visible_gpu_count(world_size: int, visible_gpu_count: int) -> None:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if visible_gpu_count < world_size:
        raise RuntimeError(
            f"DDP requested {world_size} processes but only {visible_gpu_count} visible GPUs are available"
        )


def _init_ddp(timeout_minutes: int = 45) -> tuple[bool, int, int]:
    if timeout_minutes <= 0:
        raise ValueError("DDP timeout must be positive")
    if "RANK" not in os.environ:
        return False, 0, 1
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=timeout_minutes),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA, but CUDA is not available")
    validate_visible_gpu_count(world_size, torch.cuda.device_count())
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible GPU count")
    torch.cuda.set_device(local_rank)
    return True, local_rank, world_size


def load_resume_model(model: torch.nn.Module, checkpoint_path: str | Path) -> tuple[int, float]:
    """Restore model weights and return the next epoch plus prior best loss."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("resume checkpoint must contain a model state dictionary")
    model.load_state_dict(payload["model"], strict=True)
    return int(payload.get("epoch", -1)) + 1, float(payload.get("loss", float("inf")))


def _pose_loss(pred, target):
    rot = pred.rotation
    gt = target[..., :3, :3]
    trace = (rot.transpose(-1, -2) @ gt).diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)).mean()
    return angle + F.smooth_l1_loss(pred.center, target[..., :3, 3])


def _masked_pose_loss(pred, target, mask):
    """Pose loss over selected batch elements, with a graph-preserving zero."""
    mask = mask.to(dtype=pred.center.dtype)
    rotation = pred.rotation
    gt = target[..., :3, :3]
    trace = (rotation.transpose(-1, -2) @ gt).diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    center = F.smooth_l1_loss(pred.center, target[..., :3, 3], reduction="none").mean(dim=-1)
    return ((angle + center) * mask).sum() / mask.sum().clamp_min(1.0)


def _relative_pose_loss(relative, target, support_poses, support_mask, stage_id):
    """Supervise each support-query edge plus its robustly fused global pose."""
    active_views = support_mask & (stage_id.reshape(-1, 1) > 0)
    weights = active_views.to(dtype=relative.translation.dtype)
    support_rotation = support_poses[..., :3, :3]
    support_center = support_poses[..., :3, 3]
    target_rotation = target[:, None, :3, :3]
    target_center = target[:, None, :3, 3]
    target_relative_rotation = support_rotation.transpose(-1, -2) @ target_rotation
    target_relative_translation = (
        support_rotation.transpose(-1, -2) @ (target_center - support_center).unsqueeze(-1)
    ).squeeze(-1)
    trace = (relative.rotation.transpose(-1, -2) @ target_relative_rotation).diagonal(dim1=-2, dim2=-1).sum(-1)
    edge_rotation = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    edge_translation = F.smooth_l1_loss(relative.translation, target_relative_translation, reduction="none").mean(dim=-1)
    edge_loss = ((edge_rotation + edge_translation) * weights).sum() / weights.sum().clamp_min(1.0)
    fused_loss = _masked_pose_loss(relative.fused, target, stage_id.reshape(-1) > 0)
    return edge_loss + fused_loss


STAGE_ORDER = ("global", "semi", "dense", "fine")
METRIC_KEYS = ("rotation_deg", "center_error", "translation_error", "ray_mse", "moment_mse")


def validation_score(metrics: dict[str, float]) -> float:
    """Return the scalar used to select a checkpoint."""
    return float(metrics["rotation_deg"]) + float(metrics["center_error"])


def stage_indices(dataset: EpisodeDataset, max_episodes: int = 0) -> dict[str, list[int]]:
    """Group manifest indices by stage; max_episodes applies independently per stage."""
    grouped = {stage: [] for stage in STAGE_ORDER}
    for index, episode in enumerate(dataset.episodes):
        stage = str(episode.get("stage", "global"))
        if stage not in grouped:
            stage = "global"
        if not max_episodes or len(grouped[stage]) < max_episodes:
            grouped[stage].append(index)
    return grouped


def fixed_training_probe(dataset: EpisodeDataset, per_stage: int) -> EpisodeDataset:
    if per_stage <= 0:
        raise ValueError("probe count must be positive")
    probe = copy.copy(dataset)
    selected = []
    for indices in stage_indices(dataset).values():
        positions = torch.linspace(0, len(indices) - 1, min(per_stage, len(indices))).long().tolist() if indices else []
        selected.extend(indices[i] for i in positions)
    probe.episodes = [dataset.episodes[i] for i in selected]
    return probe


def stagewise_validation_score(metrics_by_stage: dict[str, dict[str, float]]) -> float:
    """Give each present stage equal weight when selecting a checkpoint."""
    scores = [validation_score(metrics) for stage, metrics in metrics_by_stage.items() if metrics and stage in STAGE_ORDER]
    if not scores:
        raise ValueError("no stage metrics available")
    return sum(scores) / len(scores)


@torch.inference_mode()
def evaluate_validation(
    model, dataset: EpisodeDataset, device: torch.device, image_size: int,
    max_episodes: int = 0, desc: str = "Validate", *,
    ddp: bool = False, local_rank: int = 0, world_size: int = 1, batch_size: int = 1,
    diffusion_sample_steps: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Evaluate all stages in parallel across ranks without cross-stage padding."""
    if batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    model.eval()
    grouped = stage_indices(dataset, max_episodes=max_episodes)
    global_sums = {key: 0.0 for key in METRIC_KEYS}
    global_count = 0
    metrics_by_stage: dict[str, dict[str, float]] = {}
    for stage in STAGE_ORDER:
        indices = grouped[stage]
        rank_indices = indices[local_rank::world_size] if ddp else indices
        subset = Subset(dataset, rank_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        sums = torch.zeros(len(METRIC_KEYS), device=device, dtype=torch.float64)
        count = torch.zeros(1, device=device, dtype=torch.float64)
        progress = tqdm(loader, total=(len(rank_indices) + batch_size - 1) // batch_size,
                         desc=f"{desc} [{stage}]", dynamic_ncols=True, leave=True,
                         disable=local_rank != 0)
        for batch_index, batch in enumerate(progress):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            # Evaluation does not consume the training RNG; sample seeds are stable across epochs.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(20261015 + rank_indices[batch_index * batch_size])
                _, _, metrics = predict(model, batch, image_size, diffusion_steps=diffusion_sample_steps)
            n = int(batch["query_pose"].shape[0])
            sums += torch.tensor([float(metrics[key]) * n for key in METRIC_KEYS], device=device, dtype=torch.float64)
            count += n
            if local_rank == 0 and count.item() > 0:
                progress.set_postfix(rot=f"{(sums[0] / count).item():.3f}deg", center=f"{(sums[1] / count).item():.4f}")
        if ddp:
            packed = torch.cat((sums, count))
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            sums, count = packed[:-1], packed[-1:]
        stage_count = int(count.item())
        if stage_count:
            stage_metrics = {key: float(sums[i].item() / stage_count) for i, key in enumerate(METRIC_KEYS)}
            metrics_by_stage[stage] = stage_metrics
            for i, key in enumerate(METRIC_KEYS):
                global_sums[key] += float(sums[i].item())
            global_count += stage_count
            if local_rank == 0:
                print(f"{desc} stage={stage} episodes={stage_count} rot={stage_metrics['rotation_deg']:.4f}deg center={stage_metrics['center_error']:.6f}", flush=True)
    if global_count == 0:
        raise ValueError("validation manifest produced no episodes")
    overall = {key: value / global_count for key, value in global_sums.items()}
    if ddp:
        dist.barrier()
    return overall, metrics_by_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--unseen-val-manifest")
    parser.add_argument("--output", default="outputs/lvsm_neuloc")
    parser.add_argument("--lvsm-checkpoint")
    parser.add_argument("--official-lvsm-checkpoint")
    parser.add_argument("--ray-checkpoint")
    parser.add_argument("--resume", help="model checkpoint from which to continue training")
    parser.add_argument("--full-lvsm-ray", action="store_true")
    parser.add_argument("--ray-diffusion", action="store_true", help="replace direct ray regression with conditional ray diffusion")
    parser.add_argument("--no-z", action="store_true", help="keep encoded per-support tokens instead of the unified Z bottleneck")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--train-probe-per-stage", type=int, default=0, help="fixed training episodes per stage evaluated by pure-noise sampling")
    parser.add_argument("--diffusion-layers", type=int, default=4)
    parser.add_argument("--diffusion-train-steps", type=int, default=50)
    parser.add_argument("--diffusion-sample-steps", type=int, default=16)
    parser.add_argument("--moment-scale", type=float, default=1.0)
    parser.add_argument("--diffusion-ray-loss-weight", type=float, default=0.5)
    parser.add_argument("--diffusion-pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--relative-pose", action="store_true", help="predict per-support query-relative poses for semi/dense/fine")
    parser.add_argument("--relative-pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--relative-robust-center-scale", type=float, default=0.25)
    parser.add_argument("--local-refinement", action="store_true", help="enable query-local residual refinement modules")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4, help="data loader workers per DDP rank")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--support-views", type=int, default=4)
    parser.add_argument("--view-token-count", type=int, default=None, help="pool each support view to this many tokens")
    parser.add_argument("--support-token-budget", type=int, default=None, help="pad flattened support tokens to this budget")
    parser.add_argument("--max-support-views", type=int, default=0, help="pad support tensors to this many slots; 0 disables padding")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--latent-count", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="none")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps; 0 means full epochs")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--ddp-timeout-minutes", type=int, default=45)
    parser.add_argument("--val-max-episodes", type=int, default=0, help="0 means evaluate the full validation manifest")
    parser.add_argument("--unseen-val-max-episodes", type=int, default=0, help="0 means evaluate the full unseen validation manifest")
    parser.add_argument("--val-batch-size", type=int, default=1, help="batch size within each validation stage")
    return parser


def main() -> None:
    parser = build_parser()
    args = apply_full_lvsm_ray_preset(parser.parse_args())
    validate_training_args(args)
    if args.train_probe_per_stage < 0:
        raise ValueError("training probe count must be non-negative")
    torch.manual_seed(args.seed)
    configure_cuda_performance()
    ddp, local_rank, world_size = _init_ddp(args.ddp_timeout_minutes)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    max_support_views = args.max_support_views if args.max_support_views > 0 else None
    dataset = EpisodeDataset(args.manifest, image_size=args.image_size, support_views=args.support_views, max_support_views=max_support_views)
    val_dataset = EpisodeDataset(args.val_manifest, image_size=args.image_size, support_views=args.support_views, max_support_views=max_support_views) if args.val_manifest else None
    unseen_val_dataset = EpisodeDataset(args.unseen_val_manifest, image_size=args.image_size, support_views=args.support_views, max_support_views=max_support_views) if args.unseen_val_manifest else None
    train_probe = fixed_training_probe(dataset, args.train_probe_per_stage) if args.train_probe_per_stage else None
    sampler = DistributedSampler(dataset, shuffle=True) if ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, **train_loader_kwargs(args.num_workers))
    model_class = RayDiffusionNeuLoc if args.ray_diffusion else LVSMNeuLoc
    model_kwargs = dict(
        image_size=args.image_size,
        patch_size=8,
        dim=args.dim,
        latent_count=args.latent_count,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        gradient_checkpointing=args.gradient_checkpointing,
        relative_pose=args.relative_pose,
        relative_robust_center_scale=args.relative_robust_center_scale,
        local_refinement=args.local_refinement,
        no_z=args.no_z,
    )
    if args.ray_diffusion:
        model_kwargs.update(
            diffusion_layers=args.diffusion_layers,
            diffusion_steps=args.diffusion_train_steps,
            moment_scale=args.moment_scale,
        )
    model = model_class(**model_kwargs).to(device)
    if args.resume and args.lvsm_checkpoint:
        raise ValueError("use either --resume or --lvsm-checkpoint, not both")
    start_epoch = 0
    best_epoch_loss = float("inf")
    best_val_score = float("inf")
    best_unseen_score = float("inf")
    if args.resume:
        start_epoch, best_epoch_loss = load_resume_model(model, args.resume)
        if local_rank == 0:
            print(f"resumed={args.resume} start_epoch={start_epoch} prior_best={best_epoch_loss:.6f}", flush=True)
    elif args.lvsm_checkpoint:
        report = load_partial_checkpoint(model, args.lvsm_checkpoint)
        if local_rank == 0:
            print(f"loaded={report.loaded} skipped={report.skipped} missing={report.missing}", flush=True)
    elif args.full_lvsm_ray:
        official_report, ray_report = load_full_ray_initialization(
            model, args.official_lvsm_checkpoint, args.ray_checkpoint,
        )
        if local_rank == 0:
            print(
                f"official_loaded={official_report.loaded} official_skipped={official_report.skipped} "
                f"official_missing={official_report.missing}",
                flush=True,
            )
            print(
                f"ray_loaded={ray_report.loaded} ray_skipped={ray_report.skipped} "
                f"ray_missing={ray_report.missing}",
                flush=True,
            )
    inherited, new = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (new if name.startswith(("query_tokenizer", "direction_head", "moment_head", "token_type_embedding", "view_token_pools", "diffusion_", "relative_")) else inherited).append(parameter)
    optimizer = torch.optim.AdamW([{"params": inherited, "lr": args.lr}, {"params": new, "lr": args.new_lr}], weight_decay=0.05)
    # Token-budget episodes activate only the pooling branch used by the
    # current sample (80/160/400/1024 tokens per view). In DDP, the other
    # branches legitimately have no gradient on that iteration.
    wrapped = (
        DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if ddp else model
    )
    output_dir = Path(args.output)
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        print(f"train_episodes={len(dataset)} seen_episodes={len(val_dataset) if val_dataset else 0} unseen_episodes={len(unseen_val_dataset) if unseen_val_dataset else 0} no_z={args.no_z} world_size={world_size}", flush=True)
    torch.manual_seed(args.seed + local_rank)
    global_step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        wrapped.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        train_pose_sums = torch.zeros(3, device=device, dtype=torch.float64)
        progress = tqdm(
            enumerate(loader), total=len(loader), desc=f"Train epoch {epoch}",
            dynamic_ncols=True, leave=True, disable=local_rank != 0,
        )
        for step, batch in progress:
            if epoch == start_epoch and step == 0 and local_rank == 0:
                print(f"first_batch_shapes={ {k: list(v.shape) for k, v in batch.items()} }", flush=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            amp_enabled = args.amp_dtype == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                view_token_count = _batch_scalar(batch, "view_token_count", args.view_token_count)
                support_token_budget = _batch_scalar(batch, "support_token_budget", args.support_token_budget)
                camera_rays = _query_camera_rays(batch["query_intrinsics"], args.image_size, 8, model.patch_count)
                gt_dirs = _gt_directions(batch["query_pose"], batch["query_intrinsics"], args.image_size, 8)
                gt_centers = batch["query_pose"][..., :3, 3]
                gt_moments = torch.cross(gt_centers[:, None].expand_as(gt_dirs), gt_dirs, dim=-1)
                global_stage = batch["stage_id"].reshape(-1) == 0
                if args.ray_diffusion:
                    x0 = torch.cat((gt_dirs, gt_moments / args.moment_scale), dim=-1)
                    timesteps = torch.randint(args.diffusion_train_steps, (x0.shape[0],), device=device)
                    noisy_rays = model.diffusion_schedule.q_sample(x0, timesteps)
                    result = wrapped(
                        batch["support_images"], batch["support_intrinsics"], batch["support_poses"],
                        batch["query_image"], batch["query_intrinsics"], batch.get("support_type_ids"),
                        batch.get("support_mask"), view_token_count=view_token_count,
                        support_token_budget=support_token_budget, noisy_rays=noisy_rays, timesteps=timesteps,
                    )
                    pose = solve_car_pose(result.directions, result.moments, camera_rays)
                    relative_loss = (
                        _relative_pose_loss(
                            result.relative_pose, batch["query_pose"], batch["support_poses"],
                            batch["support_mask"], batch["stage_id"],
                        ) if result.relative_pose is not None else 0.0
                    )
                    loss = (
                        F.mse_loss(result.raw_rays, x0)
                        + args.diffusion_ray_loss_weight * (
                            F.smooth_l1_loss(result.directions, gt_dirs)
                            + 0.5 * F.smooth_l1_loss(result.moments, gt_moments)
                        )
                        + args.diffusion_pose_loss_weight * _masked_pose_loss(pose, batch["query_pose"], global_stage)
                        + args.relative_pose_loss_weight * relative_loss
                    )
                else:
                    result = wrapped(
                        batch["support_images"], batch["support_intrinsics"], batch["support_poses"],
                        batch["query_image"], batch["query_intrinsics"], batch.get("support_type_ids"),
                        batch.get("support_mask"), view_token_count=view_token_count,
                        support_token_budget=support_token_budget,
                    )
                    pose = solve_car_pose(result.directions, result.moments, camera_rays)
                    relative_loss = (
                        _relative_pose_loss(
                            result.relative_pose, batch["query_pose"], batch["support_poses"],
                            batch["support_mask"], batch["stage_id"],
                        ) if result.relative_pose is not None else 0.0
                    )
                    loss = (
                        _masked_pose_loss(pose, batch["query_pose"], global_stage)
                        + F.smooth_l1_loss(result.directions, gt_dirs)
                        + 0.5 * F.smooth_l1_loss(result.moments, gt_moments)
                        + args.relative_pose_loss_weight * relative_loss
                    )
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                metrics = pose_metrics(pose.rotation.detach().float(), pose.center.detach().float(), batch["query_pose"])
                count = batch["query_pose"].shape[0]
                train_pose_sums += train_pose_sums.new_tensor([metrics["rotation_deg"] * count, metrics["center_error"] * count, count])
            epoch_loss_sum += loss.detach().item()
            epoch_steps += 1
            if local_rank == 0:
                progress.set_postfix(**_progress_postfix_train(
                    loss.item(), epoch_loss_sum / epoch_steps,
                    rotation_deg=float(train_pose_sums[0] / train_pose_sums[2]),
                    center_error=float(train_pose_sums[1] / train_pose_sums[2]),
                ))
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        if ddp:
            epoch_stats = torch.tensor(
                [epoch_loss_sum, float(epoch_steps)], device=device, dtype=torch.float64
            )
            dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_pose_sums, op=dist.ReduceOp.SUM)
            dist.barrier()
        else:
            epoch_stats = torch.tensor(
                [epoch_loss_sum, float(epoch_steps)], device=device, dtype=torch.float64
            )
        checkpoint = None
        if local_rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": float(epoch_stats[0] / epoch_stats[1].clamp_min(1.0)),
                "train_denoise_metrics": {
                    "rotation_deg": float(train_pose_sums[0] / train_pose_sums[2]),
                    "center_error": float(train_pose_sums[1] / train_pose_sums[2]),
                },
                "model_config": {
                    "no_z": args.no_z,
                    "seed": args.seed,
                    "ray_diffusion": args.ray_diffusion,
                    "diffusion_layers": args.diffusion_layers,
                    "diffusion_train_steps": args.diffusion_train_steps,
                    "moment_scale": args.moment_scale,
                    "relative_pose": args.relative_pose,
                    "relative_pose_loss_weight": args.relative_pose_loss_weight,
                    "relative_robust_center_scale": args.relative_robust_center_scale,
                    "local_refinement": args.local_refinement,
                },
            }
            print(f"epoch={epoch} train_denoise_metrics={checkpoint['train_denoise_metrics']}", flush=True)
            if checkpoint["loss"] < best_epoch_loss:
                best_epoch_loss = checkpoint["loss"]
                torch.save(checkpoint, output_dir / "trainbest.pt")
                print(f"epoch={epoch} trainbest={best_epoch_loss:.6f}", flush=True)
        # All ranks must run validation because it contains collective all_reduce calls.
        if train_probe is not None and (epoch + 1) % args.val_every == 0:
            probe_metrics, probe_by_stage = evaluate_validation(
                model, train_probe, device, args.image_size, desc=f"Train pure-noise probe epoch {epoch}",
                ddp=ddp, local_rank=local_rank, world_size=world_size,
                diffusion_sample_steps=args.diffusion_sample_steps if args.ray_diffusion else 0,
            )
            if local_rank == 0:
                checkpoint["train_pure_noise_metrics"] = probe_metrics
                checkpoint["train_pure_noise_metrics_by_stage"] = probe_by_stage
        if (val_dataset is not None or unseen_val_dataset is not None) and (epoch + 1) % args.val_every == 0:
            if val_dataset is not None:
                seen_metrics, seen_by_stage = evaluate_validation(
                    model, val_dataset, device, args.image_size, args.val_max_episodes,
                    desc=f"Validate seen epoch {epoch}", ddp=ddp, local_rank=local_rank,
                    world_size=world_size, batch_size=args.val_batch_size,
                    diffusion_sample_steps=args.diffusion_sample_steps if args.ray_diffusion else 0,
                )
                if local_rank == 0:
                    checkpoint["seen_val_metrics"] = seen_metrics
                    checkpoint["seen_val_metrics_by_stage"] = seen_by_stage
                    checkpoint["seen_val_score"] = stagewise_validation_score(seen_by_stage)
                    checkpoint["val_metrics"] = seen_metrics
                    checkpoint["val_score"] = checkpoint["seen_val_score"]
                    print(f"epoch={epoch} seen_val_score={checkpoint['seen_val_score']:.6f} seen_rotation={seen_metrics['rotation_deg']:.4f} seen_center={seen_metrics['center_error']:.6f}", flush=True)
                    if checkpoint["seen_val_score"] < best_val_score:
                        best_val_score = checkpoint["seen_val_score"]
                        torch.save(checkpoint, output_dir / "seenbest.pt")
                        torch.save(checkpoint, output_dir / "valbest.pt")
                        print(f"epoch={epoch} seen_valbest={best_val_score:.6f}", flush=True)
            if unseen_val_dataset is not None:
                unseen_metrics, unseen_by_stage = evaluate_validation(
                    model, unseen_val_dataset, device, args.image_size, args.unseen_val_max_episodes,
                    desc=f"Validate unseen epoch {epoch}", ddp=ddp, local_rank=local_rank,
                    world_size=world_size, batch_size=args.val_batch_size,
                    diffusion_sample_steps=args.diffusion_sample_steps if args.ray_diffusion else 0,
                )
                if local_rank == 0:
                    checkpoint["unseen_val_metrics"] = unseen_metrics
                    checkpoint["unseen_val_metrics_by_stage"] = unseen_by_stage
                    checkpoint["unseen_val_score"] = stagewise_validation_score(unseen_by_stage)
                    print(f"epoch={epoch} unseen_val_score={checkpoint['unseen_val_score']:.6f} unseen_rotation={unseen_metrics['rotation_deg']:.4f} unseen_center={unseen_metrics['center_error']:.6f}", flush=True)
                    if checkpoint["unseen_val_score"] < best_unseen_score:
                        best_unseen_score = checkpoint["unseen_val_score"]
                        torch.save(checkpoint, output_dir / "unseenbest.pt")
                        print(f"epoch={epoch} unseen_valbest={best_unseen_score:.6f}", flush=True)
        if local_rank == 0:
            history = {k: v for k, v in checkpoint.items() if k != "model"}
            with (output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(history) + "\n")
            torch.save(checkpoint, output_dir / "last.pt")
        if ddp:
            dist.barrier()
        if args.max_steps and global_step >= args.max_steps:
            break
    if ddp:
        dist.destroy_process_group()


def _query_camera_rays(intrinsics, image_size, patch_size, count):
    h = w = image_size
    ys = (torch.arange(h // patch_size, device=intrinsics.device) + .5) * patch_size
    xs = (torch.arange(w // patch_size, device=intrinsics.device) + .5) * patch_size
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx, gy), -1).reshape(-1, 2)
    hom = torch.cat((grid, torch.ones_like(grid[:, :1])), -1).expand(intrinsics.shape[0], -1, -1)
    return F.normalize(torch.linalg.solve(intrinsics[:, None], hom[..., None]).squeeze(-1), dim=-1)


def _gt_directions(poses, intrinsics, image_size, patch_size):
    local = _query_camera_rays(intrinsics, image_size, patch_size, poses.shape[0])
    return F.normalize((poses[:, None, :3, :3] @ local[..., None]).squeeze(-1), dim=-1)


if __name__ == "__main__":
    main()
