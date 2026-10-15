"""Compact LVSM-compatible encoder and query-to-world-ray decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .solver import (
    RelativePosePrediction,
    axis_angle_to_matrix,
    fuse_relative_pose_candidates,
    transform_pluecker_c2w_to_local,
)
from .transformer import QKNormCrossAttention, QKNormTransformerBlock


@dataclass(frozen=True)
class RayOutput:
    latent: torch.Tensor
    directions: torch.Tensor
    moments: torch.Tensor
    moments_raw: torch.Tensor | None = None
    relative_pose: RelativePosePrediction | None = None


def _patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, v, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    patches = images.reshape(b * v, c, h, w).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(b * v, -1, c * patch_size * patch_size)


def _patch_centers(h: int, w: int, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = (torch.arange(h // patch_size, device=device, dtype=dtype) + 0.5) * patch_size
    xs = (torch.arange(w // patch_size, device=device, dtype=dtype) + 0.5) * patch_size
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx.flatten(), gy.flatten()), dim=-1)


def _camera_rays(pixels: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    solved = torch.linalg.solve(intrinsics.unsqueeze(-3), homogeneous.unsqueeze(-1)).squeeze(-1)
    return F.normalize(solved, dim=-1)


class ViewTokenPool(nn.Module):
    """Pool patch tokens independently inside each view.

    The input is [B, views, patches, dim].  No token from one view can
    attend to a token from another view.  When target_tokens equals the
    original patch count, this is an exact identity path so the existing
    8-view full-LVSM checkpoint has the same fine-stage computation.
    """

    def __init__(self, dim: int, patch_count: int, target_tokens: int) -> None:
        super().__init__()
        if dim <= 0 or patch_count <= 0 or target_tokens <= 0 or target_tokens > patch_count:
            raise ValueError("invalid token pooling dimensions")
        self.dim = dim
        self.patch_count = patch_count
        self.target_tokens = target_tokens
        self.identity = target_tokens == patch_count
        if not self.identity:
            self.queries = nn.Parameter(torch.randn(target_tokens, dim) * 0.02)
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("view tokens must be [B, views, patches, dim]")
        b, views, patches, dim = tokens.shape
        if patches != self.patch_count or dim != self.dim:
            raise ValueError("view token shape does not match the pool")
        if self.identity:
            return tokens
        flat = tokens.reshape(b * views, patches, dim)
        query = self.queries.unsqueeze(0).expand(b * views, -1, -1)
        query = self.q_proj(query)
        key = self.k_proj(flat)
        value = self.v_proj(flat)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (dim ** 0.5)
        weights = scores.softmax(dim=-1)
        pooled = torch.matmul(weights, value)
        return self.out_proj(pooled).reshape(b, views, self.target_tokens, dim)


class LVSMNeuLoc(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        dim: int = 768,
        latent_count: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        heads: int = 12,
        gradient_checkpointing: bool = False,
        pool_token_counts: tuple[int, ...] = (80, 160, 400, 1024),
        relative_pose: bool = False,
        relative_robust_center_scale: float = 0.25,
        local_refinement: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size or dim % heads or latent_count <= 0:
            raise ValueError("invalid image, transformer, or latent dimensions")
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.gradient_checkpointing = gradient_checkpointing
        self.relative_pose_enabled = bool(relative_pose)
        self.local_refinement_enabled = bool(local_refinement)
        if relative_robust_center_scale <= 0:
            raise ValueError("relative_robust_center_scale must be positive")
        self.relative_robust_center_scale = float(relative_robust_center_scale)
        self.patch_count = (image_size // patch_size) ** 2
        patch_dim = 9 * patch_size * patch_size
        self.image_tokenizer = nn.Linear(patch_dim, dim, bias=False)
        self.query_tokenizer = nn.Linear(patch_dim, dim, bias=False)
        self.token_type_embedding = nn.Embedding(3, dim)
        self.latent_tokens = nn.Parameter(torch.randn(latent_count, dim) * 0.02)
        self.encoder = nn.ModuleList([QKNormTransformerBlock(dim, heads) for _ in range(encoder_layers)])
        self.decoder = nn.ModuleList([QKNormTransformerBlock(dim, heads) for _ in range(decoder_layers)])
        self.decoder_norm = nn.LayerNorm(dim)
        self.direction_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
        self.moment_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
        if self.relative_pose_enabled:
            self.relative_pose_fusion = nn.Sequential(
                nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(),
            )
            self.relative_pose_head = nn.Linear(dim, 6)
            self.relative_confidence_head = nn.Linear(dim, 1)
        if self.local_refinement_enabled:
            self.refinement_stage_embedding = nn.Embedding(3, dim)
            self.refinement_memory_norm = nn.LayerNorm(dim)
            self.refinement_query_norm = nn.LayerNorm(dim)
            self.refinement_cross_attention = nn.ModuleList(
                [QKNormCrossAttention(dim, heads) for _ in range(decoder_layers)]
            )
            self.refinement_confidence_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        requested_pool_counts = tuple(int(k) for k in pool_token_counts)
        if any(k <= 0 for k in requested_pool_counts):
            raise ValueError("pool token counts must be positive")
        valid_pool_counts = tuple(dict.fromkeys(k for k in requested_pool_counts if k <= self.patch_count))
        if self.patch_count not in valid_pool_counts:
            valid_pool_counts = (*valid_pool_counts, self.patch_count)
        self.view_token_pools = nn.ModuleDict({
            str(k): ViewTokenPool(dim, self.patch_count, k) for k in valid_pool_counts
        })

    def _run(self, blocks: nn.ModuleList, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in blocks:
            if self.training and self.gradient_checkpointing:
                if key_padding_mask is None:
                    tokens = checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = checkpoint(block, tokens, key_padding_mask, use_reentrant=False)
            else:
                tokens = block(tokens, key_padding_mask=key_padding_mask)
        return tokens

    def _pool_support_tokens(self, image_tokens: torch.Tensor, target_tokens: int | None) -> torch.Tensor:
        if target_tokens is None or target_tokens == self.patch_count:
            return image_tokens
        key = str(int(target_tokens))
        if key not in self.view_token_pools:
            raise ValueError(
                f"unsupported view token count {target_tokens}; available={tuple(self.view_token_pools.keys())}"
            )
        pool = self.view_token_pools[key]
        return pool(image_tokens)

    def encode_scene(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
        return_view_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if support_images.ndim != 5 or support_intrinsics.ndim != 4 or support_poses.ndim != 4:
            raise ValueError("invalid support tensor ranks")
        b, views, _, h, w = support_images.shape
        if support_intrinsics.shape[:2] != (b, views) or support_poses.shape != (b, views, 4, 4):
            raise ValueError("support metadata shapes must agree")
        if support_mask is None:
            support_mask = torch.ones((b, views), dtype=torch.bool, device=support_images.device)
        if support_mask.shape != (b, views):
            raise ValueError("support_mask must be [B, views]")
        support_mask = support_mask.to(dtype=torch.bool, device=support_images.device)
        pixels = _patch_centers(h, w, self.patch_size, support_images.device, support_images.dtype)
        pixels = pixels.view(1, 1, -1, 2).expand(b, views, -1, -1)
        local = _camera_rays(pixels, support_intrinsics)
        rotations = support_poses[..., :3, :3]
        centers = support_poses[..., :3, 3]
        world = F.normalize((rotations.unsqueeze(-3) @ local.unsqueeze(-1)).squeeze(-1), dim=-1)
        moments = torch.cross(centers.unsqueeze(-2).expand_as(world), world, dim=-1)
        rgb = _patchify(support_images, self.patch_size).reshape(b, views, -1, 3 * self.patch_size * self.patch_size)
        ray_repeat = self.patch_size * self.patch_size
        world_patch = world.unsqueeze(-2).expand(-1, -1, -1, ray_repeat, -1).reshape(b, views, -1, 3 * ray_repeat)
        moment_patch = moments.unsqueeze(-2).expand(-1, -1, -1, ray_repeat, -1).reshape(b, views, -1, 3 * ray_repeat)
        cond = torch.cat((rgb, world_patch, moment_patch), dim=-1)
        image_tokens = self.image_tokenizer(cond.reshape(b * views, -1, cond.shape[-1])).reshape(b, views, self.patch_count, self.dim)
        if support_type_ids is None:
            support_type_ids = torch.ones((b, views), dtype=torch.long, device=support_images.device)
        if support_type_ids.shape != (b, views):
            raise ValueError("support_type_ids must be [B, views]")
        type_tokens = self.token_type_embedding(support_type_ids).unsqueeze(2)
        image_tokens = image_tokens + type_tokens
        # This per-view descriptor retains the identity of each support.  The
        # flattened encoder latent is still shared with the original ray head.
        view_features = image_tokens.mean(dim=2)
        image_tokens = self._pool_support_tokens(image_tokens, view_token_count)
        per_view_tokens = image_tokens.shape[2]
        image_tokens = image_tokens.reshape(b, views * per_view_tokens, self.dim)
        view_valid = support_mask.unsqueeze(-1).expand(-1, -1, per_view_tokens)
        support_valid = view_valid.reshape(b, views * per_view_tokens)
        if support_token_budget is not None:
            if support_token_budget <= 0:
                raise ValueError("support_token_budget must be positive")
            if image_tokens.shape[1] > support_token_budget:
                raise ValueError(
                    f"support tokens {image_tokens.shape[1]} exceed budget {support_token_budget}"
                )
            pad = support_token_budget - image_tokens.shape[1]
            if pad:
                image_tokens = F.pad(image_tokens, (0, 0, 0, pad))
                support_valid = F.pad(support_valid, (0, pad), value=False)
        latent = self.latent_tokens.unsqueeze(0).expand(b, -1, -1)
        token_mask = torch.cat(
            (
                torch.ones((b, latent.shape[1]), dtype=torch.bool, device=support_images.device),
                support_valid,
            ),
            dim=1,
        )
        latent = self._run(self.encoder, torch.cat((latent, image_tokens), dim=1), key_padding_mask=token_mask)[:, : latent.shape[1]]
        if return_view_features:
            return latent, view_features
        return latent

    def decode_query_features(
        self, latent: torch.Tensor, query_image: torch.Tensor, query_intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Decode image and scene conditioning into one token per query patch."""
        if query_image.ndim != 4 or query_intrinsics.shape != (query_image.shape[0], 3, 3):
            raise ValueError("invalid query image or intrinsics")
        b, _, h, w = query_image.shape
        pixels = _patch_centers(h, w, self.patch_size, query_image.device, query_image.dtype)
        local = _camera_rays(pixels.unsqueeze(0).expand(b, -1, -1), query_intrinsics)
        rgb = _patchify(query_image.unsqueeze(1), self.patch_size).squeeze(0 if b == 1 else 1)
        if rgb.ndim == 2:
            rgb = rgb.unsqueeze(0)
        zeros = torch.zeros_like(local)
        ray_repeat = self.patch_size * self.patch_size
        local_patch = local.unsqueeze(-2).expand(-1, -1, ray_repeat, -1).reshape(b, -1, 3 * ray_repeat)
        zero_patch = zeros.unsqueeze(-2).expand(-1, -1, ray_repeat, -1).reshape(b, -1, 3 * ray_repeat)
        cond = torch.cat((rgb, local_patch, zero_patch), dim=-1)
        query_tokens = self.query_tokenizer(cond) + self.token_type_embedding.weight[2].view(1, 1, -1)
        decoder_input = self.decoder_norm(torch.cat((query_tokens, latent), dim=1))
        return self._run(self.decoder, decoder_input)[:, : self.patch_count]

    def _stage_tokens(self, batch: int, stage_index: int | torch.Tensor, device: torch.device) -> torch.Tensor:
        if not self.local_refinement_enabled:
            raise RuntimeError("local refinement is disabled")
        if isinstance(stage_index, int):
            stage = torch.full((batch,), stage_index, dtype=torch.long, device=device)
        else:
            stage = stage_index.to(device=device, dtype=torch.long).reshape(-1)
        if stage.shape != (batch,) or torch.any(stage < 0) or torch.any(stage >= 3):
            raise ValueError("stage_index must be Semi/Dense/Fine encoded as [B] values in [0, 2]")
        return self.refinement_stage_embedding(stage)

    def encode_local_support_memory(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        current_pose: torch.Tensor,
        *,
        stage_index: int | torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build support memory with rays re-expressed in the current query frame."""
        if not self.local_refinement_enabled:
            raise RuntimeError("local refinement is disabled")
        if support_images.ndim != 5 or support_intrinsics.ndim != 4 or support_poses.ndim != 4:
            raise ValueError("invalid support tensor ranks")
        b, views, _, h, w = support_images.shape
        if support_intrinsics.shape[:2] != (b, views) or support_poses.shape != (b, views, 4, 4):
            raise ValueError("support metadata shapes must agree")
        if current_pose.shape != (b, 4, 4):
            raise ValueError("current_pose must be [B, 4, 4]")
        if support_mask is None:
            support_mask = torch.ones((b, views), dtype=torch.bool, device=support_images.device)
        support_mask = support_mask.to(device=support_images.device, dtype=torch.bool)
        if support_mask.shape != (b, views) or not torch.all(support_mask.any(dim=-1)):
            raise ValueError("support_mask must contain a valid support for every sample")
        pixels = _patch_centers(h, w, self.patch_size, support_images.device, support_images.dtype)
        pixels = pixels.view(1, 1, -1, 2).expand(b, views, -1, -1)
        local_camera = _camera_rays(pixels, support_intrinsics)
        rotations, centers = support_poses[..., :3, :3], support_poses[..., :3, 3]
        world_directions = F.normalize((rotations.unsqueeze(-3) @ local_camera.unsqueeze(-1)).squeeze(-1), dim=-1)
        world_moments = torch.cross(centers.unsqueeze(-2).expand_as(world_directions), world_directions, dim=-1)
        local_directions, local_moments = transform_pluecker_c2w_to_local(
            world_directions.reshape(b, -1, 3), world_moments.reshape(b, -1, 3), current_pose,
        )
        local_directions = local_directions.reshape_as(world_directions)
        local_moments = local_moments.reshape_as(world_moments)
        rgb = _patchify(support_images, self.patch_size).reshape(b, views, -1, 3 * self.patch_size * self.patch_size)
        repeat = self.patch_size * self.patch_size
        direction_patch = local_directions.unsqueeze(-2).expand(-1, -1, -1, repeat, -1).reshape(b, views, -1, 3 * repeat)
        moment_patch = local_moments.unsqueeze(-2).expand(-1, -1, -1, repeat, -1).reshape(b, views, -1, 3 * repeat)
        tokens = self.image_tokenizer(torch.cat((rgb, direction_patch, moment_patch), dim=-1).reshape(b * views, -1, 9 * repeat))
        tokens = tokens.reshape(b, views, self.patch_count, self.dim)
        if support_type_ids is None:
            support_type_ids = torch.ones((b, views), dtype=torch.long, device=support_images.device)
        if support_type_ids.shape != (b, views):
            raise ValueError("support_type_ids must be [B, views]")
        stage = self._stage_tokens(b, stage_index, support_images.device)
        tokens = tokens + self.token_type_embedding(support_type_ids).unsqueeze(2) + stage[:, None, None]
        tokens = self._pool_support_tokens(tokens, view_token_count)
        per_view = tokens.shape[2]
        memory = tokens.reshape(b, views * per_view, self.dim)
        mask = support_mask.unsqueeze(-1).expand(-1, -1, per_view).reshape(b, views * per_view)
        if support_token_budget is not None:
            if support_token_budget < memory.shape[1]:
                raise ValueError("support memory exceeds support_token_budget")
            padding = support_token_budget - memory.shape[1]
            if padding:
                memory = F.pad(memory, (0, 0, 0, padding))
                mask = F.pad(mask, (0, padding), value=False)
        return memory, mask

    def decode_refinement_features(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        *,
        stage_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserved query decoder plus direct cross-attention to support memory."""
        if not self.local_refinement_enabled:
            raise RuntimeError("local refinement is disabled")
        b, _, h, w = query_image.shape
        if query_intrinsics.shape != (b, 3, 3) or memory_mask.shape != memory.shape[:2]:
            raise ValueError("invalid query geometry or support memory mask")
        pixels = _patch_centers(h, w, self.patch_size, query_image.device, query_image.dtype)
        camera_directions = _camera_rays(pixels.unsqueeze(0).expand(b, -1, -1), query_intrinsics)
        rgb = _patchify(query_image.unsqueeze(1), self.patch_size)
        zeros = torch.zeros_like(camera_directions)
        repeat = self.patch_size * self.patch_size
        direction_patch = camera_directions.unsqueeze(-2).expand(-1, -1, repeat, -1).reshape(b, -1, 3 * repeat)
        zero_patch = zeros.unsqueeze(-2).expand(-1, -1, repeat, -1).reshape(b, -1, 3 * repeat)
        query = self.query_tokenizer(torch.cat((rgb, direction_patch, zero_patch), dim=-1))
        query = query + self.token_type_embedding.weight[2].view(1, 1, -1) + self._stage_tokens(b, stage_index, query.device)[:, None]
        query = self.decoder_norm(query)
        memory = self.refinement_memory_norm(memory)
        for block, cross in zip(self.decoder, self.refinement_cross_attention):
            query = block(query)
            query = query + cross(self.refinement_query_norm(query), memory, memory_mask)
        return query, camera_directions

    def predict_relative_pose(
        self,
        query_features: torch.Tensor,
        support_view_features: torch.Tensor,
        support_poses: torch.Tensor,
        support_mask: torch.Tensor | None,
    ) -> RelativePosePrediction:
        if not self.relative_pose_enabled:
            raise RuntimeError("relative pose head is disabled")
        if support_view_features.ndim != 3:
            raise ValueError("support_view_features must be [B, views, dim]")
        query_summary = query_features.mean(dim=1, keepdim=True).expand(-1, support_view_features.shape[1], -1)
        pair_features = self.relative_pose_fusion(torch.cat((support_view_features, query_summary), dim=-1))
        raw_pose = self.relative_pose_head(pair_features)
        rotation = axis_angle_to_matrix(raw_pose[..., :3])
        translation = raw_pose[..., 3:]
        logits = self.relative_confidence_head(pair_features).squeeze(-1)
        fused, weights = fuse_relative_pose_candidates(
            support_poses, rotation, translation, logits, support_mask,
            robust_center_scale=self.relative_robust_center_scale,
        )
        support_rotation = support_poses[..., :3, :3].float()
        support_center = support_poses[..., :3, 3].float()
        candidate_rotation = support_rotation @ rotation.float()
        candidate_center = support_center + (support_rotation @ translation.float().unsqueeze(-1)).squeeze(-1)
        return RelativePosePrediction(
            rotation=rotation, translation=translation, confidence_logits=logits,
            candidate_rotation=candidate_rotation, candidate_center=candidate_center,
            weights=weights, fused=fused,
        )

    def decode_query(
        self,
        latent: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        support_poses: torch.Tensor | None = None,
        support_view_features: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
    ) -> RayOutput:
        output = self.decode_query_features(latent, query_image, query_intrinsics)
        directions = F.normalize(self.direction_head(output), dim=-1)
        moments_raw = self.moment_head(output)
        moments = moments_raw - (moments_raw * directions).sum(-1, keepdim=True) * directions
        relative = None
        if self.relative_pose_enabled:
            if support_poses is None or support_view_features is None:
                raise ValueError("relative pose prediction needs support poses and per-view features")
            relative = self.predict_relative_pose(output, support_view_features, support_poses, support_mask)
        return RayOutput(latent=latent, directions=directions, moments=moments, moments_raw=moments_raw, relative_pose=relative)

    def forward(self, support_images, support_intrinsics, support_poses, query_image, query_intrinsics, support_type_ids=None, support_mask=None, view_token_count=None, support_token_budget=None):
        encoded = self.encode_scene(
            support_images,
            support_intrinsics,
            support_poses,
            support_type_ids=support_type_ids,
            support_mask=support_mask,
            view_token_count=view_token_count,
            support_token_budget=support_token_budget,
            return_view_features=self.relative_pose_enabled,
        )
        if self.relative_pose_enabled:
            latent, view_features = encoded
        else:
            latent, view_features = encoded, None
        return self.decode_query(
            latent, query_image, query_intrinsics, support_poses=support_poses,
            support_view_features=view_features, support_mask=support_mask,
        )
