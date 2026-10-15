"""Conditional full-ray diffusion model built on the LVSM scene encoder."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .diffusion import (
    RayDiffusionSchedule,
    canonicalize_rays,
    decode_local_ray_residual,
    sinusoidal_timestep_embedding,
)
from .model import LVSMNeuLoc, RayOutput
from .solver import CarPose, RelativePosePrediction, weighted_solve_car_pose
from .transformer import QKNormTransformerBlock


@dataclass(frozen=True)
class RayDiffusionOutput:
    """Clean-ray prediction at one diffusion timestep."""

    latent: torch.Tensor
    raw_rays: torch.Tensor
    directions: torch.Tensor
    moments: torch.Tensor
    relative_pose: RelativePosePrediction | None = None


@dataclass(frozen=True)
class LocalRefinementOutput:
    """One identity-centred local ray-diffusion pose update."""

    raw_residual: torch.Tensor
    directions: torch.Tensor
    moments: torch.Tensor
    confidence: torch.Tensor
    delta: CarPose


class RayDiffusionNeuLoc(LVSMNeuLoc):
    """Generate full query Pluecker-ray maps conditioned on LVSM features."""

    def __init__(
        self,
        *args,
        diffusion_layers: int = 4,
        diffusion_steps: int = 50,
        moment_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if diffusion_layers <= 0:
            raise ValueError("diffusion_layers must be positive")
        if moment_scale <= 0:
            raise ValueError("moment_scale must be positive")
        self.moment_scale = float(moment_scale)
        self.diffusion_schedule = RayDiffusionSchedule(diffusion_steps)
        self.diffusion_ray_tokenizer = nn.Linear(6, self.dim, bias=False)
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim),
        )
        self.diffusion_blocks = nn.ModuleList(
            [QKNormTransformerBlock(self.dim, self.decoder[0].attn.heads) for _ in range(diffusion_layers)]
        )
        self.diffusion_norm = nn.LayerNorm(self.dim)
        self.diffusion_head = nn.Linear(self.dim, 6)
        if self.local_refinement_enabled:
            # Refinement is initialized from the Global denoiser but owns its
            # parameters, so local fine-tuning cannot degrade Global sampling.
            self.local_diffusion_ray_tokenizer = copy.deepcopy(self.diffusion_ray_tokenizer)
            self.local_diffusion_time_mlp = copy.deepcopy(self.diffusion_time_mlp)
            self.local_diffusion_blocks = copy.deepcopy(self.diffusion_blocks)
            self.local_diffusion_norm = copy.deepcopy(self.diffusion_norm)
            self.local_diffusion_head = copy.deepcopy(self.diffusion_head)

    def initialize_local_diffusion_from_global(self) -> None:
        """Copy a loaded Global denoiser into the independent local branch."""
        if not self.local_refinement_enabled:
            raise RuntimeError("local refinement is disabled")
        self.local_diffusion_ray_tokenizer.load_state_dict(self.diffusion_ray_tokenizer.state_dict())
        self.local_diffusion_time_mlp.load_state_dict(self.diffusion_time_mlp.state_dict())
        self.local_diffusion_blocks.load_state_dict(self.diffusion_blocks.state_dict())
        self.local_diffusion_norm.load_state_dict(self.diffusion_norm.state_dict())
        self.local_diffusion_head.load_state_dict(self.diffusion_head.state_dict())

    def condition_features(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run expensive LVSM conditioning exactly once for a query episode."""
        encoded = self.encode_scene(
            support_images,
            support_intrinsics,
            support_poses,
            support_type_ids=support_type_ids,
            support_mask=support_mask,
            view_token_count=view_token_count,
            support_token_budget=support_token_budget,
            return_view_features=self.relative_pose_enabled,
            return_memory_mask=self.no_z_enabled,
        )
        memory_mask = None
        if self.no_z_enabled:
            *encoded, memory_mask = encoded
            encoded = tuple(encoded) if self.relative_pose_enabled else encoded[0]
        if self.relative_pose_enabled:
            latent, view_features = encoded
        else:
            latent, view_features = encoded, None
        return latent, self.decode_query_features(latent, query_image, query_intrinsics, memory_mask), view_features

    def denoise_from_features(
        self, query_features: torch.Tensor, noisy_rays: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predict clean normalized six-channel rays from noise and cached conditions."""
        if query_features.ndim != 3 or query_features.shape[-1] != self.dim:
            raise ValueError("query_features must be [B, patches, dim]")
        if noisy_rays.shape != (*query_features.shape[:2], 6):
            raise ValueError("noisy_rays must be [B, patches, 6]")
        if timesteps.shape != (query_features.shape[0],):
            raise ValueError("timesteps must be [B]")
        time = self.diffusion_time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.dim).to(dtype=query_features.dtype)
        ).unsqueeze(1)
        tokens = query_features + self.diffusion_ray_tokenizer(noisy_rays) + time
        tokens = self._run(self.diffusion_blocks, tokens)
        return self.diffusion_head(self.diffusion_norm(tokens))

    def denoise_local_from_features(
        self, query_features: torch.Tensor, noisy_residual: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Denoise identity-centred residual rays with local-only parameters."""
        if not self.local_refinement_enabled:
            raise RuntimeError("local refinement is disabled")
        if query_features.ndim != 3 or query_features.shape[-1] != self.dim:
            raise ValueError("query_features must be [B, patches, dim]")
        if noisy_residual.shape != (*query_features.shape[:2], 6):
            raise ValueError("noisy_residual must be [B, patches, 6]")
        if timesteps.shape != (query_features.shape[0],):
            raise ValueError("timesteps must be [B]")
        time = self.local_diffusion_time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.dim).to(dtype=query_features.dtype)
        ).unsqueeze(1)
        tokens = query_features + self.local_diffusion_ray_tokenizer(noisy_residual) + time
        tokens = self._run(self.local_diffusion_blocks, tokens)
        return self.local_diffusion_head(self.local_diffusion_norm(tokens))

    def denoise(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        noisy_rays: torch.Tensor,
        timesteps: torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
    ) -> RayDiffusionOutput:
        latent, features, view_features = self.condition_features(
            support_images, support_intrinsics, support_poses, query_image, query_intrinsics,
            support_type_ids=support_type_ids, support_mask=support_mask,
            view_token_count=view_token_count, support_token_budget=support_token_budget,
        )
        raw_rays = self.denoise_from_features(features, noisy_rays, timesteps)
        directions, moments = canonicalize_rays(raw_rays, self.moment_scale)
        relative = None
        if self.relative_pose_enabled:
            relative = self.predict_relative_pose(features, view_features, support_poses, support_mask)
        return RayDiffusionOutput(latent, raw_rays, directions, moments, relative)

    def forward(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
        *,
        noisy_rays: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> RayDiffusionOutput:
        """DDP-compatible full conditioning and denoising forward path."""
        return self.denoise(
            support_images, support_intrinsics, support_poses, query_image, query_intrinsics,
            noisy_rays, timesteps, support_type_ids=support_type_ids, support_mask=support_mask,
            view_token_count=view_token_count, support_token_budget=support_token_budget,
        )

    @torch.inference_mode()
    def sample(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
        sample_steps: int = 16,
        generator: torch.Generator | None = None,
    ) -> RayOutput:
        """Sample a query ray field by deterministic DDIM from Gaussian noise."""
        if sample_steps < 2 or sample_steps > self.diffusion_schedule.steps:
            raise ValueError("sample_steps must be in [2, diffusion_steps]")
        latent, features, view_features = self.condition_features(
            support_images, support_intrinsics, support_poses, query_image, query_intrinsics,
            support_type_ids=support_type_ids, support_mask=support_mask,
            view_token_count=view_token_count, support_token_budget=support_token_budget,
        )
        raw_rays = torch.randn(
            features.shape[0], features.shape[1], 6,
            device=features.device, dtype=features.dtype, generator=generator,
        )
        schedule_times = torch.linspace(
            self.diffusion_schedule.steps - 1, 0, sample_steps, device=features.device,
        ).round().to(torch.long).unique_consecutive()
        predicted_x0 = raw_rays
        for index, timestep in enumerate(schedule_times):
            current = torch.full((features.shape[0],), timestep, device=features.device, dtype=torch.long)
            predicted_x0 = self.denoise_from_features(features, raw_rays, current)
            previous_value = schedule_times[index + 1] if index + 1 < len(schedule_times) else -1
            previous = torch.full_like(current, previous_value)
            raw_rays = self.diffusion_schedule.ddim_step(raw_rays, predicted_x0, current, previous)
        directions, moments = canonicalize_rays(predicted_x0, self.moment_scale)
        relative = None
        if self.relative_pose_enabled:
            relative = self.predict_relative_pose(features, view_features, support_poses, support_mask)
        return RayOutput(
            latent=latent, directions=directions, moments=moments,
            moments_raw=predicted_x0[..., 3:], relative_pose=relative,
        )

    def refine_once(
        self,
        support_images: torch.Tensor,
        support_intrinsics: torch.Tensor,
        support_poses: torch.Tensor,
        query_image: torch.Tensor,
        query_intrinsics: torch.Tensor,
        *,
        current_pose: torch.Tensor,
        stage_index: int | torch.Tensor,
        support_type_ids: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        view_token_count: int | None = None,
        support_token_budget: int | None = None,
        sample_steps: int = 16,
        generator: torch.Generator | None = None,
    ) -> LocalRefinementOutput:
        """Infer a local c2w delta from current-query-frame support memory."""
        if not self.local_refinement_enabled:
            raise RuntimeError("refine_once requires local_refinement=True")
        if sample_steps < 2 or sample_steps > self.diffusion_schedule.steps:
            raise ValueError("sample_steps must be in [2, diffusion_steps]")
        memory, memory_mask = self.encode_local_support_memory(
            support_images, support_intrinsics, support_poses, current_pose,
            stage_index=stage_index, support_type_ids=support_type_ids, support_mask=support_mask,
            view_token_count=view_token_count, support_token_budget=support_token_budget,
        )
        features, base_directions = self.decode_refinement_features(
            memory, memory_mask, query_image, query_intrinsics, stage_index=stage_index,
        )
        residual = torch.randn(
            features.shape[0], features.shape[1], 6,
            device=features.device, dtype=features.dtype, generator=generator,
        )
        schedule_times = torch.linspace(
            self.diffusion_schedule.steps - 1, 0, sample_steps, device=features.device,
        ).round().to(torch.long).unique_consecutive()
        predicted_x0 = residual
        for index, timestep in enumerate(schedule_times):
            current = torch.full((features.shape[0],), timestep, device=features.device, dtype=torch.long)
            predicted_x0 = self.denoise_local_from_features(features, residual, current)
            previous_value = schedule_times[index + 1] if index + 1 < len(schedule_times) else -1
            residual = self.diffusion_schedule.ddim_step(
                residual, predicted_x0, current, torch.full_like(current, previous_value),
            )
        directions, moments = decode_local_ray_residual(base_directions, predicted_x0, self.moment_scale)
        confidence = torch.sigmoid(self.refinement_confidence_head(features).squeeze(-1))
        delta = weighted_solve_car_pose(directions, moments, base_directions, confidence)
        return LocalRefinementOutput(predicted_x0, directions, moments, confidence, delta)
