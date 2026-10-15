"""DDPM/DDIM utilities for geometrically valid Pluecker-ray diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the standard transformer timestep embedding for `[B]` integer times."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be [B]")
    if dim <= 0:
        raise ValueError("embedding dimension must be positive")
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def canonicalize_rays(raw_rays: torch.Tensor, moment_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Map six raw channels to unit directions and valid world Pluecker moments."""
    if raw_rays.ndim < 2 or raw_rays.shape[-1] != 6:
        raise ValueError("raw_rays must have shape [..., 6]")
    if moment_scale <= 0:
        raise ValueError("moment_scale must be positive")
    directions = F.normalize(raw_rays[..., :3], dim=-1)
    moments = raw_rays[..., 3:] * moment_scale
    moments = moments - (moments * directions).sum(dim=-1, keepdim=True) * directions
    return directions, moments


def encode_local_ray_residual(
    base_directions: torch.Tensor,
    target_directions: torch.Tensor,
    target_moments: torch.Tensor,
    moment_scale: float = 1.0,
) -> torch.Tensor:
    """Encode target Pluecker rays as identity-centred local residuals.

    The first three channels are the logarithmic map on the unit sphere at
    ``base_directions``.  The remaining channels are the local moment scaled
    for diffusion.  Thus the identity query pose has an exact all-zero target.
    """
    if base_directions.shape != target_directions.shape or base_directions.shape != target_moments.shape:
        raise ValueError("base directions, target directions, and target moments must have identical shapes")
    if base_directions.ndim < 2 or base_directions.shape[-1] != 3:
        raise ValueError("ray tensors must have shape [..., 3]")
    if moment_scale <= 0:
        raise ValueError("moment_scale must be positive")
    base = F.normalize(base_directions, dim=-1)
    target = F.normalize(target_directions, dim=-1)
    cosine = (base * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = target - cosine * base
    sine = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    angle = torch.acos(cosine)
    direction_residual = torch.where(
        sine > 1e-6,
        tangent * (angle / sine.clamp_min(1e-6)),
        torch.zeros_like(tangent),
    )
    moment = target_moments - (target_moments * target).sum(dim=-1, keepdim=True) * target
    return torch.cat((direction_residual, moment / moment_scale), dim=-1)


def decode_local_ray_residual(
    base_directions: torch.Tensor,
    residual: torch.Tensor,
    moment_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode tangent-space residuals to valid local Pluecker rays."""
    if residual.shape != (*base_directions.shape[:-1], 6):
        raise ValueError("residual must have shape base_directions.shape[:-1] + (6,)")
    if base_directions.ndim < 2 or base_directions.shape[-1] != 3:
        raise ValueError("base_directions must have shape [..., 3]")
    if moment_scale <= 0:
        raise ValueError("moment_scale must be positive")
    base = F.normalize(base_directions, dim=-1)
    tangent = residual[..., :3]
    tangent = tangent - (tangent * base).sum(dim=-1, keepdim=True) * base
    angle = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    unit_tangent = tangent / angle.clamp_min(1e-6)
    directions = torch.where(
        angle > 1e-6,
        torch.cos(angle) * base + torch.sin(angle) * unit_tangent,
        base,
    )
    directions = F.normalize(directions, dim=-1)
    moments = residual[..., 3:] * moment_scale
    moments = moments - (moments * directions).sum(dim=-1, keepdim=True) * directions
    return directions, moments


class RayDiffusionSchedule(nn.Module):
    """Linear DDPM schedule with deterministic DDIM transitions."""

    def __init__(self, steps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if steps < 2:
            raise ValueError("steps must be at least two")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.steps = steps
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != reference.shape[0]:
            raise ValueError("timesteps must be [B] matching the ray batch")
        if timesteps.dtype != torch.long:
            raise ValueError("timesteps must use torch.long")
        if (timesteps < 0).any() or (timesteps >= self.steps).any():
            raise ValueError("timesteps are outside the schedule")
        return values.gather(0, timesteps).view(reference.shape[0], *([1] * (reference.ndim - 1))).to(reference.dtype)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the DDPM forward process to clean normalized ray channels."""
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0")
        return self._extract(self.sqrt_alphas_cumprod, timesteps, x0) * x0 + self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x0
        ) * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        predicted_x0: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Take one deterministic DDIM step; use -1 as the terminal clean state."""
        if x_t.shape != predicted_x0.shape:
            raise ValueError("x_t and predicted_x0 must have the same shape")
        if previous_timesteps.shape != timesteps.shape or previous_timesteps.ndim != 1:
            raise ValueError("previous_timesteps must match timesteps")
        if (previous_timesteps >= timesteps).any() or (previous_timesteps < -1).any():
            raise ValueError("DDIM steps must descend and end no lower than -1")
        alpha_t = self._extract(self.alphas_cumprod, timesteps, x_t)
        epsilon = (x_t - alpha_t.sqrt() * predicted_x0) / (1.0 - alpha_t).clamp_min(1e-8).sqrt()
        alpha_previous = torch.where(
            previous_timesteps >= 0,
            self.alphas_cumprod.gather(0, previous_timesteps.clamp_min(0)),
            torch.ones_like(previous_timesteps, dtype=self.alphas_cumprod.dtype),
        ).view(x_t.shape[0], *([1] * (x_t.ndim - 1))).to(x_t.dtype)
        return alpha_previous.sqrt() * predicted_x0 + (1.0 - alpha_previous).clamp_min(0.0).sqrt() * epsilon
