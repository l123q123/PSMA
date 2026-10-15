"""Small xformers-free implementation of the LVSM transformer block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class QKNormSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, use_qk_norm: bool = True) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        if key_padding_mask is not None and key_padding_mask.shape != (b, n):
            raise ValueError("key_padding_mask must be [B, N]")
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attention_mask = None if key_padding_mask is None else key_padding_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attended = attended.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj(attended)


class QKNormCrossAttention(nn.Module):
    """QK-normalized attention from query tokens into an external memory."""

    def __init__(self, dim: int, heads: int, use_qk_norm: bool = True) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3 or query.shape[0] != memory.shape[0]:
            raise ValueError("query and memory must be [B, tokens, dim] with matching batches")
        if query.shape[-1] != self.dim or memory.shape[-1] != self.dim:
            raise ValueError("query and memory dimensions must match attention dimension")
        if memory_mask is not None and memory_mask.shape != memory.shape[:2]:
            raise ValueError("memory_mask must be [B, memory_tokens]")
        b, query_count, _ = query.shape
        memory_count = memory.shape[1]
        q = self.to_q(query).view(b, query_count, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(memory).chunk(2, dim=-1)
        k = k.view(b, memory_count, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, memory_count, self.heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        mask = None if memory_mask is None else memory_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(attended.transpose(1, 2).reshape(b, query_count, self.dim))


class QKNormTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, use_qk_norm: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QKNormSelfAttention(dim, heads, use_qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        return x + self.mlp(self.norm2(x))
