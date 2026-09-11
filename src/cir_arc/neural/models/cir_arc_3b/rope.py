"""2D Spatial and 1D Temporal Rotary Position Embedding (RoPE) for CIR-ARC-3B.

Provides rotary positional embeddings for:
1. 1D Temporal / Sequential reasoning tokens
2. 2D Spatial grid tokens (coordinates y, x)
Has 0 learnable parameters (pure trigonometric operations).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute frequency complex exponentials for 1D RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Supports both [B, S, H, D] and [B, H, S, D] layouts across standard MHA and GQA.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Detect whether layout is [B, S, H, D] or [B, H, S, D]
    if xq_.ndim == 4 and xk_.ndim == 4 and xq_.shape[2] != xk_.shape[2]:
        # GQA layout [B, S, H, D // 2]: Dim 1 is sequence length, Dim 2 is heads
        seq_len = xq_.shape[1]
        freqs = freqs_cis[:seq_len].view(1, seq_len, 1, -1)
    else:
        # Standard layout [B, H, S, D // 2]: Dim -2 is sequence length
        seq_len = xq_.shape[-2]
        freqs = freqs_cis[:seq_len].view(1, 1, seq_len, -1)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding cache module for GQA attention."""

    def __init__(self, dim: int = 128, max_seq_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _apply(self, fn, recurse=True):
        res = super()._apply(fn, recurse=recurse)
        if hasattr(self, "freqs_cis") and self.freqs_cis is not None:
            self.freqs_cis = self.freqs_cis.to(dtype=torch.complex64)
        return res

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if xq.ndim == 4 and xk.ndim == 4 and xq.shape[2] != xk.shape[2]:
            seq_len = xq.shape[1]
        else:
            seq_len = xq.shape[-2]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(device=xq.device)
        return apply_rotary_emb(xq, xk, freqs_cis)
