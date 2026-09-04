"""Rotary Position Embeddings (RoPE) for CIR-ARC Reasoner.

Purely functional implementation with zero learnable parameters.
Supports Grouped-Query Attention (GQA) and arbitrary sequence lengths up to max_context.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary embeddings to query and key tensors in Grouped-Query Attention.
    Zero learnable parameters.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: theta_i = base^(-2*(i-1)/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build initial cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim / 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to queries and keys.

        Args:
            q: [batch_size, seq_len, num_q_heads, head_dim]
            k: [batch_size, seq_len, num_kv_heads, head_dim]
            seq_len: Current sequence length.
            offset: Offset for incremental KV-cache decoding.

        Returns:
            Tuple of rotated (q, k).
        """
        if offset + seq_len > self.cos_cached.shape[0]:
            self._build_cache(max(offset + seq_len, self.max_seq_len * 2))

        cos = self.cos_cached[offset : offset + seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[offset : offset + seq_len, :].to(dtype=q.dtype, device=q.device)

        # Reshape for broadcasting over heads: [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
