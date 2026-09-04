"""Cognitive Transformer Block with RMSNorm, GQA, and SwiGLU.

Each block contains exactly:
- Pre-attention RMSNorm: 768
- Grouped-Query Attention (12Q/4KV): 1,572,864
- Pre-FFN RMSNorm: 768
- SwiGLU (d_ff=1856): 4,276,224
Total per block: 5,850,624 parameters.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from cir_arc.neural.reasoner.gqa import GroupedQueryAttention
from cir_arc.neural.reasoner.swiglu import SwiGLU


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CognitiveTransformerBlock(nn.Module):
    """A single Transformer layer in the CIR-ARC Reasoner trunk."""

    def __init__(
        self,
        d_model: int = 768,
        n_q_heads: int = 12,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        d_ff: int = 1856,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        max_context_len: int = 8192,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            max_context_len=max_context_len,
            rope_base=rope_base,
        )
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with pre-normalization and residual connections.

        Formula:
            x' = x + GQA(RMSNorm(x))
            x'' = x' + SwiGLU(RMSNorm(x'))
        """
        # Attention block with residual
        norm_x = self.input_layernorm(x)
        attn_out, new_kv_cache = self.self_attn(
            norm_x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out

        # FFN block with residual
        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x, new_kv_cache
