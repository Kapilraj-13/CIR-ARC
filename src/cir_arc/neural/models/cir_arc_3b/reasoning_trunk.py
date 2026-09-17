"""24-Layer GQA-SwiGLU Reasoning Trunk for CIR-ARC-3B.

Mathematical Audit:
- d_model = 2560
- n_layers = 24
- n_q_heads = 20, n_kv_heads = 5, head_dim = 128
- d_ff = 6912 (SwiGLU)
- Attention subtotal per block: 16,384,000 + 2,560 (RMSNorm) = 16,386,560
- SwiGLU subtotal per block: 53,084,160 + 2,560 (RMSNorm) = 53,086,720
- Block Total: 69,473,280
- Trunk Total (24 blocks): 24 * 69,473,280 = 1,667,358,720 parameters.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cir_arc.neural.models.cir_arc_3b.config import CirArc3BConfig
from cir_arc.neural.models.cir_arc_3b.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GroupedQueryAttention3B(nn.Module):
    """Grouped-Query Attention with 20 Q heads and 5 KV heads (4:1 sharing ratio)."""

    def __init__(
        self,
        d_model: int = 2560,
        n_q_heads: int = 20,
        n_kv_heads: int = 5,
        head_dim: int = 128,
        dropout: float = 0.0,
        max_context_len: int = 8192,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_q_heads // n_kv_heads  # 20 // 5 = 4
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout

        # Projections strictly without bias
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)      # 2560 -> 2560: 6,553,600
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)     # 2560 -> 640:  1,638,400
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)     # 2560 -> 640:  1,638,400
        self.out_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)    # 2560 -> 2560: 6,553,600

        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=max_context_len, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(B, S, self.n_q_heads, self.head_dim)
        xk = xk.view(B, S, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, S, self.n_kv_heads, self.head_dim)

        xq, xk = self.rotary_emb(xq, xk)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            xk = torch.cat([k_prev, xk], dim=1)
            xv = torch.cat([v_prev, xv], dim=1)

        new_kv_cache = (xk, xv) if use_cache else None

        # Repeat KV heads for GQA (4 query heads per KV head)
        xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
        xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)

        # Transpose to [B, H, S, D]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        output = torch.matmul(attn_weights, xv)  # [B, H, S, D]
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        output = self.out_proj(output)
        return output, new_kv_cache


class SwiGLU3B(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""

    def __init__(self, d_model: int = 2560, d_ff: int = 6912, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # 2560 -> 6912: 17,694,720
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)    # 2560 -> 6912: 17,694,720
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # 6912 -> 2560: 17,694,720
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class CognitiveTransformerBlock3B(nn.Module):
    """A single Transformer layer in the CIR-ARC-3B Reasoner trunk (69,473,280 parameters)."""

    def __init__(
        self,
        d_model: int = 2560,
        n_q_heads: int = 20,
        n_kv_heads: int = 5,
        head_dim: int = 128,
        d_ff: int = 6912,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        max_context_len: int = 8192,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention3B(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            max_context_len=max_context_len,
            rope_base=rope_base,
        )
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = SwiGLU3B(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Residual GQA
        normed = self.input_layernorm(x)
        attn_out, new_kv = self.self_attn(
            normed, attention_mask=attention_mask, kv_cache=kv_cache, use_cache=use_cache
        )
        h = x + attn_out

        # Residual SwiGLU
        normed_h = self.post_attention_layernorm(h)
        mlp_out = self.mlp(normed_h)
        out = h + mlp_out

        return out, new_kv


class GQA_SwiGLU_Trunk3B(nn.Module):
    """The 24-Layer CIR-ARC-3B Cognitive Reasoning Trunk (1,667,358,720 parameters)."""

    def __init__(self, config: Optional[CirArc3BConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = CirArc3BConfig()
        self.config = config

        self.layers = nn.ModuleList([
            CognitiveTransformerBlock3B(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                d_ff=config.d_ff,
                rms_norm_eps=config.rms_norm_eps,
                dropout=config.dropout,
                max_context_len=config.max_context_len,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        new_kv_caches = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            layer_dev = next(layer.parameters()).device
            if hidden_states.device != layer_dev:
                hidden_states = hidden_states.to(layer_dev)
            if attention_mask is not None and attention_mask.device != layer_dev:
                attention_mask = attention_mask.to(layer_dev)

            if gradient_checkpointing and self.training and not use_cache:
                def create_custom_forward(module: nn.Module, dev: torch.device) -> Any:
                    def custom_forward(h: torch.Tensor, mask: Any = None, cache: Any = None, uc: bool = False) -> Any:
                        return module(h.to(dev), attention_mask=mask.to(dev) if mask is not None else None, kv_cache=cache, use_cache=uc)
                    return custom_forward

                hidden_states, _ = checkpoint(
                    create_custom_forward(layer, layer_dev),
                    hidden_states,
                    attention_mask,
                    layer_cache,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, updated_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    kv_cache=layer_cache,
                    use_cache=use_cache,
                )
                if use_cache and updated_kv is not None:
                    new_kv_caches.append(updated_kv)

        return hidden_states, new_kv_caches
