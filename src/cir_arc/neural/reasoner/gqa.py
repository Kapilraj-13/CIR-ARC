"""Grouped-Query Attention (GQA) module for CIR-ARC Reasoner.

Configuration:
- d_model = 768
- n_q_heads = 12
- n_kv_heads = 4 (3 Q heads share 1 KV head)
- head_dim = 64
- Parameters: W_q (768x768) + W_k (768x256) + W_v (768x256) + W_o (768x768) = 1,572,864
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.neural.reasoner.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with Rotary Position Embeddings.

    Reduces KV cache footprint by sharing key/value heads across query head groups.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_q_heads: int = 12,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.0,
        max_context_len: int = 8192,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_q_heads // n_kv_heads  # 12 // 4 = 3
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = dropout

        # Projections without bias (standard modern LLM design)
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)      # 768 -> 768: 589,824
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)     # 768 -> 256: 196,608
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)     # 768 -> 256: 196,608
        self.out_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)    # 768 -> 768: 589,824

        # Rotary Positional Embedding (functional, 0 parameters)
        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=max_context_len, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for GQA.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            attention_mask: Optional mask [batch_size, 1, seq_len, seq_len] or boolean mask.
            kv_cache: Optional cached (k, v) from previous steps.
            use_cache: Whether to return updated kv_cache.

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model] and optional kv_cache.
        """
        B, S, D = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, S, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0].shape[1]

        # Apply RoPE to queries and keys
        q, k = self.rotary_emb(q, k, seq_len=S, offset=offset)

        # Update KV cache if present
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)

        new_kv_cache = (k, v) if use_cache else None

        # Transpose for multi-head attention: [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [B, 12, S, 64]
        k = k.transpose(1, 2)  # [B, 4, S_kv, 64]
        v = v.transpose(1, 2)  # [B, 4, S_kv, 64]

        # Repeat KV heads to match query heads: [B, 4, S_kv, 64] -> [B, 12, S_kv, 64]
        k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
        v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)

        # Scaled dot-product attention
        # Note: Uses PyTorch's native scaled_dot_product_attention (FlashAttention when supported)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )

        # Reshape back to [B, S, d_model]
        out = out.transpose(1, 2).contiguous().view(B, S, self.n_q_heads * self.head_dim)
        out = self.out_proj(out)

        return out, new_kv_cache
