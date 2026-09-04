"""18-Layer Transformer Trunk for the CIR-ARC ~120.18M Reasoner.

Contains:
- 18 CognitiveTransformerBlocks (105,311,232 parameters)
- 1 Final RMSNorm (768 parameters)
Total parameters: 105,312,000.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from cir_arc.neural.reasoner.block import CognitiveTransformerBlock, RMSNorm
from cir_arc.neural.reasoner.config import ReasonerConfig


class CognitiveTransformerTrunk(nn.Module):
    """18-layer Transformer trunk executing cognitive reasoning over fused token sequences."""

    def __init__(self, config: Optional[ReasonerConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ReasonerConfig()
        self.config = config

        self.layers = nn.ModuleList([
            CognitiveTransformerBlock(
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

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass through all 18 Transformer layers.

        Args:
            hidden_states: Input token embeddings [batch_size, seq_len, d_model].
            attention_mask: Optional attention mask.
            kv_caches: Optional list of KV caches for each layer.
            use_cache: Whether to return updated KV caches.

        Returns:
            Tuple of (final_hidden_states, updated_kv_caches).
        """
        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache and new_cache is not None:
                new_kv_caches.append(new_cache)

        hidden_states = self.final_norm(hidden_states)
        return hidden_states, new_kv_caches
