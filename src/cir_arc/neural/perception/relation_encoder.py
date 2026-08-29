"""Relational Set Transformer module for cross-object reasoning in CIR-ARC.

Implements a 2-layer Set Transformer / Multi-Head Self-Attention block over slot representations:
(B, K, slot_dim) -> (B, K, slot_dim)

Allows object slots to exchange global comparative context (e.g. relative sizing, spatial ordering,
color contrast, containment) prior to symbolic property decoding and reconstruction.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class SetTransformerBlock(nn.Module):
    """Single Set Transformer block with Multi-Head Self-Attention, LayerNorm, and residual MLP."""

    def __init__(
        self,
        slot_dim: int = 128,
        num_heads: int = 4,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(slot_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of a Set Transformer block."""
        normed = self.norm1(x)
        attn_out, _ = self.mha(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SlotRelationEncoder(nn.Module):
    """Multi-layer relational encoder refining object slots via cross-slot self-attention.

    Args:
        slot_dim: Dimensionality of each slot vector (default: 128).
        num_heads: Number of attention heads (default: 4).
        mlp_hidden_dim: Hidden dimension of the residual MLP (default: 256).
        num_layers: Number of stacked Set Transformer blocks (default: 2).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        num_heads: int = 4,
        mlp_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            SetTransformerBlock(
                slot_dim=slot_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(slot_dim)

    def forward(
        self,
        slots: torch.Tensor,
        objectness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refines slot representations through relational self-attention.

        Args:
            slots: Input slot tensor of shape (B, K, slot_dim).
            objectness: Optional slot objectness scores of shape (B, K) in [0, 1].

        Returns:
            Refined slot tensor of shape (B, K, slot_dim).
        """
        x = slots
        # Soft-weight active slots to communicate proportionally to objectness confidence
        if objectness is not None:
            obj_weight = 0.2 + 0.8 * objectness.unsqueeze(-1)
            x = x * obj_weight

        for layer in self.layers:
            x = layer(x, key_padding_mask=None)

        return self.final_norm(x)
