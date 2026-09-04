"""Relational Set Transformer module for cross-object reasoning in CIR-ARC.

Implements a Set Transformer / Multi-Head Self-Attention block over slot representations:
(B, K, slot_dim) -> (B, K, slot_dim) + (B, K, K, rel_dim)

Allows object slots to exchange global comparative context (e.g. relative sizing, spatial ordering,
color contrast, containment) and produces continuous pairwise relation latents for DenseLatentState.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
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

    Emits both refined slot vectors and continuous pairwise interaction tensors for DenseLatentState.
    """

    def __init__(
        self,
        slot_dim: int = 128,
        num_heads: int = 4,
        mlp_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        rel_dim: int = 64,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.num_layers = num_layers
        self.rel_dim = rel_dim

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

        # Pairwise relational projection: maps (s_i, s_j, s_i - s_j) -> rel_dim
        self.pairwise_proj = nn.Sequential(
            nn.Linear(slot_dim * 3, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim // 2, rel_dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        objectness: Optional[torch.Tensor] = None,
        return_pairwise: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Refines slot representations through relational self-attention.

        Args:
            slots: Input slot tensor of shape (B, K, slot_dim).
            objectness: Optional slot objectness scores of shape (B, K) in [0, 1].
            return_pairwise: If True, also returns pairwise relational latent tensor (B, K, K, rel_dim).

        Returns:
            Refined slot tensor of shape (B, K, slot_dim), or (refined_slots, pairwise_latents).
        """
        x = slots
        if objectness is not None:
            obj_weight = 0.2 + 0.8 * objectness.unsqueeze(-1)
            x = x * obj_weight

        for layer in self.layers:
            x = layer(x, key_padding_mask=None)

        refined_slots = self.final_norm(x)

        if not return_pairwise:
            return refined_slots

        # Compute continuous pairwise interaction tensor (B, K, K, rel_dim)
        B, K, D = refined_slots.shape
        s_i = refined_slots.unsqueeze(2).expand(B, K, K, D)
        s_j = refined_slots.unsqueeze(1).expand(B, K, K, D)
        s_diff = s_i - s_j
        pairwise_feat = torch.cat([s_i, s_j, s_diff], dim=-1)
        pairwise_latents = self.pairwise_proj(pairwise_feat)

        return refined_slots, pairwise_latents


if __name__ == "__main__":
    enc = SlotRelationEncoder(slot_dim=128, rel_dim=64)
    x = torch.randn(2, 24, 128)
    out = enc(x)
    assert out.shape == (2, 24, 128)
    out, p = enc(x, return_pairwise=True)
    assert out.shape == (2, 24, 128)
    assert p.shape == (2, 24, 24, 64)
    print("SlotRelationEncoder verified successfully!")
