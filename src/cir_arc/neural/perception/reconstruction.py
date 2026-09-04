"""Reconstruction decoder and slot mask decoder modules for ARC grids.

Decodes object slots back into full 2D grid representations and spatial ownership masks:
1. SlotMaskDecoder: Decodes (B, K, slot_dim) into per-slot binary masks (B, K, H, W) in [0, 1].
2. ReconstructionDecoder: Cross-attention decoding from object slots to full discrete color logits (B, H, W, 10).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMaskDecoder(nn.Module):
    """Decodes object slots into per-slot spatial ownership masks (B, K, H, W) in [0, 1].

    Args:
        slot_dim: Dimensionality of slot vectors (default: 128).
        max_h: Maximum supported grid height (default: 30).
        max_w: Maximum supported grid width (default: 30).
        hidden_dim: Intermediate MLP hidden dimension (default: 64).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        max_h: int = 30,
        max_w: int = 30,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.max_h = max_h
        self.max_w = max_w

        pos_dim = slot_dim // 2  # 64
        self.row_embed = nn.Embedding(max_h, pos_dim)
        self.col_embed = nn.Embedding(max_w, pos_dim)

        self.mask_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        slots: torch.Tensor,
        H: int = 10,
        W: int = 10,
    ) -> torch.Tensor:
        """Decode slots into spatial mask probabilities per slot.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).
            H: Target grid height.
            W: Target grid width.

        Returns:
            Spatial masks tensor of shape (B, K, H, W) with probabilities in [0, 1].
        """
        B, K, D = slots.shape
        device = slots.device

        # Generate 2D coordinate embeddings
        if H > self.max_h or W > self.max_w:
            r_idx = torch.linspace(0, self.max_h - 1, H, device=device).round().long()
            c_idx = torch.linspace(0, self.max_w - 1, W, device=device).round().long()
        else:
            r_idx = torch.arange(H, device=device)
            c_idx = torch.arange(W, device=device)

        r_emb = self.row_embed(r_idx).unsqueeze(1).expand(H, W, -1)  # (H, W, 64)
        c_emb = self.col_embed(c_idx).unsqueeze(0).expand(H, W, -1)  # (H, W, 64)
        pos = torch.cat([r_emb, c_emb], dim=-1)                      # (H, W, 128)
        pos_flat = pos.reshape(1, 1, H * W, D).expand(B, K, -1, -1)  # (B, K, H*W, D)

        slots_expand = slots.unsqueeze(2).expand(-1, -1, H * W, -1)  # (B, K, H*W, D)
        combined = torch.cat([slots_expand, pos_flat], dim=-1)       # (B, K, H*W, 2*D)

        mask_logits = self.mask_mlp(combined).squeeze(-1)            # (B, K, H*W)
        masks = mask_logits.reshape(B, K, H, W)                      # (B, K, H, W)
        return masks


class ReconstructionDecoder(nn.Module):
    """Decodes object slots into full spatial grid color logits via cross-attention.

    Args:
        slot_dim: Dimensionality of input slot vectors (default: 128).
        max_h: Maximum grid height supported (default: 30).
        max_w: Maximum grid width supported (default: 30).
        num_colors: Number of target color classes (default: 10).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        max_h: int = 30,
        max_w: int = 30,
        num_colors: int = 10,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.max_h = max_h
        self.max_w = max_w
        self.num_colors = num_colors

        pos_dim = slot_dim // 2  # 64
        self.row_embed = nn.Embedding(max_h, pos_dim)
        self.col_embed = nn.Embedding(max_w, pos_dim)

        # Cross-attention projections (bias=False)
        self.proj_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.proj_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.proj_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # Reconstruction MLP to color logits
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, num_colors),
        )

    def forward(
        self,
        slots: torch.Tensor,
        objectness: Optional[torch.Tensor] = None,
        slot_masks: Optional[torch.Tensor] = None,
        H: int = 10,
        W: int = 10,
    ) -> torch.Tensor:
        """Decode slots into grid color logits.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).
            objectness: Optional objectness scores of shape (B, K) in [0, 1].
            slot_masks: Optional per-slot spatial masks of shape (B, K, H, W).
            H: Target grid height (1 <= H <= max_h).
            W: Target grid width (1 <= W <= max_w).

        Returns:
            Color logits tensor of shape (B, H, W, num_colors).
        """
        B, K, D = slots.shape
        device = slots.device

        # Generate 2D coordinate queries from row and column embeddings
        if H > self.max_h or W > self.max_w:
            r_idx = torch.linspace(0, self.max_h - 1, H, device=device).round().long()
            c_idx = torch.linspace(0, self.max_w - 1, W, device=device).round().long()
        else:
            r_idx = torch.arange(H, device=device)
            c_idx = torch.arange(W, device=device)

        r_emb = self.row_embed(r_idx).unsqueeze(1).expand(H, W, -1)  # (H, W, 64)
        c_emb = self.col_embed(c_idx).unsqueeze(0).expand(H, W, -1)  # (H, W, 64)
        pos = torch.cat([r_emb, c_emb], dim=-1)                      # (H, W, 128)

        # Reshape to batch queries: (B, H*W, slot_dim)
        cell_queries = pos.reshape(1, H * W, self.slot_dim).expand(B, -1, -1)

        # Cross-attention: Cell queries (Q) attend over slots (K, V)
        q = self.proj_q(cell_queries)  # (B, N, slot_dim)
        k = self.proj_k(slots)         # (B, K, slot_dim)
        v = self.proj_v(slots)         # (B, K, slot_dim)

        scale = self.slot_dim ** -0.5
        attn_logits = torch.einsum("b n d, b k d -> b n k", q, k) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, N, K)

        # Aggregated cell visual features
        cell_features = torch.einsum("b n k, b k d -> b n d", attn_weights, v)  # (B, N, slot_dim)

        # Predict discrete color distribution per cell
        color_logits_flat = self.mlp(cell_features)                             # (B, N, num_colors)
        color_logits = color_logits_flat.reshape(B, H, W, self.num_colors)      # (B, H, W, num_colors)

        return color_logits


if __name__ == "__main__":
    print("Running ReconstructionDecoder & SlotMaskDecoder smoke tests...")
    dec = ReconstructionDecoder()
    mask_dec = SlotMaskDecoder()

    B, K, D = 4, 24, 128
    slots = torch.randn(B, K, D)
    obj = torch.rand(B, K)

    masks = mask_dec(slots, H=12, W=15)
    assert masks.shape == (B, K, 12, 15)
    assert (masks >= 0.0).all() and (masks <= 1.0).all()

    logits = dec(slots, objectness=obj, slot_masks=masks, H=12, W=15)
    assert logits.shape == (B, 12, 15, 10)

    print("All ReconstructionDecoder & SlotMaskDecoder smoke tests passed successfully!")
