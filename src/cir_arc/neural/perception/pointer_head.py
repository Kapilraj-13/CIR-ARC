"""Two-Stage Pointer & Click Coordinate Resolution Head (ACTION6) for CIR-ARC.

Implements two-stage coordinate selection for click/pointer interactions:
Stage 1 (Entity Intent Selection): Evaluates slot representations to score
  which entity should be targeted: p(slot_k) = Softmax(W_q s_k).
Stage 2 (Spatial Pixel Localization): Attends from the selected slot representation
  into fine-grained spatial feature tokens (from CNN stem) to derive continuous
  and integer display coordinates (x, y) / (row, col).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStagePointerHead(nn.Module):
    """Two-stage neural pointer head resolving discrete entity intent into pixel click coordinates."""

    def __init__(
        self,
        slot_dim: int = 128,
        feat_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Stage 1: Slot selection scorer (evaluates clickability / target priority)
        self.slot_scorer = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Stage 2: Spatial attention projection between selected slot and spatial tokens
        self.slot_proj = nn.Linear(slot_dim, hidden_dim)
        self.token_proj = nn.Linear(feat_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

        # Refinement MLP predicting coordinate offset from spatial peak
        self.coord_refiner = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Tanh(),  # Normalized small offset [-1, 1]
        )

    def forward(
        self,
        slots: torch.Tensor,
        spatial_tokens: torch.Tensor,
        H: int,
        W: int,
        target_slot_idx: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass resolving click coordinates.

        Args:
            slots: Slot representations of shape (B, K, slot_dim).
            spatial_tokens: Spatial tokens from CNN stem of shape (B, H*W, feat_dim).
            H: Grid height.
            W: Grid width.
            target_slot_idx: Optional explicit target slot index of shape (B,).
            slot_mask: Optional mask over valid slots of shape (B, K).

        Returns:
            Dict containing:
                "slot_logits": (B, K) logits over candidate slots
                "selected_slot": (B,) chosen slot index
                "pixel_heatmap": (B, H, W) spatial attention heatmap
                "coords_norm": (B, 2) normalized (row, col) in [0, 1]
                "coords_pixel": (B, 2) discrete integer pixel (row, col)
                "coords_xy": (B, 2) discrete integer display (x, y) = (col, row)
        """
        B, K, _ = slots.shape
        device = slots.device

        # Stage 1: Slot scoring
        slot_scores = self.slot_scorer(slots).squeeze(-1)  # (B, K)
        if slot_mask is not None:
            slot_scores = slot_scores.masked_fill(~slot_mask.bool(), -1e9)

        if target_slot_idx is not None:
            selected_idx = target_slot_idx
        else:
            selected_idx = slot_scores.argmax(dim=-1)  # (B,)

        batch_idx = torch.arange(B, device=device)
        selected_slots = slots[batch_idx, selected_idx]  # (B, slot_dim)

        # Stage 2: Spatial localization
        q = self.slot_proj(selected_slots).unsqueeze(1)  # (B, 1, hidden_dim)
        k = self.token_proj(spatial_tokens)              # (B, H*W, hidden_dim)

        spatial_attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, 1, H*W)
        spatial_attn = F.softmax(spatial_attn, dim=-1)               # (B, 1, H*W)
        heatmap = spatial_attn.reshape(B, H, W)

        # Compute centroid from heatmap
        r_coords = torch.linspace(0, 1, H, device=device).unsqueeze(1).expand(H, W)
        c_coords = torch.linspace(0, 1, W, device=device).unsqueeze(0).expand(H, W)

        r_exp = (heatmap * r_coords).sum(dim=(-1, -2), keepdim=True).squeeze(-1)  # (B, 1)
        c_exp = (heatmap * c_coords).sum(dim=(-1, -2), keepdim=True).squeeze(-1)  # (B, 1)
        base_norm = torch.cat([r_exp, c_exp], dim=-1)                            # (B, 2)

        # Pixel coordinates
        discrete_r = torch.clamp((base_norm[:, 0] * (H - 1)).round().long(), 0, H - 1)
        discrete_c = torch.clamp((base_norm[:, 1] * (W - 1)).round().long(), 0, W - 1)

        coords_pixel = torch.stack([discrete_r, discrete_c], dim=-1)              # (row, col)
        coords_xy = torch.stack([discrete_c, discrete_r], dim=-1)                 # (x, y) for ARC-AGI-3

        return {
            "slot_logits": slot_scores,
            "selected_slot": selected_idx,
            "pixel_heatmap": heatmap,
            "coords_norm": base_norm,
            "coords_pixel": coords_pixel,
            "coords_xy": coords_xy,
        }


if __name__ == "__main__":
    head = TwoStagePointerHead(slot_dim=128, feat_dim=128)
    slots = torch.randn(2, 24, 128)
    spatial = torch.randn(2, 100, 128)
    out = head(slots, spatial, H=10, W=10)
    assert out["slot_logits"].shape == (2, 24)
    assert out["selected_slot"].shape == (2,)
    assert out["pixel_heatmap"].shape == (2, 10, 10)
    assert out["coords_xy"].shape == (2, 2)
    print("TwoStagePointerHead verified successfully!")
