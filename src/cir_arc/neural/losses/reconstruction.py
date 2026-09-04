"""Reconstruction loss for CIR-ARC Phase 2."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(
    pred_logits: torch.Tensor,
    target_grid: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute masked cell-level cross-entropy loss between predicted color logits and target grid.

    Args:
        pred_logits: Tensor of shape (B, H, W, 10) representing class logits.
        target_grid: Tensor of shape (B, H, W) containing target color IDs in [0, 9].
        mask: Optional Tensor of shape (B, H, W) where 1.0 (or True) denotes valid cells,
              and 0.0 (or False) denotes masked/padded cells.
        weight: Optional 1D Tensor of shape (10,) containing class loss weights.
        ignore_index: Class index to ignore in cross-entropy computation (default: 10).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar cross-entropy loss (0-dim Tensor).
    """
    if pred_logits.dim() != 4 or pred_logits.shape[-1] != 10:
        raise ValueError(
            f"Expected pred_logits to have shape (B, H, W, 10), got {pred_logits.shape}"
        )
    if target_grid.dim() != 3:
        raise ValueError(
            f"Expected target_grid to have shape (B, H, W), got {target_grid.shape}"
        )

    # Permute to (B, C=10, H, W) for F.cross_entropy
    logits_perm = pred_logits.permute(0, 3, 1, 2)
    # Compute unreduced cell-level loss: (B, H, W) with ignore_index to skip padded tokens
    ce_loss = F.cross_entropy(
        logits_perm, target_grid, weight=weight, reduction="none", ignore_index=ignore_index
    )

    if mask is not None:
        mask_f = mask.to(dtype=ce_loss.dtype, device=ce_loss.device)
        masked_ce = ce_loss * mask_f
        normalizer = mask_f.sum().clamp(min=eps)
        return masked_ce.sum() / normalizer
    else:
        normalizer = (target_grid != ignore_index).to(dtype=ce_loss.dtype, device=ce_loss.device).sum().clamp(min=eps)
        return ce_loss.sum() / normalizer


class ReconstructionLoss(nn.Module):
    """
    Module wrapper for cell-level masked cross-entropy reconstruction loss.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 10,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("weight", weight) if weight is not None else None
        self.weight = weight
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_grid: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return reconstruction_loss(
            pred_logits,
            target_grid,
            mask=mask,
            weight=self.weight,
            ignore_index=self.ignore_index,
            eps=self.eps,
        )


if __name__ == "__main__":
    B, H, W = 2, 8, 8
    pred_logits = torch.randn(B, H, W, 10, requires_grad=True)
    target_grid = torch.randint(0, 10, (B, H, W), dtype=torch.long)
    mask = torch.ones(B, H, W)
    loss = reconstruction_loss(pred_logits, target_grid, mask=mask)
    loss.backward()
    print(f"Reconstruction loss smoke test passed: loss={loss.item():.4f}")
