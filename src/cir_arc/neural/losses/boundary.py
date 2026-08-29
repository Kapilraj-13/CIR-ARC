"""Boundary and cell-level objectness loss modules for CIR-ARC perception.

Computes auxiliary supervision losses on spatial feature maps:
1. boundary_loss: Binary cross-entropy between predicted boundary probability map and GT boundaries.
2. cell_objectness_loss: Binary cross-entropy between predicted cell foreground map and GT foreground.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F


def boundary_loss(
    pred_boundary: torch.Tensor,
    target_boundary: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary cross-entropy loss for spatial object boundary prediction.

    Args:
        pred_boundary: Predicted boundary map of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        target_boundary: Ground-truth binary boundary map of shape (B, 1, H, W) or (B, H, W).
        mask: Optional spatial validity mask of shape (B, H, W) or (B, 1, H, W).
        pos_weight: Weight multiplier for positive boundary pixels (default: 2.0).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar boundary loss tensor.
    """
    if pred_boundary.dim() == 4 and pred_boundary.shape[1] == 1:
        pred = pred_boundary.squeeze(1)
    else:
        pred = pred_boundary

    if target_boundary.dim() == 4 and target_boundary.shape[1] == 1:
        target = target_boundary.squeeze(1).float()
    else:
        target = target_boundary.float()

    # Numerical clamp
    pred = pred.clamp(min=eps, max=1.0 - eps)

    # Weighted BCE
    bce = - (pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))

    if mask is not None:
        if mask.dim() == 4 and mask.shape[1] == 1:
            m = mask.squeeze(1).float()
        else:
            m = mask.float()
        bce = bce * m
        return bce.sum() / (m.sum() + eps)

    return bce.mean()


def cell_objectness_loss(
    pred_objectness: torch.Tensor,
    target_objectness: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary cross-entropy loss for cell-level foreground objectness prediction.

    Args:
        pred_objectness: Predicted cell objectness of shape (B, 1, H, W) or (B, H, W) in [0, 1].
        target_objectness: Ground-truth binary foreground mask of shape (B, 1, H, W) or (B, H, W).
        mask: Optional spatial validity mask of shape (B, H, W) or (B, 1, H, W).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar cell objectness loss tensor.
    """
    if pred_objectness.dim() == 4 and pred_objectness.shape[1] == 1:
        pred = pred_objectness.squeeze(1)
    else:
        pred = pred_objectness

    if target_objectness.dim() == 4 and target_objectness.shape[1] == 1:
        target = target_objectness.squeeze(1).float()
    else:
        target = target_objectness.float()

    pred = pred.clamp(min=eps, max=1.0 - eps)
    bce = - (target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))

    if mask is not None:
        if mask.dim() == 4 and mask.shape[1] == 1:
            m = mask.squeeze(1).float()
        else:
            m = mask.float()
        bce = bce * m
        return bce.sum() / (m.sum() + eps)

    return bce.mean()
