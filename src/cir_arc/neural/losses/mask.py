"""Spatial mask loss and exclusivity regularization for CIR-ARC slot attention.

Computes:
1. slot_mask_loss: Combined BCE + Soft Dice loss on Hungarian-matched slot masks.
2. mask_exclusivity_loss: Regularization penalizing spatial overlap between active slot masks.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import torch
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject


def slot_mask_loss(
    pred_masks: torch.Tensor,
    gt_objects_batch: List[List[ArcObject]],
    matches_batch: List[List[Tuple[int, int]]],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes BCE + Soft Dice loss on Hungarian-matched slot masks against GT object pixel masks.

    Args:
        pred_masks: Predicted per-slot masks of shape (B, K, H, W) in [0, 1].
        gt_objects_batch: List of GT ArcObject lists per batch sample.
        matches_batch: List of (slot_idx, gt_obj_idx) match tuples per batch sample.
        eps: Small epsilon for numerical stability in Dice denominator.

    Returns:
        Scalar mask loss tensor.
    """
    B, K, H, W = pred_masks.shape
    device = pred_masks.device
    total_loss = torch.tensor(0.0, device=device, dtype=pred_masks.dtype)
    num_matched = 0

    for b in range(B):
        matches = matches_batch[b] if b < len(matches_batch) else []
        gt_objs = gt_objects_batch[b] if b < len(gt_objects_batch) else []

        for slot_idx, gt_idx in matches:
            if gt_idx >= len(gt_objs) or slot_idx >= K:
                continue

            gt_obj = gt_objs[gt_idx]
            # Construct binary ground-truth mask (H, W)
            gt_mask = torch.zeros((H, W), device=device, dtype=pred_masks.dtype)
            if hasattr(gt_obj, "pixels") and len(gt_obj.pixels) > 0:
                rows = torch.tensor(gt_obj.pixels[:, 0], device=device, dtype=torch.long)
                cols = torch.tensor(gt_obj.pixels[:, 1], device=device, dtype=torch.long)
                valid_r = (rows >= 0) & (rows < H)
                valid_c = (cols >= 0) & (cols < W)
                valid = valid_r & valid_c
                if valid.any():
                    gt_mask[rows[valid], cols[valid]] = 1.0

            gt_mask = torch.nan_to_num(gt_mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
            p_mask = pred_masks[b, slot_idx]
            p_mask = torch.nan_to_num(p_mask, nan=0.5, posinf=1.0 - eps, neginf=eps).clamp(min=eps, max=1.0 - eps)

            # 1. Binary Cross-Entropy
            bce = - (gt_mask * torch.log(p_mask) + (1.0 - gt_mask) * torch.log(1.0 - p_mask)).mean()

            # 2. Soft Dice Loss
            intersection = (p_mask * gt_mask).sum()
            cardinality = p_mask.sum() + gt_mask.sum()
            dice = 1.0 - (2.0 * intersection + eps) / (cardinality + eps)

            total_loss = total_loss + (bce + dice)
            num_matched += 1

    if num_matched == 0:
        return torch.tensor(0.0, device=device, dtype=pred_masks.dtype)

    return total_loss / num_matched


def mask_exclusivity_loss(
    pred_masks: torch.Tensor,
    objectness: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalizes spatial overlap between distinct active slot masks.

    Args:
        pred_masks: Predicted masks of shape (B, K, H, W) in [0, 1].
        objectness: Optional slot objectness scores of shape (B, K).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar exclusivity penalty tensor.
    """
    B, K, H, W = pred_masks.shape
    if K <= 1:
        return pred_masks.sum() * 0.0

    masks = torch.nan_to_num(pred_masks, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

    # Flatten spatial dimensions: (B, K, H*W)
    flat_masks = masks.reshape(B, K, H * W)

    # Weight masks by objectness confidence if provided
    if objectness is not None:
        obj_w = torch.nan_to_num(objectness, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        flat_masks = flat_masks * obj_w.unsqueeze(-1)

    # Compute pairwise spatial overlap matrix: (B, K, K)
    overlap = torch.bmm(flat_masks, flat_masks.transpose(1, 2))  # (B, K, K)

    # Mask out diagonal (self-overlap)
    eye = torch.eye(K, device=pred_masks.device, dtype=torch.bool).unsqueeze(0)
    off_diag_overlap = overlap.masked_fill(eye, 0.0)

    num_pairs = float(B * K * (K - 1))
    return off_diag_overlap.sum() / (num_pairs * H * W + eps)
