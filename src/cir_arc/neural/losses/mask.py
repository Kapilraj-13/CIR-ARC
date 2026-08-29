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
            if gt_idx >= len(gt_objs):
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

            p_mask = pred_masks[b, slot_idx].clamp(min=1e-6, max=1.0 - 1e-6)

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
        pred_masks: Predicted per-slot masks of shape (B, K, H, W) in [0, 1].
        objectness: Optional slot objectness scores of shape (B, K) in [0, 1].
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar exclusivity penalty tensor.
    """
    B, K, H, W = pred_masks.shape
    if K <= 1:
        return torch.tensor(0.0, device=pred_masks.device, dtype=pred_masks.dtype)

    masks = pred_masks  # (B, K, H, W)
    if objectness is not None:
        # Scale masks by slot objectness
        obj_unsq = objectness.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        masks = masks * obj_unsq

    # Sum of pairwise overlaps: sum_{i != j} M_i * M_j = (sum_k M_k)^2 - sum_k M_k^2
    mask_sum = masks.sum(dim=1)             # (B, H, W)
    mask_sq_sum = (masks ** 2).sum(dim=1)   # (B, H, W)

    overlap = F.relu(mask_sum ** 2 - mask_sq_sum)  # (B, H, W)
    return overlap.mean()
