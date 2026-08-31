"""Relational graph multi-label cross-entropy loss module for Phase 2.5."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.neural.world_state import NUM_RELATIONS


def relation_loss(
    pred_rel_logits: torch.Tensor,
    gt_rel_targets: Optional[Union[List[np.ndarray], List[torch.Tensor], torch.Tensor]] = None,
    matches_batch: Optional[List[List[Tuple[int, int]]]] = None,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """Computes multi-label binary cross-entropy loss on predicted slot relations.

    Args:
        pred_rel_logits: Predicted relational logits of shape (B, K, K, 14).
        gt_rel_targets: Ground-truth relation matrices per batch element of shape (M_b, M_b, 14).
        matches_batch: List of Hungarian match pairs [(slot_i, gt_u), ...] per batch element.
        pos_weight: Weight applied to positive relation edges (default: 2.0).

    Returns:
        Scalar relation loss tensor.
    """
    if pred_rel_logits.numel() == 0:
        return torch.tensor(0.0, device=pred_rel_logits.device, requires_grad=True)

    B, K, _, num_rels = pred_rel_logits.shape
    device = pred_rel_logits.device
    target = torch.zeros((B, K, K, num_rels), dtype=torch.float32, device=device)

    if gt_rel_targets is not None and matches_batch is not None:
        for b in range(min(B, len(matches_batch), len(gt_rel_targets))):
            matches = matches_batch[b]
            gt_mat = gt_rel_targets[b]
            if isinstance(gt_mat, np.ndarray):
                gt_mat_t = torch.from_numpy(gt_mat).float().to(device)
            elif isinstance(gt_mat, torch.Tensor):
                gt_mat_t = gt_mat.float().to(device)
            else:
                continue

            M = gt_mat_t.shape[0] if gt_mat_t.dim() >= 2 else 0
            if M == 0:
                continue

            for slot_i, gt_u in matches:
                if slot_i >= K or gt_u >= M:
                    continue
                for slot_j, gt_v in matches:
                    if slot_j >= K or gt_v >= M or slot_i == slot_j:
                        continue
                    target[b, slot_i, slot_j] = gt_mat_t[gt_u, gt_v]

    # Weighted BCE with logits
    pos_w_tensor = torch.full((num_rels,), pos_weight, device=device)
    loss = F.binary_cross_entropy_with_logits(
        pred_rel_logits,
        target,
        pos_weight=pos_w_tensor,
    )
    return loss
