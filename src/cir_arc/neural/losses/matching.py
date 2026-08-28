"""Hungarian bipartite matching between predicted slots and ground-truth objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject


def hungarian_matching(
    pred_props: Dict[str, torch.Tensor],
    gt_objects: List[ArcObject],
    H: int,
    W: int,
    color_cost_weight: float = 1.0,
    pos_cost_weight: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Perform Hungarian matching to find optimal bijective assignment between
    predicted slots and ground-truth ArcObject instances.

    Args:
        pred_props: Dict containing predicted property tensors for a single sample:
            - 'color': Tensor of shape (K, 10) representing color logits.
            - 'position': Tensor of shape (K, 2) representing normalized (row, col) in [0, 1].
            - optional other properties.
        gt_objects: List of M ground-truth ArcObject instances.
        H: Grid height (int).
        W: Grid width (int).
        color_cost_weight: Weight for color cross-entropy cost (default: 1.0).
        pos_cost_weight: Weight for position L2 distance cost (default: 1.0).

    Returns:
        List of (slot_idx, gt_idx) matched pairs of length min(K, M).
        If M == 0 or K == 0, returns an empty list [].
    """
    M = len(gt_objects)
    if M == 0:
        return []

    color_tensor = pred_props.get("color")
    pos_tensor = pred_props.get("position")

    # Determine K from color or position tensor
    if color_tensor is not None:
        K = color_tensor.shape[0] if color_tensor.dim() == 2 else color_tensor.shape[1]
    elif pos_tensor is not None:
        K = pos_tensor.shape[0] if pos_tensor.dim() == 2 else pos_tensor.shape[1]
    else:
        return []

    if K == 0:
        return []

    H_float = float(max(H, 1))
    W_float = float(max(W, 1))

    # Initialize cost matrix (K, M)
    cost_matrix = np.zeros((K, M), dtype=np.float64)

    # 1. Color cost: Cross-Entropy (negative log probability)
    if color_tensor is not None and color_cost_weight > 0.0:
        with torch.no_grad():
            color_logits = color_tensor.detach().float()
            if color_logits.dim() == 3 and color_logits.shape[0] == 1:
                color_logits = color_logits.squeeze(0)
            log_probs = F.log_softmax(color_logits, dim=-1).cpu().numpy()  # (K, 10)

        for m, obj in enumerate(gt_objects):
            target_c = int(obj.color)
            if 0 <= target_c < 10:
                cost_matrix[:, m] += color_cost_weight * (-log_probs[:, target_c])

    # 2. Position cost: Euclidean L2 distance in normalized [0, 1] coords
    if pos_tensor is not None and pos_cost_weight > 0.0:
        with torch.no_grad():
            pred_pos = pos_tensor.detach().float()
            if pred_pos.dim() == 3 and pred_pos.shape[0] == 1:
                pred_pos = pred_pos.squeeze(0)
            pred_pos_np = pred_pos.cpu().numpy()  # (K, 2)

        for m, obj in enumerate(gt_objects):
            r_centroid, c_centroid = obj.centroid
            # Normalized centroid: (row + 0.5) / H, (col + 0.5) / W
            gt_pos = np.array([(r_centroid + 0.5) / H_float, (c_centroid + 0.5) / W_float], dtype=np.float64)
            diff = pred_pos_np - gt_pos[None, :]  # (K, 2)
            dist_l2 = np.sqrt(np.sum(diff ** 2, axis=-1))  # (K,)
            cost_matrix[:, m] += pos_cost_weight * dist_l2

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = [
        (int(r), int(c)) for r, c in zip(row_ind, col_ind)
    ]
    return matches


if __name__ == "__main__":
    K, H, W = 24, 10, 10
    pred_props = {
        "color": torch.randn(K, 10),
        "position": torch.rand(K, 2),
    }
    dummy_pixels = np.array([[0, 0], [0, 1]], dtype=np.int64)
    gt_objects = [ArcObject(color=1, pixels=dummy_pixels, connectivity=4)]
    matches = hungarian_matching(pred_props, gt_objects, H=H, W=W)
    print(f"Hungarian matching smoke test passed: matches={matches}")
