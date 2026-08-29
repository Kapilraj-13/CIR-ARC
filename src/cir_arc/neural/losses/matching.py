"""Hungarian bipartite matching between predicted slots and ground-truth objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject


def compute_cost_matrix(
    pred_props: Dict[str, torch.Tensor],
    gt_objects: List[ArcObject],
    H: int,
    W: int,
    color_cost_weight: float = 1.0,
    pos_cost_weight: float = 1.0,
) -> np.ndarray:
    """Compute the cost matrix between predicted slots and ground truth objects."""
    M = len(gt_objects)
    color_tensor = pred_props.get("color")
    pos_tensor = pred_props.get("position")

    if color_tensor is not None:
        K = color_tensor.shape[0] if color_tensor.dim() == 2 else color_tensor.shape[1]
    elif pos_tensor is not None:
        K = pos_tensor.shape[0] if pos_tensor.dim() == 2 else pos_tensor.shape[1]
    else:
        K = 0

    if M == 0 or K == 0:
        return np.zeros((K, M), dtype=np.float64)

    H_float = float(max(H, 1))
    W_float = float(max(W, 1))
    cost_matrix = np.zeros((K, M), dtype=np.float64)

    # 1. Color cost
    if color_tensor is not None and color_cost_weight > 0.0:
        with torch.no_grad():
            color_logits = color_tensor.detach().float()
            if color_logits.dim() == 3 and color_logits.shape[0] == 1:
                color_logits = color_logits.squeeze(0)
            log_probs = F.log_softmax(color_logits, dim=-1).cpu().numpy()
            log_probs = np.nan_to_num(log_probs, nan=-20.0, posinf=0.0, neginf=-20.0)

        for m, obj in enumerate(gt_objects):
            target_c = int(obj.color)
            if 0 <= target_c < log_probs.shape[-1]:
                cost_matrix[:, m] += color_cost_weight * (-log_probs[:, target_c])

    # 2. Position cost
    if pos_tensor is not None and pos_cost_weight > 0.0:
        with torch.no_grad():
            pred_pos = pos_tensor.detach().float()
            if pred_pos.dim() == 3 and pred_pos.shape[0] == 1:
                pred_pos = pred_pos.squeeze(0)
            pred_pos_np = pred_pos.cpu().numpy()
            pred_pos_np = np.nan_to_num(pred_pos_np, nan=0.5, posinf=1.0, neginf=0.0)

        for m, obj in enumerate(gt_objects):
            r_centroid, c_centroid = obj.centroid
            gt_pos = np.array([(r_centroid + 0.5) / H_float, (c_centroid + 0.5) / W_float], dtype=np.float64)
            diff = pred_pos_np - gt_pos[None, :]
            dist_l2 = np.sqrt(np.sum(diff ** 2, axis=-1))
            cost_matrix[:, m] += pos_cost_weight * dist_l2

    cost_matrix = np.nan_to_num(cost_matrix, nan=1e4, posinf=1e4, neginf=0.0)
    cost_matrix = np.clip(cost_matrix, a_min=0.0, a_max=1e5)
    return cost_matrix


def hungarian_matching(
    pred_props: Dict[str, torch.Tensor],
    gt_objects: List[ArcObject],
    H: int,
    W: int,
    color_cost_weight: float = 1.0,
    pos_cost_weight: float = 1.0,
) -> List[Tuple[int, int]]:
    """Perform Hungarian matching to find optimal bijective assignment."""
    M = len(gt_objects)
    if M == 0:
        return []

    cost_matrix = compute_cost_matrix(
        pred_props, gt_objects, H=H, W=W,
        color_cost_weight=color_cost_weight,
        pos_cost_weight=pos_cost_weight
    )

    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return []

    # Safe Hungarian assignment on sanitized finite cost matrix
    cost_matrix = np.ascontiguousarray(cost_matrix, dtype=np.float64)
    cost_matrix = np.nan_to_num(cost_matrix, nan=1e4, posinf=1e4, neginf=0.0)

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: List[Tuple[int, int]] = [
            (int(r), int(c)) for r, c in zip(row_ind, col_ind)
        ]
        return matches
    except Exception:
        # Fallback greedy matching if scipy throws any exception
        K, M = cost_matrix.shape
        matched = []
        used_cols = set()
        for r in range(min(K, M)):
            best_c = -1
            best_val = float("inf")
            for c in range(M):
                if c not in used_cols and cost_matrix[r, c] < best_val:
                    best_val = cost_matrix[r, c]
                    best_c = c
            if best_c != -1:
                matched.append((r, best_c))
                used_cols.add(best_c)
        return matched
