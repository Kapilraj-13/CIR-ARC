"""Perception evaluation metrics module for CIR-ARC Phase 2.

Implements quantitative perception metrics to evaluate slot-based grid representations:
1. reconstruction_accuracy: Cell-level accuracy between predicted color logits and target grid.
2. object_detection_f1: Slot objectness F1 score against ground-truth object count.
3. color_accuracy: Color classification accuracy on Hungarian-matched slots.
4. position_mae: Normalized MAE between predicted and ground-truth centroids in [0, 1].
5. size_mae: Normalized MAE between predicted and ground-truth object sizes in [0, 1].
6. compute_perception_metrics: Aggregated perception metrics summary dictionary.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject


def _internal_hungarian_matching(
    pred_props: Dict[str, torch.Tensor],
    gt_objects: List[Any],
    H: int = 10,
    W: int = 10,
) -> List[Tuple[int, int]]:
    """Compute optimal bijective bipartite matching between predicted slots and GT objects.

    Cost matrix is formed by color cross-entropy + position L2 distance.

    Args:
        pred_props: Dict containing 'color' (K, 10) and 'position' (K, 2).
        gt_objects: List of ArcObject instances.
        H: Grid height for coordinate normalization.
        W: Grid width for coordinate normalization.

    Returns:
        List of (slot_index, gt_object_index) tuples.
    """
    if len(gt_objects) == 0:
        return []

    color_logits = pred_props["color"]  # (K, 10)
    positions = pred_props["position"]  # (K, 2)
    if color_logits.dim() == 1:
        color_logits = color_logits.unsqueeze(0)
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)

    K = min(color_logits.shape[0], positions.shape[0])
    M = len(gt_objects)

    if K == 0 or M == 0:
        return []

    cost_matrix = np.zeros((K, M), dtype=np.float64)

    for k in range(K):
        slot_color = color_logits[k].unsqueeze(0)  # (1, 10)
        slot_pos = positions[k]  # (2,)
        for m in range(M):
            obj = gt_objects[m]
            # 1. Color Cross-Entropy Cost
            if hasattr(obj, "color"):
                gt_col = obj.color
            elif isinstance(obj, (int, np.integer)):
                gt_col = int(obj)
            else:
                gt_col = 0
            gt_col_tensor = torch.tensor([gt_col], dtype=torch.long, device=slot_color.device)
            ce_cost = F.cross_entropy(slot_color, gt_col_tensor).item()

            # 2. Position L2 Cost (normalized continuous centroid with 0.5 cell centering)
            if hasattr(obj, "centroid"):
                r_c, c_c = obj.centroid
                gt_norm_pos = torch.tensor(
                    [(r_c + 0.5) / float(H), (c_c + 0.5) / float(W)],
                    dtype=slot_pos.dtype,
                    device=slot_pos.device,
                )
            elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
                gt_norm_pos = torch.tensor(
                    [(obj[0] + 0.5) / float(H), (obj[1] + 0.5) / float(W)],
                    dtype=slot_pos.dtype,
                    device=slot_pos.device,
                )
            elif isinstance(obj, torch.Tensor):
                gt_norm_pos = obj.to(slot_pos.device)
            else:
                gt_norm_pos = torch.zeros(2, dtype=slot_pos.dtype, device=slot_pos.device)

            pos_cost = torch.norm(slot_pos - gt_norm_pos, p=2).item()
            cost_matrix[k, m] = ce_cost + pos_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]


def reconstruction_accuracy(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Calculate cell-level reconstruction accuracy (% of correctly reconstructed cells).

    Args:
        pred_logits: Predicted color logits of shape (B, H, W, 10), (B, 10, H, W), or (H, W, 10).
        target: Target grid tensor of shape (B, H, W) or (H, W) with integer color IDs (0-9).
        mask: Optional spatial validity mask (1 for valid cells, 0 for padded cells).

    Returns:
        Reconstruction accuracy as a float in [0.0, 1.0].
    """
    logits = pred_logits.detach()
    tgt = target.detach()

    # Determine class dimension and extract argmax predictions
    if logits.dim() == 4:
        if logits.shape[-1] == 10:
            preds = logits.argmax(dim=-1)
        elif logits.shape[1] == 10:
            preds = logits.argmax(dim=1)
        else:
            preds = logits.argmax(dim=-1)
    elif logits.dim() == 3:
        if logits.shape[-1] == 10:
            preds = logits.argmax(dim=-1)
        elif logits.shape[0] == 10:
            preds = logits.argmax(dim=0)
        else:
            preds = logits.argmax(dim=-1)
    elif logits.dim() == 2:
        preds = logits.argmax(dim=-1)
    else:
        preds = logits.argmax(dim=-1)

    correct_cells = (preds == tgt)

    if mask is not None:
        mask_bool = mask.detach().bool()
        total_valid = mask_bool.sum().item()
        if total_valid == 0:
            return 1.0
        correct_count = correct_cells[mask_bool].sum().item()
        return float(correct_count / total_valid)

    total_cells = tgt.numel()
    if total_cells == 0:
        return 1.0
    return float(correct_cells.sum().item() / total_cells)


def object_detection_f1(
    pred_objectness: torch.Tensor,
    gt_object_count: Union[int, List[int], torch.Tensor, None] = None,
    threshold: float = 0.5,
    gt_counts: Optional[Union[int, List[int], torch.Tensor]] = None,
) -> float:
    """Calculate object detection F1 score for slot objectness predictions.

    Args:
        pred_objectness: Objectness probabilities of shape (B, K) or (K,) in [0, 1].
        gt_object_count: Ground-truth object count(s) per batch item.
        threshold: Threshold for activating a slot as a detected object (default: 0.5).
        gt_counts: Alias for gt_object_count.

    Returns:
        Mean object detection F1 score across the batch in [0.0, 1.0].
    """
    counts_input = gt_object_count if gt_object_count is not None else gt_counts
    if counts_input is None:
        raise ValueError("Either gt_object_count or gt_counts must be provided.")

    obj = pred_objectness.detach()
    if obj.dim() == 1:
        obj = obj.unsqueeze(0)
    B, _ = obj.shape

    if isinstance(counts_input, (int, np.integer)):
        counts_list = [int(counts_input)] * B
    elif isinstance(counts_input, (list, tuple)):
        counts_list = list(counts_input)
        if len(counts_list) == 1 and B > 1:
            counts_list = counts_list * B
    elif isinstance(counts_input, torch.Tensor):
        flat_counts = counts_input.detach().cpu().view(-1).tolist()
        if len(flat_counts) == 1 and B > 1:
            counts_list = flat_counts * B
        else:
            counts_list = flat_counts
    else:
        counts_list = [int(counts_input)] * B

    f1_scores = []
    for b in range(B):
        n_pred = int((obj[b] >= threshold).sum().item())
        n_gt = int(counts_list[b])
        tp = min(n_pred, n_gt)

        if n_pred == 0 and n_gt == 0:
            f1 = 1.0
        elif n_pred == 0 or n_gt == 0:
            f1 = 0.0
        else:
            precision = tp / float(n_pred)
            recall = tp / float(n_gt)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def color_accuracy(
    pred_color_logits: torch.Tensor,
    gt_objects: Optional[Any] = None,
    matches: Optional[List[Tuple[int, int]]] = None,
    matched_gt_colors: Optional[torch.Tensor] = None,
) -> float:
    """Calculate color classification accuracy on Hungarian-matched slots.

    Args:
        pred_color_logits: Predicted color logits of shape (K, 10), (B, K, 10), or (M, 10).
        gt_objects: List of ArcObject instances or list of color integers.
        matches: List of (slot_idx, gt_idx) matching pairs.
        matched_gt_colors: Ground-truth color integer tensor of shape (M,).

    Returns:
        Color accuracy as a float in [0.0, 1.0].
    """
    logits = pred_color_logits.detach()

    # Case 1: Provided explicit matches and gt_objects list
    if matches is not None and gt_objects is not None:
        if len(matches) == 0:
            return 1.0
        correct = 0
        for slot_idx, gt_idx in matches:
            if logits.dim() == 3:
                pred_c = logits[0, slot_idx].argmax(dim=-1).item()
            else:
                pred_c = logits[slot_idx].argmax(dim=-1).item()

            gt_item = gt_objects[gt_idx]
            gt_c = gt_item.color if hasattr(gt_item, "color") else int(gt_item)
            if pred_c == gt_c:
                correct += 1
        return float(correct / len(matches))

    # Case 2: Provided matched_gt_colors directly or gt_objects as Tensor
    target = matched_gt_colors if matched_gt_colors is not None else gt_objects
    if target is not None:
        if isinstance(target, list):
            target_t = torch.tensor(
                [obj.color if hasattr(obj, "color") else int(obj) for obj in target],
                dtype=torch.long,
                device=logits.device,
            )
        elif isinstance(target, np.ndarray):
            target_t = torch.from_numpy(target).long().to(logits.device)
        elif isinstance(target, torch.Tensor):
            target_t = target.detach().long().to(logits.device)
        else:
            target_t = torch.tensor([int(target)], dtype=torch.long, device=logits.device)

        if logits.numel() == 0 or target_t.numel() == 0:
            return 1.0

        preds = logits.argmax(dim=-1)
        correct_count = (preds == target_t).sum().item()
        return float(correct_count / target_t.numel())

    return 1.0


def position_mae(
    pred_positions: torch.Tensor,
    gt_objects: Optional[Any] = None,
    matches: Optional[List[Tuple[int, int]]] = None,
    matched_gt_positions: Optional[torch.Tensor] = None,
    H: int = 10,
    W: int = 10,
) -> float:
    """Calculate normalized MAE between predicted and ground-truth centroids in [0, 1].

    Normalized coordinates use continuous centroid + 0.5 cell centering:
        row_norm = (centroid_row + 0.5) / H
        col_norm = (centroid_col + 0.5) / W

    Args:
        pred_positions: Predicted positions tensor of shape (K, 2), (B, K, 2), or (M, 2).
        gt_objects: List of ArcObject instances or coordinates.
        matches: List of (slot_idx, gt_idx) matching pairs.
        matched_gt_positions: Normalized ground-truth position tensor of shape (M, 2).
        H: Grid height for coordinate normalization (default: 10).
        W: Grid width for coordinate normalization (default: 10).

    Returns:
        Normalized Position MAE as a float in [0.0, 1.0].
    """
    pos = pred_positions.detach()

    # Case 1: Provided explicit matches and gt_objects list
    if matches is not None and gt_objects is not None:
        if len(matches) == 0:
            return 0.0
        errors = []
        for slot_idx, gt_idx in matches:
            if pos.dim() == 3:
                p_coord = pos[0, slot_idx]
            else:
                p_coord = pos[slot_idx]

            gt_item = gt_objects[gt_idx]
            if hasattr(gt_item, "centroid"):
                r_c, c_c = gt_item.centroid
                gt_norm = torch.tensor(
                    [(r_c + 0.5) / float(H), (c_c + 0.5) / float(W)],
                    dtype=p_coord.dtype,
                    device=p_coord.device,
                )
            elif isinstance(gt_item, (tuple, list)) and len(gt_item) >= 2:
                gt_norm = torch.tensor(gt_item[:2], dtype=p_coord.dtype, device=p_coord.device)
            elif isinstance(gt_item, torch.Tensor):
                gt_norm = gt_item.to(p_coord.device)
            else:
                gt_norm = torch.tensor(gt_item, dtype=p_coord.dtype, device=p_coord.device)

            err = torch.abs(p_coord - gt_norm).mean().item()
            errors.append(err)
        return float(np.mean(errors)) if errors else 0.0

    # Case 2: Provided matched_gt_positions directly or gt_objects as Tensor
    target = matched_gt_positions if matched_gt_positions is not None else gt_objects
    if target is not None:
        if isinstance(target, torch.Tensor):
            target_t = target.detach().to(pos.device)
        else:
            target_t = torch.tensor(target, dtype=pos.dtype, device=pos.device)

        if pos.numel() == 0 or target_t.numel() == 0:
            return 0.0

        return float(torch.abs(pos - target_t).mean().item())

    return 0.0


def size_mae(
    pred_sizes: torch.Tensor,
    gt_objects: Optional[Any] = None,
    matches: Optional[List[Tuple[int, int]]] = None,
    matched_gt_sizes: Optional[torch.Tensor] = None,
    H: int = 10,
    W: int = 10,
) -> float:
    """Calculate normalized MAE between predicted and ground-truth object sizes in [0, 1].

    Normalized size is defined as: size_pixels / (H * W).

    Args:
        pred_sizes: Predicted size scalar tensor of shape (K, 1), (B, K, 1), (K,), or (M, 1).
        gt_objects: List of ArcObject instances or size scalars.
        matches: List of (slot_idx, gt_idx) matching pairs.
        matched_gt_sizes: Normalized ground-truth size tensor of shape (M, 1) or (M,).
        H: Grid height for size normalization (default: 10).
        W: Grid width for size normalization (default: 10).

    Returns:
        Normalized Size MAE as a float in [0.0, 1.0].
    """
    sizes = pred_sizes.detach()
    grid_area = float(H * W) if H * W > 0 else 1.0

    # Case 1: Provided explicit matches and gt_objects list
    if matches is not None and gt_objects is not None:
        if len(matches) == 0:
            return 0.0
        errors = []
        for slot_idx, gt_idx in matches:
            if sizes.dim() == 3:
                p_sz = sizes[0, slot_idx]
            elif sizes.dim() == 2:
                p_sz = sizes[slot_idx]
            else:
                p_sz = sizes[slot_idx]

            gt_item = gt_objects[gt_idx]
            if hasattr(gt_item, "size"):
                gt_norm_val = gt_item.size / grid_area
            elif isinstance(gt_item, (int, float)):
                gt_norm_val = gt_item / grid_area if gt_item > 1.0 else float(gt_item)
            elif isinstance(gt_item, torch.Tensor):
                gt_norm_val = gt_item.item() / grid_area if gt_item.item() > 1.0 else gt_item.item()
            else:
                gt_norm_val = float(gt_item) / grid_area

            gt_norm = torch.tensor([gt_norm_val], dtype=p_sz.dtype, device=p_sz.device)
            err = torch.abs(p_sz.view(-1) - gt_norm.view(-1)).mean().item()
            errors.append(err)
        return float(np.mean(errors)) if errors else 0.0

    # Case 2: Provided matched_gt_sizes directly or gt_objects as Tensor
    target = matched_gt_sizes if matched_gt_sizes is not None else gt_objects
    if target is not None:
        if isinstance(target, torch.Tensor):
            target_t = target.detach().to(sizes.device)
        else:
            target_t = torch.tensor(target, dtype=sizes.dtype, device=sizes.device)

        if sizes.numel() == 0 or target_t.numel() == 0:
            return 0.0

        return float(torch.abs(sizes.view(-1) - target_t.view(-1)).mean().item())

    return 0.0


def mask_iou(
    pred_masks: torch.Tensor,
    gt_objects: List[Any],
    matches: List[Tuple[int, int]],
    threshold: float = 0.5,
) -> float:
    """Calculate mean Intersection-over-Union between predicted slot masks and ground-truth pixel sets."""
    if len(matches) == 0:
        return 1.0

    masks_np = (pred_masks.detach().cpu().numpy() >= threshold) if isinstance(pred_masks, torch.Tensor) else (pred_masks >= threshold)
    if masks_np.ndim == 4:
        masks_np = masks_np[0]

    ious = []
    H, W = masks_np.shape[-2], masks_np.shape[-1]
    for slot_idx, gt_idx in matches:
        if slot_idx >= masks_np.shape[0] or gt_idx >= len(gt_objects):
            continue
        obj = gt_objects[gt_idx]
        gt_mask = np.zeros((H, W), dtype=bool)
        if hasattr(obj, "pixels"):
            for r, c in obj.pixels:
                if 0 <= r < H and 0 <= c < W:
                    gt_mask[r, c] = True

        pred_m = masks_np[slot_idx]
        intersection = np.logical_and(pred_m, gt_mask).sum()
        union = np.logical_or(pred_m, gt_mask).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(float(intersection / union))

    return float(np.mean(ious)) if ious else 1.0


def bbox_iou(
    pred_bboxes: torch.Tensor,
    gt_objects: List[Any],
    matches: List[Tuple[int, int]],
    H: int = 30,
    W: int = 30,
) -> float:
    """Calculate mean Bounding Box IoU between predicted and ground-truth bounding boxes."""
    if len(matches) == 0:
        return 1.0

    bboxes_np = pred_bboxes.detach().cpu().numpy() if isinstance(pred_bboxes, torch.Tensor) else pred_bboxes
    if bboxes_np.ndim == 3:
        bboxes_np = bboxes_np[0]

    ious = []
    for slot_idx, gt_idx in matches:
        if slot_idx >= bboxes_np.shape[0] or gt_idx >= len(gt_objects):
            continue
        obj = gt_objects[gt_idx]
        min_r, min_c, max_r, max_c = obj.bounding_box
        gt_box = np.array([min_r / float(H), min_c / float(W), (max_r + 1) / float(H), (max_c + 1) / float(W)])
        pred_box = bboxes_np[slot_idx]

        inter_r1 = max(pred_box[0], gt_box[0])
        inter_c1 = max(pred_box[1], gt_box[1])
        inter_r2 = min(pred_box[2], gt_box[2])
        inter_c2 = min(pred_box[3], gt_box[3])

        inter_area = max(0.0, inter_r2 - inter_r1) * max(0.0, inter_c2 - inter_c1)
        pred_area = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
        gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])

        union_area = pred_area + gt_area - inter_area
        if union_area <= 0:
            ious.append(1.0)
        else:
            ious.append(float(inter_area / union_area))

    return float(np.mean(ious)) if ious else 1.0


def relation_accuracy(
    pred_rel_logits: torch.Tensor,
    gt_objects: List[Any],
    matches: List[Tuple[int, int]],
    H: int = 30,
    W: int = 30,
    threshold: float = 0.5,
) -> float:
    """Calculate multi-label relation classification accuracy across all matched pairs."""
    from cir_arc.neural.perception.relation_graph import extract_ground_truth_relations
    if len(matches) <= 1 or len(gt_objects) <= 1:
        return 1.0

    gt_rel_mat = extract_ground_truth_relations(gt_objects, H=H, W=W)
    probs = torch.sigmoid(pred_rel_logits).detach().cpu().numpy()
    if probs.ndim == 4:
        probs = probs[0]

    correct = 0
    total = 0
    for slot_i, gt_u in matches:
        for slot_j, gt_v in matches:
            if slot_i == slot_j or gt_u == gt_v:
                continue
            pred_rels = (probs[slot_i, slot_j] >= threshold)
            gt_rels = (gt_rel_mat[gt_u, gt_v] >= 0.5)
            correct += int((pred_rels == gt_rels).sum())
            total += len(gt_rels)

    return float(correct / max(total, 1))


def identity_contrastive_score(
    pred_identities: torch.Tensor,
    matches: List[Tuple[int, int]],
) -> float:
    """Calculate mean off-diagonal cosine distance (separation) between active object slots."""
    if len(matches) <= 1:
        return 1.0

    ident = F.normalize(pred_identities.detach(), p=2, dim=-1)
    if ident.dim() == 3:
        ident = ident[0]

    active_indices = [slot_i for slot_i, _ in matches if slot_i < ident.shape[0]]
    if len(active_indices) <= 1:
        return 1.0

    sub_ident = ident[active_indices]
    sim = torch.mm(sub_ident, sub_ident.t())
    eye = torch.eye(len(active_indices), device=sim.device)
    off_diag_sim = sim * (1.0 - eye)
    mean_sim = float(off_diag_sim.sum().item() / (len(active_indices) * (len(active_indices) - 1)))
    return float(1.0 - mean_sim)  # Separation score: higher is more distinct


def compute_perception_metrics(
    pred_logits: torch.Tensor,
    target_grid: torch.Tensor,
    objectness: torch.Tensor,
    pred_props: Dict[str, torch.Tensor],
    gt_objects_batch: List[List[Any]],
    mask: Optional[torch.Tensor] = None,
    heights: Optional[List[int]] = None,
    widths: Optional[List[int]] = None,
    threshold: float = 0.5,
    pred_masks: Optional[torch.Tensor] = None,
    pred_relations: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Aggregate all perception metrics into a structured summary dictionary."""
    recon_acc = reconstruction_accuracy(pred_logits, target_grid, mask=mask)

    gt_counts = [len(objs) for objs in gt_objects_batch]
    obj_f1 = object_detection_f1(objectness, gt_counts=gt_counts, threshold=threshold)

    B = objectness.shape[0] if objectness.dim() > 1 else 1
    color_accs: List[float] = []
    pos_maes: List[float] = []
    size_maes: List[float] = []
    mask_ious: List[float] = []
    bbox_ious: List[float] = []
    rel_accs: List[float] = []
    ident_scores: List[float] = []

    for b in range(B):
        gt_objs = gt_objects_batch[b] if b < len(gt_objects_batch) else []
        H = heights[b] if (heights is not None and b < len(heights)) else target_grid.shape[-2]
        W = widths[b] if (widths is not None and b < len(widths)) else target_grid.shape[-1]

        color_b = pred_props["color"][b] if pred_props["color"].dim() == 3 else pred_props["color"]
        pos_b = pred_props["position"][b] if pred_props["position"].dim() == 3 else pred_props["position"]
        size_b = pred_props["size"][b] if pred_props["size"].dim() == 3 else pred_props["size"]

        if len(gt_objs) == 0:
            continue

        sample_props = {"color": color_b, "position": pos_b, "size": size_b}
        matches = _internal_hungarian_matching(sample_props, gt_objs, H=H, W=W)

        if len(matches) > 0:
            color_accs.append(color_accuracy(color_b, gt_objs, matches))
            pos_maes.append(position_mae(pos_b, gt_objs, matches, H=H, W=W))
            size_maes.append(size_mae(size_b, gt_objs, matches, H=H, W=W))

            if pred_masks is not None:
                mask_b = pred_masks[b] if pred_masks.dim() == 4 else pred_masks
                mask_ious.append(mask_iou(mask_b, gt_objs, matches))

            if "bbox" in pred_props:
                bbox_b = pred_props["bbox"][b] if pred_props["bbox"].dim() == 3 else pred_props["bbox"]
                bbox_ious.append(bbox_iou(bbox_b, gt_objs, matches, H=H, W=W))

            if pred_relations is not None:
                rel_b = pred_relations[b] if pred_relations.dim() == 4 else pred_relations
                rel_accs.append(relation_accuracy(rel_b, gt_objs, matches, H=H, W=W))

            if "identity" in pred_props:
                ident_b = pred_props["identity"][b] if pred_props["identity"].dim() == 3 else pred_props["identity"]
                ident_scores.append(identity_contrastive_score(ident_b, matches))

    res = {
        "recon_acc": float(recon_acc),
        "object_f1": float(obj_f1),
        "color_acc": float(np.mean(color_accs)) if color_accs else 1.0,
        "pos_mae": float(np.mean(pos_maes)) if pos_maes else 0.0,
        "size_mae": float(np.mean(size_maes)) if size_maes else 0.0,
    }
    if mask_ious:
        res["mask_iou"] = float(np.mean(mask_ious))
    if bbox_ious:
        res["bbox_iou"] = float(np.mean(bbox_ious))
    if rel_accs:
        res["relation_acc"] = float(np.mean(rel_accs))
    if ident_scores:
        res["identity_separation"] = float(np.mean(ident_scores))

    return res


if __name__ == "__main__":
    print("Running PerceptionMetrics smoke tests...")

    # 1. Test reconstruction accuracy
    logits = torch.zeros(1, 4, 4, 10)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    logits[0, :, :, 0] = 10.0
    acc_perf = reconstruction_accuracy(logits, target)
    print(f"Reconstruction accuracy (perfect): {acc_perf:.4f}")
    assert acc_perf == 1.0

    # 2. Test object detection F1
    obj_scores = torch.tensor([[0.9, 0.8, 0.1, 0.2]])
    f1_perf = object_detection_f1(obj_scores, gt_counts=[2], threshold=0.5)
    print(f"Object detection F1 (perfect 2/2): {f1_perf:.4f}")
    assert f1_perf == 1.0

    # 3. Test color accuracy
    dummy_objs = [
        ArcObject(color=2, pixels=np.array([[0, 0]])),
        ArcObject(color=7, pixels=np.array([[1, 1]])),
    ]
    pred_c = torch.zeros(2, 10)
    pred_c[0, 2] = 10.0
    pred_c[1, 7] = 10.0
    c_acc = color_accuracy(pred_c, dummy_objs, matches=[(0, 0), (1, 1)])
    print(f"Color accuracy (perfect): {c_acc:.4f}")
    assert c_acc == 1.0

    # 4. Test position and size MAE
    pred_p = torch.tensor([[0.15, 0.2], [0.5, 0.5]])
    pred_s = torch.tensor([[0.02], [0.05]])
    gt_obj_pos = [ArcObject(color=1, pixels=np.array([[1, 1], [1, 2]]))]
    p_mae = position_mae(pred_p, gt_obj_pos, matches=[(0, 0)], H=10, W=10)
    s_mae = size_mae(pred_s, gt_obj_pos, matches=[(0, 0)], H=10, W=10)
    print(f"Position MAE: {p_mae:.6f}, Size MAE: {s_mae:.6f}")
    assert p_mae < 1e-4
    assert s_mae < 1e-4

    # 5. Test compute_perception_metrics
    summary = compute_perception_metrics(
        pred_logits=logits,
        target_grid=target,
        objectness=obj_scores,
        pred_props={"color": pred_c, "position": pred_p, "size": pred_s},
        gt_objects_batch=[gt_obj_pos],
        heights=[10],
        widths=[10],
    )
    print(f"Aggregated metrics summary: {summary}")
    for k in ["recon_acc", "object_f1", "color_acc", "pos_mae", "size_mae"]:
        assert k in summary
        assert 0.0 <= summary[k] <= 1.0

    print("All PerceptionMetrics smoke tests passed successfully!")
