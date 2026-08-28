"""Supervised property losses on Hungarian-matched slot-object pairs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject


def _is_batched_matches(matches: Any) -> bool:
    """Determine if matches is a list of batch lists or a flat list of pairs."""
    if not isinstance(matches, list):
        return False
    if len(matches) == 0:
        return False
    first = matches[0]
    return isinstance(first, list)


def _is_batched_gt(gt_objects: Any) -> bool:
    """Determine if gt_objects is a list of batch lists or a flat list of ArcObjects."""
    if not isinstance(gt_objects, list):
        return False
    if len(gt_objects) == 0:
        return False
    first = gt_objects[0]
    return isinstance(first, list)


def color_loss(
    pred_color: torch.Tensor,
    gt_objects: Union[List[ArcObject], List[List[ArcObject]]],
    matches: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
) -> torch.Tensor:
    """
    Compute cross-entropy loss on matched slots' predicted color logits.

    Args:
        pred_color: Tensor of shape (K, 10) or (B, K, 10) color logits.
        gt_objects: List of ArcObject (unbatched) or List of List of ArcObject (batched).
        matches: List of (slot_idx, gt_idx) tuples or List of Lists (batched).

    Returns:
        Scalar loss tensor.
    """
    if pred_color.dim() == 2:
        # Unbatched (K, 10)
        flat_matches: List[Tuple[int, int]] = matches  # type: ignore
        flat_gt: List[ArcObject] = gt_objects  # type: ignore
        if len(flat_matches) == 0 or len(flat_gt) == 0:
            return pred_color.sum() * 0.0

        slot_indices = [p[0] for p in flat_matches]
        gt_indices = [p[1] for p in flat_matches]
        matched_logits = pred_color[slot_indices]  # (M_m, 10)
        target_colors = torch.tensor(
            [int(flat_gt[idx].color) for idx in gt_indices],
            dtype=torch.long,
            device=pred_color.device,
        )
        return F.cross_entropy(matched_logits, target_colors)

    elif pred_color.dim() == 3:
        # Batched (B, K, 10)
        B = pred_color.shape[0]
        batch_matches: List[List[Tuple[int, int]]] = matches if _is_batched_matches(matches) else [matches] * B  # type: ignore
        batch_gt: List[List[ArcObject]] = gt_objects if _is_batched_gt(gt_objects) else [gt_objects] * B  # type: ignore

        total_loss = pred_color.sum() * 0.0
        total_matched = 0

        for b in range(B):
            b_matches = batch_matches[b]
            b_gt = batch_gt[b]
            if len(b_matches) == 0 or len(b_gt) == 0:
                continue
            slot_indices = [p[0] for p in b_matches]
            gt_indices = [p[1] for p in b_matches]
            matched_logits = pred_color[b, slot_indices]
            target_colors = torch.tensor(
                [int(b_gt[idx].color) for idx in gt_indices],
                dtype=torch.long,
                device=pred_color.device,
            )
            loss_b = F.cross_entropy(matched_logits, target_colors, reduction="sum")
            total_loss = total_loss + loss_b
            total_matched += len(b_matches)

        if total_matched == 0:
            return pred_color.sum() * 0.0
        return total_loss / float(total_matched)

    else:
        raise ValueError(f"Unexpected pred_color dimension: {pred_color.dim()}")


def position_loss(
    pred_pos: torch.Tensor,
    gt_objects: Union[List[ArcObject], List[List[ArcObject]]],
    matches: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
    H: Union[int, List[int]],
    W: Union[int, List[int]],
) -> torch.Tensor:
    """
    Compute MSE loss between predicted normalized positions [0, 1]^2 and ground truth.

    Args:
        pred_pos: Tensor of shape (K, 2) or (B, K, 2).
        gt_objects: List of ArcObject or List of List of ArcObject.
        matches: List of (slot_idx, gt_idx) tuples or List of Lists.
        H: Grid height (int or list of ints).
        W: Grid width (int or list of ints).

    Returns:
        Scalar MSE loss tensor.
    """
    if pred_pos.dim() == 2:
        flat_matches: List[Tuple[int, int]] = matches  # type: ignore
        flat_gt: List[ArcObject] = gt_objects  # type: ignore
        if len(flat_matches) == 0 or len(flat_gt) == 0:
            return pred_pos.sum() * 0.0

        H_float = float(max(H if isinstance(H, int) else H[0], 1))
        W_float = float(max(W if isinstance(W, int) else W[0], 1))

        slot_indices = [p[0] for p in flat_matches]
        gt_indices = [p[1] for p in flat_matches]
        matched_pos = pred_pos[slot_indices]  # (M_m, 2)

        target_coords = []
        for idx in gt_indices:
            r, c = flat_gt[idx].centroid
            target_coords.append([(r + 0.5) / H_float, (c + 0.5) / W_float])

        target_pos = torch.tensor(
            target_coords, dtype=pred_pos.dtype, device=pred_pos.device
        )
        return F.mse_loss(matched_pos, target_pos)

    elif pred_pos.dim() == 3:
        B = pred_pos.shape[0]
        batch_matches: List[List[Tuple[int, int]]] = matches if _is_batched_matches(matches) else [matches] * B  # type: ignore
        batch_gt: List[List[ArcObject]] = gt_objects if _is_batched_gt(gt_objects) else [gt_objects] * B  # type: ignore
        H_list = H if isinstance(H, list) else [H] * B
        W_list = W if isinstance(W, list) else [W] * B

        total_loss = pred_pos.sum() * 0.0
        total_matched = 0

        for b in range(B):
            b_matches = batch_matches[b]
            b_gt = batch_gt[b]
            if len(b_matches) == 0 or len(b_gt) == 0:
                continue
            H_float = float(max(H_list[b], 1))
            W_float = float(max(W_list[b], 1))

            slot_indices = [p[0] for p in b_matches]
            gt_indices = [p[1] for p in b_matches]
            matched_pos = pred_pos[b, slot_indices]

            target_coords = []
            for idx in gt_indices:
                r, c = b_gt[idx].centroid
                target_coords.append([(r + 0.5) / H_float, (c + 0.5) / W_float])

            target_pos = torch.tensor(
                target_coords, dtype=pred_pos.dtype, device=pred_pos.device
            )
            loss_b = F.mse_loss(matched_pos, target_pos, reduction="sum")
            total_loss = total_loss + loss_b
            total_matched += len(b_matches) * 2

        if total_matched == 0:
            return pred_pos.sum() * 0.0
        return total_loss / float(total_matched)

    else:
        raise ValueError(f"Unexpected pred_pos dimension: {pred_pos.dim()}")


def size_loss(
    pred_size: torch.Tensor,
    gt_objects: Union[List[ArcObject], List[List[ArcObject]]],
    matches: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
    H: Union[int, List[int]],
    W: Union[int, List[int]],
) -> torch.Tensor:
    """
    Compute MSE loss between predicted normalized size in [0, 1] and ground truth.

    Args:
        pred_size: Tensor of shape (K, 1), (K,), (B, K, 1), or (B, K).
        gt_objects: List of ArcObject or List of List of ArcObject.
        matches: List of (slot_idx, gt_idx) tuples or List of Lists.
        H: Grid height (int or list of ints).
        W: Grid width (int or list of ints).

    Returns:
        Scalar MSE loss tensor.
    """
    if pred_size.dim() in (1, 2) and (pred_size.dim() == 1 or pred_size.shape[0] != len(gt_objects) or not _is_batched_gt(gt_objects)):
        flat_matches: List[Tuple[int, int]] = matches  # type: ignore
        flat_gt: List[ArcObject] = gt_objects  # type: ignore
        if len(flat_matches) == 0 or len(flat_gt) == 0:
            return pred_size.sum() * 0.0

        H_val = H if isinstance(H, int) else H[0]
        W_val = W if isinstance(W, int) else W[0]
        area = float(max(H_val * W_val, 1))

        slot_indices = [p[0] for p in flat_matches]
        gt_indices = [p[1] for p in flat_matches]
        matched_size = pred_size[slot_indices].view(-1)

        target_sizes = torch.tensor(
            [float(flat_gt[idx].size) / area for idx in gt_indices],
            dtype=pred_size.dtype,
            device=pred_size.device,
        )
        return F.mse_loss(matched_size, target_sizes)

    else:
        # Batched
        B = pred_size.shape[0]
        batch_matches: List[List[Tuple[int, int]]] = matches if _is_batched_matches(matches) else [matches] * B  # type: ignore
        batch_gt: List[List[ArcObject]] = gt_objects if _is_batched_gt(gt_objects) else [gt_objects] * B  # type: ignore
        H_list = H if isinstance(H, list) else [H] * B
        W_list = W if isinstance(W, list) else [W] * B

        total_loss = pred_size.sum() * 0.0
        total_matched = 0

        for b in range(B):
            b_matches = batch_matches[b]
            b_gt = batch_gt[b]
            if len(b_matches) == 0 or len(b_gt) == 0:
                continue
            area = float(max(H_list[b] * W_list[b], 1))

            slot_indices = [p[0] for p in b_matches]
            gt_indices = [p[1] for p in b_matches]
            matched_size = pred_size[b, slot_indices].view(-1)

            target_sizes = torch.tensor(
                [float(b_gt[idx].size) / area for idx in gt_indices],
                dtype=pred_size.dtype,
                device=pred_size.device,
            )
            loss_b = F.mse_loss(matched_size, target_sizes, reduction="sum")
            total_loss = total_loss + loss_b
            total_matched += len(b_matches)

        if total_matched == 0:
            return pred_size.sum() * 0.0
        return total_loss / float(total_matched)


def objectness_loss(
    objectness: torch.Tensor,
    matches: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
    total_slots: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss on slot objectness probabilities.
    Matched slots target 1.0; unmatched slots target 0.0.

    Args:
        objectness: Tensor of shape (K,), (K, 1), (B, K), or (B, K, 1) in [0, 1].
        matches: List of (slot_idx, gt_idx) or List of Lists (batched).
        total_slots: Optional integer total number of slots.

    Returns:
        Scalar BCE loss tensor.
    """
    target = torch.zeros_like(objectness)
    if objectness.dim() == 1:
        K = total_slots if total_slots is not None else objectness.shape[0]
        flat_matches: List[Tuple[int, int]] = matches  # type: ignore
        for slot_idx, _ in flat_matches:
            if 0 <= slot_idx < K:
                target[slot_idx] = 1.0
        clamped_obj = objectness.clamp(min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(clamped_obj, target)

    elif objectness.dim() == 2 and objectness.shape[1] == 1 and not _is_batched_matches(matches):
        K = total_slots if total_slots is not None else objectness.shape[0]
        flat_matches: List[Tuple[int, int]] = matches  # type: ignore
        for slot_idx, _ in flat_matches:
            if 0 <= slot_idx < K:
                target[slot_idx, 0] = 1.0
        clamped_obj = objectness.clamp(min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(clamped_obj, target)

    elif objectness.dim() == 2:
        # Batched (B, K)
        B, K = objectness.shape
        if total_slots is not None:
            K = total_slots
        if _is_batched_matches(matches):
            batch_matches: List[List[Tuple[int, int]]] = matches  # type: ignore
            for b in range(min(B, len(batch_matches))):
                for slot_idx, _ in batch_matches[b]:
                    if 0 <= slot_idx < K:
                        target[b, slot_idx] = 1.0
        else:
            flat_matches: List[Tuple[int, int]] = matches  # type: ignore
            for slot_idx, _ in flat_matches:
                if 0 <= slot_idx < K:
                    if B == 1:
                        target[0, slot_idx] = 1.0
                    else:
                        target[:, slot_idx] = 1.0
        clamped_obj = objectness.clamp(min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(clamped_obj, target)

    elif objectness.dim() == 3:
        # Batched (B, K, 1)
        B, K, _ = objectness.shape
        if total_slots is not None:
            K = total_slots
        if _is_batched_matches(matches):
            batch_matches: List[List[Tuple[int, int]]] = matches  # type: ignore
            for b in range(min(B, len(batch_matches))):
                for slot_idx, _ in batch_matches[b]:
                    if 0 <= slot_idx < K:
                        target[b, slot_idx, 0] = 1.0
        else:
            flat_matches: List[Tuple[int, int]] = matches  # type: ignore
            for slot_idx, _ in flat_matches:
                if 0 <= slot_idx < K:
                    if B == 1:
                        target[0, slot_idx, 0] = 1.0
                    else:
                        target[:, slot_idx, 0] = 1.0
        clamped_obj = objectness.clamp(min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(clamped_obj, target)

    else:
        raise ValueError(f"Unexpected objectness tensor shape: {objectness.shape}")


def compute_property_losses(
    pred_props: Dict[str, torch.Tensor],
    objectness: torch.Tensor,
    gt_objects: Union[List[ArcObject], List[List[ArcObject]]],
    matches: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
    H: Union[int, List[int]],
    W: Union[int, List[int]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all property losses and return a component dictionary.

    Args:
        pred_props: Dict with 'color', 'position', 'size', etc.
        objectness: Tensor of objectness probabilities.
        gt_objects: Ground-truth ArcObjects.
        matches: Hungarian matched pairs.
        H: Height(s).
        W: Width(s).
        weights: Optional dict of loss weights.

    Returns:
        Dict with 'color_loss', 'pos_loss', 'size_loss', 'obj_loss', and 'total_property_loss'.
    """
    if weights is None:
        weights = {
            "color": 0.5,
            "position": 1.0,
            "size": 0.5,
            "objectness": 0.2,
        }

    c_loss = color_loss(pred_props["color"], gt_objects, matches) if "color" in pred_props else objectness.new_tensor(0.0)
    p_loss = position_loss(pred_props["position"], gt_objects, matches, H, W) if "position" in pred_props else objectness.new_tensor(0.0)
    s_loss = size_loss(pred_props["size"], gt_objects, matches, H, W) if "size" in pred_props else objectness.new_tensor(0.0)
    o_loss = objectness_loss(objectness, matches, total_slots=pred_props.get("color", objectness).shape[-2] if pred_props.get("color") is not None else None)

    total = (
        weights.get("color", 0.5) * c_loss
        + weights.get("position", 1.0) * p_loss
        + weights.get("size", 0.5) * s_loss
        + weights.get("objectness", 0.2) * o_loss
    )

    return {
        "color_loss": c_loss,
        "pos_loss": p_loss,
        "size_loss": s_loss,
        "obj_loss": o_loss,
        "total_property_loss": total,
    }


class PropertyLoss(nn.Module):
    """
    PyTorch nn.Module for computing property losses on Hungarian-matched slots.
    """

    def __init__(
        self,
        color_weight: float = 0.5,
        pos_weight: float = 1.0,
        size_weight: float = 0.5,
        obj_weight: float = 0.2,
    ):
        super().__init__()
        self.color_weight = color_weight
        self.pos_weight = pos_weight
        self.size_weight = size_weight
        self.obj_weight = obj_weight

    def forward(
        self,
        pred_props: Dict[str, torch.Tensor],
        objectness: torch.Tensor,
        gt_objects: Union[List[ArcObject], List[List[ArcObject]]],
        matches: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
        H: Union[int, List[int]],
        W: Union[int, List[int]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weights = {
            "color": self.color_weight,
            "position": self.pos_weight,
            "size": self.size_weight,
            "objectness": self.obj_weight,
        }
        loss_dict = compute_property_losses(
            pred_props=pred_props,
            objectness=objectness,
            gt_objects=gt_objects,
            matches=matches,
            H=H,
            W=W,
            weights=weights,
        )
        return loss_dict["total_property_loss"], loss_dict


if __name__ == "__main__":
    K, H, W = 24, 10, 10
    pred_props = {
        "color": torch.randn(K, 10, requires_grad=True),
        "position": torch.rand(K, 2, requires_grad=True),
        "size": torch.rand(K, 1, requires_grad=True),
    }
    objectness = torch.rand(K, requires_grad=True)
    dummy_pixels = np.array([[1, 1], [1, 2]], dtype=np.int64)
    gt_objects = [ArcObject(color=2, pixels=dummy_pixels, connectivity=4)]
    matches = [(0, 0)]
    prop_loss_mod = PropertyLoss()
    total_loss, loss_dict = prop_loss_mod(pred_props, objectness, gt_objects, matches, H, W)
    total_loss.backward()
    print(f"Property loss smoke test passed: total_loss={total_loss.item():.4f}, loss_dict={loss_dict}")
