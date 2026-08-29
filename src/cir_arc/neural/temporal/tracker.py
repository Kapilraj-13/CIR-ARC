from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from cir_arc.neural.temporal.kinematics import classify_motion_direction, compute_velocity


@dataclass
class TrackedSlot:
    track_id: int
    slot_embedding: torch.Tensor          # shape: (slot_dim,)
    predicted_color: int                  # 0..15
    predicted_shape: int                  # 0..7
    predicted_size: float                 # in [0, 1]
    centroid: Tuple[float, float]         # (row, col)
    velocity: Tuple[float, float] = (0.0, 0.0)
    motion_direction: str = "STILL"
    layer_index: int = 0
    attention_mask: Optional[np.ndarray] = None
    lifecycle_state: str = "SPAWNED"      # SPAWNED, ACTIVE, DESTROYED
    confidence: float = 1.0
    history: List[Tuple[float, float]] = field(default_factory=list)


class TemporalSlotTracker:
    """Tracks object slots across time frames via Hungarian matching on embeddings & spatial centroids."""

    def __init__(
        self,
        objectness_threshold: float = 0.4,
        match_threshold: float = 1.5,
        slot_dim: int = 128,
    ) -> None:
        self.objectness_threshold = objectness_threshold
        self.match_threshold = match_threshold
        self.slot_dim = slot_dim
        self._next_track_id = 1
        self.active_tracks: Dict[int, TrackedSlot] = {}
        self.step_count = 0

    def reset(self) -> None:
        self._next_track_id = 1
        self.active_tracks.clear()
        self.step_count = 0

    def update_from_perception(
        self,
        slots: torch.Tensor,                # (K, slot_dim) or (1, K, slot_dim)
        objectness: torch.Tensor,           # (K,) or (1, K)
        attn_maps: Optional[torch.Tensor] = None, # (K, H*W)
        H: int = 30,
        W: int = 30,
        color_preds: Optional[torch.Tensor] = None, # (K, 10 or 16)
        pos_preds: Optional[torch.Tensor] = None,   # (K, 2)
        shape_preds: Optional[torch.Tensor] = None, # (K, 8)
        size_preds: Optional[torch.Tensor] = None,  # (K, 1)
    ) -> List[TrackedSlot]:
        """Update tracker with new frame slot predictions."""
        self.step_count += 1

        if slots.dim() == 3:
            slots = slots.squeeze(0)
        if objectness.dim() == 2:
            objectness = objectness.squeeze(0)
        if attn_maps is not None and attn_maps.dim() == 3:
            attn_maps = attn_maps.squeeze(0)
        if color_preds is not None and color_preds.dim() == 3:
            color_preds = color_preds.squeeze(0)
        if pos_preds is not None and pos_preds.dim() == 3:
            pos_preds = pos_preds.squeeze(0)
        if shape_preds is not None and shape_preds.dim() == 3:
            shape_preds = shape_preds.squeeze(0)
        if size_preds is not None and size_preds.dim() == 3:
            size_preds = size_preds.squeeze(0)

        K = slots.shape[0]
        device = slots.device

        # Filter active slots by objectness
        active_indices = [
            i for i in range(K) if objectness[i].item() >= self.objectness_threshold
        ]

        if not active_indices:
            # All existing tracks become destroyed
            for track in self.active_tracks.values():
                track.lifecycle_state = "DESTROYED"
            return []

        # Extract slot features for active indices
        cand_slots = slots[active_indices]  # (M, slot_dim)
        cand_obj = objectness[active_indices]

        cand_colors: List[int] = []
        cand_shapes: List[int] = []
        cand_sizes: List[float] = []
        cand_centroids: List[Tuple[float, float]] = []
        cand_masks: List[Optional[np.ndarray]] = []

        for idx, orig_k in enumerate(active_indices):
            # Color
            if color_preds is not None:
                color = int(color_preds[orig_k].argmax().item())
            else:
                color = 0
            cand_colors.append(color)

            # Shape
            if shape_preds is not None:
                shape_idx = int(shape_preds[orig_k].argmax().item())
            else:
                shape_idx = 0
            cand_shapes.append(shape_idx)

            # Size
            if size_preds is not None:
                size_val = float(size_preds[orig_k].item())
            else:
                size_val = 0.1
            cand_sizes.append(size_val)

            # Centroid
            if pos_preds is not None:
                r_norm = float(pos_preds[orig_k, 0].item())
                c_norm = float(pos_preds[orig_k, 1].item())
                centroid = (r_norm * H, c_norm * W)
            elif attn_maps is not None:
                # Compute centroid from attention map
                attn_2d = attn_maps[orig_k].view(H, W).detach().cpu().numpy()
                total_mass = float(attn_2d.sum())
                if total_mass > 1e-6:
                    r_coords, c_coords = np.mgrid[0:H, 0:W]
                    r_center = float((attn_2d * r_coords).sum() / total_mass)
                    c_center = float((attn_2d * c_coords).sum() / total_mass)
                    centroid = (r_center, c_center)
                else:
                    centroid = (float(H) / 2.0, float(W) / 2.0)
            else:
                centroid = (float(H) / 2.0, float(W) / 2.0)
            cand_centroids.append(centroid)

            # Mask
            if attn_maps is not None:
                cand_masks.append(attn_maps[orig_k].view(H, W).detach().cpu().numpy())
            else:
                cand_masks.append(None)

        M = len(active_indices)
        existing_track_ids = list(self.active_tracks.keys())
        N_exist = len(existing_track_ids)

        if N_exist == 0:
            # First frame: spawn all candidates
            new_tracks: List[TrackedSlot] = []
            for m in range(M):
                t_id = self._next_track_id
                self._next_track_id += 1
                slot_obj = TrackedSlot(
                    track_id=t_id,
                    slot_embedding=cand_slots[m].detach().clone(),
                    predicted_color=cand_colors[m],
                    predicted_shape=cand_shapes[m],
                    predicted_size=cand_sizes[m],
                    centroid=cand_centroids[m],
                    velocity=(0.0, 0.0),
                    motion_direction="STILL",
                    attention_mask=cand_masks[m],
                    lifecycle_state="SPAWNED",
                    confidence=float(cand_obj[m].item()),
                    history=[cand_centroids[m]],
                )
                self.active_tracks[t_id] = slot_obj
                new_tracks.append(slot_obj)
            return new_tracks

        # Hungarian matching between existing tracks and candidates
        cost_matrix = np.zeros((N_exist, M), dtype=np.float32)

        for i, t_id in enumerate(existing_track_ids):
            old_track = self.active_tracks[t_id]
            old_emb = old_track.slot_embedding.unsqueeze(0)  # (1, slot_dim)

            # Cosine distance
            cos_sim = F.cosine_similarity(old_emb, cand_slots, dim=-1).detach().cpu().numpy()  # (M,)
            cos_dist = 1.0 - cos_sim

            for j in range(M):
                # Spatial distance normalized
                dr = (old_track.centroid[0] - cand_centroids[j][0]) / max(H, 1)
                dc = (old_track.centroid[1] - cand_centroids[j][1]) / max(W, 1)
                spatial_dist = np.sqrt(dr * dr + dc * dc)

                # Color penalty
                color_penalty = 0.0 if old_track.predicted_color == cand_colors[j] else 0.5

                cost_matrix[i, j] = float(0.5 * cos_dist[j] + 0.5 * spatial_dist + color_penalty)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks: List[TrackedSlot] = []
        matched_exist = set()
        matched_cands = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            t_id = existing_track_ids[r]

            if cost <= self.match_threshold:
                matched_exist.add(r)
                matched_cands.add(c)

                old_track = self.active_tracks[t_id]
                new_pos = cand_centroids[c]
                velocity = compute_velocity(old_track.centroid, new_pos)
                direction = classify_motion_direction(velocity[0], velocity[1])

                old_track.slot_embedding = cand_slots[c].detach().clone()
                old_track.predicted_color = cand_colors[c]
                old_track.predicted_shape = cand_shapes[c]
                old_track.predicted_size = cand_sizes[c]
                old_track.centroid = new_pos
                old_track.velocity = velocity
                old_track.motion_direction = direction
                old_track.attention_mask = cand_masks[c]
                old_track.lifecycle_state = "ACTIVE"
                old_track.confidence = float(cand_obj[c].item())
                old_track.history.append(new_pos)

                matched_tracks.append(old_track)

        # Handle unmatched existing tracks (DESTROYED)
        for r, t_id in enumerate(existing_track_ids):
            if r not in matched_exist:
                self.active_tracks[t_id].lifecycle_state = "DESTROYED"

        # Handle unmatched candidates (SPAWNED)
        for c in range(M):
            if c not in matched_cands:
                t_id = self._next_track_id
                self._next_track_id += 1
                slot_obj = TrackedSlot(
                    track_id=t_id,
                    slot_embedding=cand_slots[c].detach().clone(),
                    predicted_color=cand_colors[c],
                    predicted_shape=cand_shapes[c],
                    predicted_size=cand_sizes[c],
                    centroid=cand_centroids[c],
                    velocity=(0.0, 0.0),
                    motion_direction="STILL",
                    attention_mask=cand_masks[c],
                    lifecycle_state="SPAWNED",
                    confidence=float(cand_obj[c].item()),
                    history=[cand_centroids[c]],
                )
                self.active_tracks[t_id] = slot_obj
                matched_tracks.append(slot_obj)

        return matched_tracks
