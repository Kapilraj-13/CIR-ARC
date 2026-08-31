"""Relational Graph Prediction Head and Ground-Truth Relation Extractor for Phase 2.5."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject
from cir_arc.neural.world_state import RELATION_TYPES, NUM_RELATIONS, SpatialRelation, RelationGraph


class RelationalGraphHead(nn.Module):
    """Predicts explicit pairwise spatial and semantic relationships between object slots.

    Maps slot pairs (s_i, s_j) -> 14-dimensional multi-label relational probability vectors:
    (B, K, slot_dim) -> (B, K, K, 14)

    Models 14 relation classes:
    - Directional: LEFT_OF, RIGHT_OF, ABOVE, BELOW
    - Topological: TOUCHING, OVERLAPPING, INSIDE, CONTAINS
    - Proximity: NEAR, FAR
    - Attribute & Alignment: SAME_COLOR, SAME_SHAPE, ALIGNED_X, ALIGNED_Y
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 128,
        num_relations: int = NUM_RELATIONS,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.num_relations = num_relations

        # Pairwise relational MLP: processes [s_i, s_j, s_i - s_j, s_i * s_j]
        in_dim = slot_dim * 4
        self.relation_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_relations),
        )

    def forward(
        self,
        slots: torch.Tensor,
        objectness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting relational logits between all slot pairs.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).
            objectness: Optional slot presence probabilities of shape (B, K).

        Returns:
            Pairwise relational logits of shape (B, K, K, num_relations).
        """
        B, K, D = slots.shape

        # Expand slot tensors to form all directed pairs (s_i, s_j)
        slots_i = slots.unsqueeze(2).expand(B, K, K, D)  # (B, K, K, D)
        slots_j = slots.unsqueeze(1).expand(B, K, K, D)  # (B, K, K, D)

        diff = slots_i - slots_j
        prod = slots_i * slots_j

        pair_feats = torch.cat([slots_i, slots_j, diff, prod], dim=-1)  # (B, K, K, 4*D)
        logits = self.relation_mlp(pair_feats)  # (B, K, K, num_relations)

        return logits

    def predict_graph(
        self,
        slots: torch.Tensor,
        objectness: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        obj_threshold: float = 0.4,
    ) -> List[RelationGraph]:
        """Inference helper converting neural slot predictions into structured RelationGraph objects."""
        with torch.no_grad():
            logits = self.forward(slots, objectness=objectness)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, K, K, 14)

        if objectness is not None:
            obj_np = objectness.detach().cpu().numpy()
        else:
            obj_np = np.ones((slots.shape[0], slots.shape[1]), dtype=np.float32)

        B, K, _, num_rels = probs.shape
        graphs: List[RelationGraph] = []

        for b in range(B):
            adj_mat = probs[b]
            edges: List[SpatialRelation] = []

            for i in range(K):
                if obj_np[b, i] < obj_threshold:
                    continue
                for j in range(K):
                    if i == j or obj_np[b, j] < obj_threshold:
                        continue
                    for r_idx in range(num_rels):
                        conf = float(adj_mat[i, j, r_idx])
                        if conf >= threshold:
                            edges.append(
                                SpatialRelation(
                                    source_id=i,
                                    relation_type=RELATION_TYPES[r_idx],
                                    target_id=j,
                                    confidence=conf,
                                )
                            )
            graphs.append(RelationGraph(adj_matrix=adj_mat, edge_list=edges))

        return graphs


def extract_ground_truth_relations(
    gt_objects: List[ArcObject],
    H: int,
    W: int,
) -> np.ndarray:
    """Computes exact ground-truth relational adjacency matrix from ArcObject instances.

    Args:
        gt_objects: List of M ground-truth ArcObject instances on grid.
        H: Grid height.
        W: Grid width.

    Returns:
        Binary numpy array of shape (M, M, 14) containing all ground-truth relations.
    """
    M = len(gt_objects)
    gt_matrix = np.zeros((M, M, NUM_RELATIONS), dtype=np.float32)

    if M <= 1:
        return gt_matrix

    for i in range(M):
        obj_a = gt_objects[i]
        ra, ca = obj_a.centroid
        min_ra, min_ca, max_ra, max_ca = obj_a.bounding_box
        pixels_a = set(map(tuple, obj_a.pixels))

        for j in range(M):
            if i == j:
                continue
            obj_b = gt_objects[j]
            rb, cb = obj_b.centroid
            min_rb, min_cb, max_rb, max_cb = obj_b.bounding_box
            pixels_b = set(map(tuple, obj_b.pixels))

            # 0: LEFT_OF (col_a < col_b by at least 1 cell)
            if ca < cb - 0.5:
                gt_matrix[i, j, 0] = 1.0

            # 1: RIGHT_OF (col_a > col_b by at least 1 cell)
            if ca > cb + 0.5:
                gt_matrix[i, j, 1] = 1.0

            # 2: ABOVE (row_a < row_b by at least 1 cell)
            if ra < rb - 0.5:
                gt_matrix[i, j, 2] = 1.0

            # 3: BELOW (row_a > row_b by at least 1 cell)
            if ra > rb + 0.5:
                gt_matrix[i, j, 3] = 1.0

            # 4: TOUCHING (any 4-connected boundary adjacent pixels)
            is_touching = False
            for r, c in pixels_a:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r + dr, c + dc) in pixels_b:
                        is_touching = True
                        break
                if is_touching:
                    break
            if is_touching:
                gt_matrix[i, j, 4] = 1.0

            # 5: OVERLAPPING (bounding boxes intersect)
            has_overlap = not (max_ra < min_rb or min_ra > max_rb or max_ca < min_cb or min_ca > max_cb)
            if has_overlap:
                gt_matrix[i, j, 5] = 1.0

            # 6: INSIDE (obj_a bbox strictly inside obj_b bbox)
            is_inside = (min_ra >= min_rb and max_ra <= max_rb and min_ca >= min_cb and max_ca <= max_cb)
            if is_inside and not (min_ra == min_rb and max_ra == max_rb and min_ca == min_cb and max_ca == max_cb):
                gt_matrix[i, j, 6] = 1.0

            # 7: CONTAINS (obj_a contains obj_b)
            if gt_matrix[j, i, 6] == 1.0:
                gt_matrix[i, j, 7] = 1.0

            # Centroid Euclidean distance in cells
            dist = np.sqrt((ra - rb) ** 2 + (ca - cb) ** 2)

            # 8: NEAR (distance <= 5 cells)
            if dist <= 5.0:
                gt_matrix[i, j, 8] = 1.0

            # 9: FAR (distance > 10 cells)
            if dist > 10.0:
                gt_matrix[i, j, 9] = 1.0

            # 10: SAME_COLOR
            if obj_a.color == obj_b.color:
                gt_matrix[i, j, 10] = 1.0

            # 11: SAME_SHAPE
            shape_a = getattr(obj_a, "is_square", False) or getattr(obj_a, "is_rectangle", False)
            shape_b = getattr(obj_b, "is_square", False) or getattr(obj_b, "is_rectangle", False)
            if (obj_a.height, obj_a.width) == (obj_b.height, obj_b.width) or (shape_a and shape_a == shape_b):
                gt_matrix[i, j, 11] = 1.0

            # 12: ALIGNED_X (same row within 1 cell)
            if abs(ra - rb) <= 1.0:
                gt_matrix[i, j, 12] = 1.0

            # 13: ALIGNED_Y (same column within 1 cell)
            if abs(ca - cb) <= 1.0:
                gt_matrix[i, j, 13] = 1.0

    return gt_matrix
