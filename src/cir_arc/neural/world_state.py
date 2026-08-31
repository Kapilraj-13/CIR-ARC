"""Structured WorldState, Object, and Relational Graph data models for CIR-ARC Phase 2.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


# 14 Canonical ARC Relational Predicates
RELATION_TYPES: List[str] = [
    "LEFT_OF",        # idx 0: obj_a is to the left of obj_b
    "RIGHT_OF",       # idx 1: obj_a is to the right of obj_b
    "ABOVE",          # idx 2: obj_a is above obj_b
    "BELOW",          # idx 3: obj_a is below obj_b
    "TOUCHING",       # idx 4: obj_a and obj_b share at least 1 adjacent boundary pixel
    "OVERLAPPING",    # idx 5: obj_a and obj_b bounding boxes overlap
    "INSIDE",         # idx 6: obj_a bounding box is strictly inside obj_b bounding box
    "CONTAINS",       # idx 7: obj_a contains obj_b
    "NEAR",           # idx 8: centroid Euclidean distance <= 5 cells
    "FAR",            # idx 9: centroid Euclidean distance > 10 cells
    "SAME_COLOR",     # idx 10: obj_a and obj_b have the same dominant color
    "SAME_SHAPE",     # idx 11: obj_a and obj_b share categorical shape class
    "ALIGNED_X",      # idx 12: obj_a and obj_b centroids share the same row (within 1.0 cell)
    "ALIGNED_Y",      # idx 13: obj_a and obj_b centroids share the same col (within 1.0 cell)
]

NUM_RELATIONS: int = len(RELATION_TYPES)


@dataclass
class StructuredObject:
    """Explicit, interpretable object representation extracted by Phase 2.5 perception."""
    slot_id: int
    color: int
    confidence: float
    centroid: Tuple[float, float]                        # (row, col) normalized in [0, 1]
    bbox: Tuple[float, float, float, float]              # (min_r, min_c, max_r, max_c) normalized in [0, 1]
    width: float                                         # Normalized width in [0, 1]
    height: float                                        # Normalized height in [0, 1]
    area: float                                          # Normalized area in [0, 1]
    perimeter: float                                     # Normalized perimeter in [0, 1]
    aspect_ratio: float                                  # Width / Height ratio
    shape_class: int                                     # Categorical shape id (0..7)
    orientation: int                                     # 0 (0°), 1 (90°), 2 (180°), 3 (270°)
    symmetries: Tuple[bool, bool, bool, bool]            # (H-sym, V-sym, MainDiag, AntiDiag)
    has_holes: bool                                      # Topology: presence of enclosed void
    mask: Optional[np.ndarray] = None                    # 2D binary spatial ownership mask (H, W)
    identity_vector: Optional[np.ndarray] = None         # 64-dim metric embedding vector
    raw_slot_vector: Optional[np.ndarray] = None         # 128-dim raw slot vector

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "color": int(self.color),
            "confidence": float(self.confidence),
            "centroid": (float(self.centroid[0]), float(self.centroid[1])),
            "bbox": tuple(float(x) for x in self.bbox),
            "width": float(self.width),
            "height": float(self.height),
            "area": float(self.area),
            "perimeter": float(self.perimeter),
            "aspect_ratio": float(self.aspect_ratio),
            "shape_class": int(self.shape_class),
            "orientation": int(self.orientation),
            "symmetries": tuple(bool(s) for s in self.symmetries),
            "has_holes": bool(self.has_holes),
        }


@dataclass
class SpatialRelation:
    """Explicit directed pairwise relationship between two structured objects."""
    source_id: int
    relation_type: str
    target_id: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": int(self.source_id),
            "relation_type": self.relation_type,
            "target_id": int(self.target_id),
            "confidence": float(self.confidence),
        }

    def __repr__(self) -> str:
        return f"Relation(Obj_{self.source_id} --[{self.relation_type}]--> Obj_{self.target_id}, conf={self.confidence:.2f})"


@dataclass
class RelationGraph:
    """Graph structure containing the complete pairwise relational matrix and edge list."""
    adj_matrix: np.ndarray                               # Shape (K, K, 14) in [0, 1]
    edge_list: List[SpatialRelation] = field(default_factory=list)

    def get_relations_for_object(self, slot_id: int) -> List[SpatialRelation]:
        return [r for r in self.edge_list if r.source_id == slot_id or r.target_id == slot_id]

    def has_relation(self, source_id: int, relation_type: str, target_id: int, threshold: float = 0.5) -> bool:
        if relation_type not in RELATION_TYPES:
            return False
        rel_idx = RELATION_TYPES.index(relation_type)
        if source_id < self.adj_matrix.shape[0] and target_id < self.adj_matrix.shape[1]:
            return bool(self.adj_matrix[source_id, target_id, rel_idx] >= threshold)
        return False


@dataclass
class WorldState:
    """The canonical interface contract between Phase 2.5 Perception and Downstream Reasoning."""
    objects: List[StructuredObject]                      # Active, high-confidence detected objects
    relations: List[SpatialRelation]                     # Active high-confidence relational edges
    relation_graph: RelationGraph                        # Complete relational matrix
    raw_grid: np.ndarray                                 # Underlying discrete 2D grid
    grid_shape: Tuple[int, int]                          # (H, W)
    frame_index: int = 0
    boundary_map: Optional[np.ndarray] = None            # (H, W) boundary probabilities
    cell_objectness: Optional[np.ndarray] = None         # (H, W) cell foreground objectness
    global_features: Optional[np.ndarray] = None         # Summary context embedding

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def get_object_by_id(self, slot_id: int) -> Optional[StructuredObject]:
        for obj in self.objects:
            if obj.slot_id == slot_id:
                return obj
        return None

    def get_objects_by_color(self, color: int) -> List[StructuredObject]:
        return [obj for obj in self.objects if obj.color == color]

    def summary(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "grid_shape": self.grid_shape,
            "num_objects": len(self.objects),
            "num_relations": len(self.relations),
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [r.to_dict() for r in self.relations[:20]],
        }
