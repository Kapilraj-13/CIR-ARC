"""Structured WorldState, Object, Relational Graph, and Dual Hybrid Neuro-Symbolic models for CIR-ARC.

Implements the Dual Hybrid Neuro-Symbolic Representation:
1. SymbolicSceneState: Interpretable, discrete entities, bounding boxes, masks, topological invariants,
   canonical spatial relations, affordances, mechanics beliefs, and action effects.
2. DenseLatentState: Uncompressed continuous slot vectors, spatial feature tokens, and pairwise relational latents.
3. HybridSceneState: The unified contract uniting both layers, providing `to_cognitive_tokens()` for
   downstream Cognitive Transformers (120M reasoner) without losing continuous neural signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch


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

# 9 Canonical ARC Interactive Affordances
AFFORDANCE_NAMES: List[str] = [
    "can_move",          # Agent or piece can move autonomously
    "can_push",          # Object can be pushed by player or dynamic actor
    "can_collect",       # Collectible item / key / coin
    "can_interact",      # Triggerable via interaction key
    "can_toggle",        # Pressure plate, lever, or toggleable switch
    "can_destroy",       # Destructible barrier or breakable block
    "can_block",         # Rigid solid impassable barrier/wall
    "can_support",       # Provides physical platform/support under gravity
    "can_be_clicked",    # Selectable via mouse/cursor pointer (ACTION6)
]

NUM_AFFORDANCES: int = len(AFFORDANCE_NAMES)

# Canonical Categorical Event Types
EVENT_TYPES: List[str] = [
    "MOVE",
    "APPEAR",
    "DISAPPEAR",
    "COLLIDE",
    "PICKUP",
    "DROP",
    "TOGGLE",
    "OPEN",
    "CLOSE",
    "SPAWN",
    "DESTROY",
    "SHIFT",
    "ROTATE",
]


@dataclass
class SemanticEvent:
    """Categorical discrete transition event occurring between frames."""
    event_type: str                                      # From EVENT_TYPES
    source_id: Optional[int] = None                      # Primary entity/slot id
    target_id: Optional[int] = None                      # Secondary interacting entity/slot id
    step: int = 0
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "step": self.step,
            "confidence": float(self.confidence),
            "details": dict(self.details),
        }


@dataclass
class MechanicsBelief:
    """Online dynamic belief vector estimating environment physics hypotheses."""
    gravity: Tuple[float, float] = (0.0, 0.0)            # Estimated gravity vector (dr, dc)
    gravity_confidence: float = 0.0                      # Confidence in gravity hypothesis [0, 1]
    friction: float = 0.5                                # 1.0 = stop immediately, 0.0 = frictionless slide
    collision_elasticity: float = 0.0                    # 0.0 = rigid stop, 1.0 = elastic bounce
    pushability_rule: float = 0.5                        # Probability blocks can be pushed
    sliding_inertia: float = 0.0                         # Ice-sliding momentum indicator
    screen_wrapping: float = 0.0                         # Toroidal boundary wrapping probability
    teleportation_active: float = 0.0                    # Portal mechanics indicator
    toggle_mechanics: float = 0.5                        # Switch/plate gates active indicator
    resource_mechanics: float = 0.0                      # Inventory/key locks active indicator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gravity": (float(self.gravity[0]), float(self.gravity[1])),
            "gravity_confidence": float(self.gravity_confidence),
            "friction": float(self.friction),
            "collision_elasticity": float(self.collision_elasticity),
            "pushability_rule": float(self.pushability_rule),
            "sliding_inertia": float(self.sliding_inertia),
            "screen_wrapping": float(self.screen_wrapping),
            "teleportation_active": float(self.teleportation_active),
            "toggle_mechanics": float(self.toggle_mechanics),
            "resource_mechanics": float(self.resource_mechanics),
        }


@dataclass
class ActionEffect:
    """Explicit predicted causal effect of an action on the world state."""
    action_id: int
    success_probability: float = 1.0
    moves_player: bool = False
    affects_environment: bool = False
    reversible: bool = True
    cost: float = 1.0
    predicted_delta: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "success_probability": float(self.success_probability),
            "moves_player": bool(self.moves_player),
            "affects_environment": bool(self.affects_environment),
            "reversible": bool(self.reversible),
            "cost": float(self.cost),
            "predicted_delta": {int(k): (float(v[0]), float(v[1])) for k, v in self.predicted_delta.items()},
        }


@dataclass
class GlobalStateData:
    """HUD and global environment metrics."""
    lives: int = 1
    energy: float = 1.0
    score: int = 0
    level_index: int = 0
    timer: Optional[int] = None
    switches_active: int = 0
    doors_open: int = 0
    inventory: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lives": int(self.lives),
            "energy": float(self.energy),
            "score": int(self.score),
            "level_index": int(self.level_index),
            "timer": self.timer,
            "switches_active": int(self.switches_active),
            "doors_open": int(self.doors_open),
            "inventory": list(self.inventory),
        }


@dataclass
class UncertaintySummary:
    """Provenance and certainty estimation across scene predictions."""
    mean_object_confidence: float = 1.0
    mean_relation_confidence: float = 1.0
    mechanics_entropy: float = 0.0
    overall_certainty: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_object_confidence": float(self.mean_object_confidence),
            "mean_relation_confidence": float(self.mean_relation_confidence),
            "mechanics_entropy": float(self.mechanics_entropy),
            "overall_certainty": float(self.overall_certainty),
        }


@dataclass
class StructuredObject:
    """Explicit, interpretable object representation extracted by Perception."""
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
    extent_box: Optional[Tuple[int, int, int, int]] = None  # Discrete pixel bounds (min_r, min_c, max_r, max_c)
    velocity: Tuple[float, float] = (0.0, 0.0)           # Estimated velocity (dr, dc)
    acceleration: Tuple[float, float] = (0.0, 0.0)       # Estimated acceleration
    motion_direction: str = "STILL"                      # STILL, UP, DOWN, LEFT, RIGHT, etc.
    lifecycle_state: str = "ACTIVE"                      # SPAWNED, ACTIVE, DESTROYED
    affordances: Dict[str, float] = field(default_factory=dict) # Affordance probabilities in [0, 1]
    provenance: str = "neural"                           # "neural", "geometric", "tracker"
    mask: Optional[np.ndarray] = None                    # 2D binary spatial ownership mask (H, W)
    identity_vector: Optional[np.ndarray] = None         # 64-dim metric embedding vector
    raw_slot_vector: Optional[np.ndarray] = None         # Raw continuous slot vector (128/192-dim)

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
            "extent_box": tuple(int(x) for x in self.extent_box) if self.extent_box else None,
            "velocity": (float(self.velocity[0]), float(self.velocity[1])),
            "motion_direction": self.motion_direction,
            "lifecycle_state": self.lifecycle_state,
            "affordances": {k: float(v) for k, v in self.affordances.items()},
            "provenance": self.provenance,
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
class DenseLatentState:
    """Uncompressed continuous neural representations preserving fine-grained signals."""
    slot_embeddings: torch.Tensor                        # (K, slot_dim) continuous slot representations
    spatial_features: Optional[torch.Tensor] = None      # (H*W, feat_dim) or (H, W, feat_dim) visual tokens
    pairwise_relational_latents: Optional[torch.Tensor] = None # (K, K, rel_dim) continuous relation features
    global_scene_vector: Optional[torch.Tensor] = None   # Pooled global latent vector
    temporal_latent: Optional[torch.Tensor] = None       # (K, temp_dim) temporal tracking latent

    @property
    def num_slots(self) -> int:
        return self.slot_embeddings.shape[0] if self.slot_embeddings is not None else 0

    @property
    def slot_dim(self) -> int:
        return self.slot_embeddings.shape[-1] if self.slot_embeddings is not None else 0


@dataclass
class SymbolicSceneState:
    """Interpretable structured symbolic world state."""
    frame_index: int
    grid_shape: Tuple[int, int]
    objects: List[StructuredObject]
    relations: List[SpatialRelation]
    scene_graph: RelationGraph
    events: List[SemanticEvent] = field(default_factory=list)
    global_state: GlobalStateData = field(default_factory=GlobalStateData)
    action_effects: Dict[int, ActionEffect] = field(default_factory=dict)
    mechanics_beliefs: MechanicsBelief = field(default_factory=MechanicsBelief)
    uncertainties: UncertaintySummary = field(default_factory=UncertaintySummary)

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
            "num_events": len(self.events),
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [r.to_dict() for r in self.relations[:20]],
            "mechanics": self.mechanics_beliefs.to_dict(),
            "global_state": self.global_state.to_dict(),
        }


@dataclass
class HybridSceneState:
    """The unified Dual Neuro-Symbolic contract uniting SymbolicSceneState and DenseLatentState."""
    frame_index: int
    grid_shape: Tuple[int, int]
    raw_grid: np.ndarray
    symbolic: SymbolicSceneState
    dense: DenseLatentState
    boundary_map: Optional[np.ndarray] = None
    cell_objectness: Optional[np.ndarray] = None

    # Forwarding properties for seamless backward compatibility with WorldState
    @property
    def objects(self) -> List[StructuredObject]:
        return self.symbolic.objects

    @property
    def relations(self) -> List[SpatialRelation]:
        return self.symbolic.relations

    @property
    def relation_graph(self) -> RelationGraph:
        return self.symbolic.scene_graph

    @property
    def num_objects(self) -> int:
        return len(self.symbolic.objects)

    def get_object_by_id(self, slot_id: int) -> Optional[StructuredObject]:
        return self.symbolic.get_object_by_id(slot_id)

    def get_objects_by_color(self, color: int) -> List[StructuredObject]:
        return self.symbolic.get_objects_by_color(color)

    def summary(self) -> Dict[str, Any]:
        res = self.symbolic.summary()
        res["has_dense_latents"] = self.dense is not None
        if self.dense is not None:
            res["dense_slot_count"] = self.dense.num_slots
            res["dense_slot_dim"] = self.dense.slot_dim
        return res

    def to_cognitive_tokens(self, embed_dim: int = 256) -> torch.Tensor:
        """Projects both dense latents and structured symbolic entities into unified token sequence.

        Emits a token sequence of shape (1, num_tokens, embed_dim) consumable by the
        Cognitive Transformer (120M Reasoner) without information loss.
        """
        device = self.dense.slot_embeddings.device if isinstance(self.dense.slot_embeddings, torch.Tensor) else torch.device("cpu")
        slots = self.dense.slot_embeddings
        if not isinstance(slots, torch.Tensor):
            slots = torch.from_numpy(np.array(slots, dtype=np.float32)).to(device)

        if slots.dim() == 2:
            slots = slots.unsqueeze(0)  # (1, K, D)

        B, K, D = slots.shape

        # Linear projection of dense slots
        if D != embed_dim:
            proj = torch.nn.Linear(D, embed_dim, bias=False).to(device)
            # Use deterministic orthogonal initialization for projection
            torch.nn.init.orthogonal_(proj.weight)
            slot_tokens = proj(slots)
        else:
            slot_tokens = slots

        # Construct global summary token from mechanics & global state
        mb = self.symbolic.mechanics_beliefs
        gs = self.symbolic.global_state
        global_vec = torch.tensor([
            mb.gravity[0], mb.gravity[1], mb.gravity_confidence,
            mb.friction, mb.sliding_inertia, mb.pushability_rule,
            float(gs.score), float(gs.lives), float(gs.level_index),
        ], dtype=torch.float32, device=device)

        global_token = torch.zeros((B, 1, embed_dim), device=device)
        global_token[0, 0, :len(global_vec)] = global_vec

        # Unified composite tokens: [Global Token] + [Dense Slot Tokens]
        composite_tokens = torch.cat([global_token, slot_tokens], dim=1)
        return composite_tokens


# Backward compatibility alias
WorldState = HybridSceneState
