from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData
from cir_arc.environment.state_delta import StateDelta, compute_state_delta


@dataclass
class SymbolicDelta:
    action_id: int
    has_mutation: bool
    pixel_diff_count: int
    movement_vectors: List[Dict[str, Any]] = field(default_factory=list)
    spawned_archetypes: List[Dict[str, Any]] = field(default_factory=list)
    destroyed_archetypes: List[Dict[str, Any]] = field(default_factory=list)
    property_changes: List[Dict[str, Any]] = field(default_factory=list)


class DeltaMapper:
    """Converts low-level StateDelta instances into symbolic transition predicates."""

    @staticmethod
    def map_delta(delta: StateDelta) -> SymbolicDelta:
        movements: List[Dict[str, Any]] = []
        for ob, oa, (dr, dc) in delta.moved_objects:
            movements.append({
                "object_id": ob.object_id,
                "color": ob.color,
                "layer": ob.layer,
                "dr": dr,
                "dc": dc,
                "old_centroid": ob.centroid,
                "new_centroid": oa.centroid,
            })

        spawned = [
            {"object_id": o.object_id, "color": o.color, "layer": o.layer, "pixel_count": o.pixel_count}
            for o in delta.spawned_objects
        ]
        destroyed = [
            {"object_id": o.object_id, "color": o.color, "layer": o.layer, "pixel_count": o.pixel_count}
            for o in delta.destroyed_objects
        ]
        mutations = [
            {"object_id": m.object_id, "prop": m.property_name, "old": m.old_value, "new": m.new_value}
            for m in delta.mutations
        ]

        return SymbolicDelta(
            action_id=delta.action_taken.action_id,
            has_mutation=delta.has_mutation,
            pixel_diff_count=delta.pixel_diff_count,
            movement_vectors=movements,
            spawned_archetypes=spawned,
            destroyed_archetypes=destroyed,
            property_changes=mutations,
        )

    @classmethod
    def from_frames(cls, before: FrameData, after: FrameData, action: Action) -> SymbolicDelta:
        delta = compute_state_delta(before, after, action)
        return cls.map_delta(delta)
