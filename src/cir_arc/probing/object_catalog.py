from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cir_arc.environment.state_delta import GridObject, StateDelta


@dataclass
class ObjectArchetype:
    object_id: str
    color: int
    layer: int
    pixel_count: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    is_controllable: bool = False  # Controlled by player actions
    is_static: bool = True         # Never moves across probes
    is_collectible: bool = False   # Vanishes on contact / interact
    is_door: bool = False          # Mutates or opens on interact/click
    is_target: bool = False        # Goal state object
    move_count: int = 0


class DynamicObjectCatalog:
    """Classifies objects into behavioral archetypes based on probe observations."""

    def __init__(self) -> None:
        self.catalog: Dict[str, ObjectArchetype] = {}

    def register_objects(self, objects: List[GridObject]) -> None:
        for obj in objects:
            if obj.object_id not in self.catalog:
                self.catalog[obj.object_id] = ObjectArchetype(
                    object_id=obj.object_id,
                    color=obj.color,
                    layer=obj.layer,
                    pixel_count=obj.pixel_count,
                    bbox=obj.bbox,
                    centroid=obj.centroid,
                )

    def update_from_delta(self, delta: StateDelta) -> None:
        self.register_objects(delta.objects_before)
        self.register_objects(delta.objects_after)

        # Controllability detection: if directional action moves this object
        for ob, oa, (dr, dc) in delta.moved_objects:
            if ob.object_id in self.catalog:
                arch = self.catalog[ob.object_id]
                arch.move_count += 1
                arch.is_static = False
                if delta.action_taken.action_id in (1, 2, 3, 4):
                    arch.is_controllable = True

        # Collectible / door detection: destroyed objects
        for obj in delta.destroyed_objects:
            if obj.object_id in self.catalog:
                self.catalog[obj.object_id].is_collectible = True

    def get_player(self) -> Optional[ObjectArchetype]:
        for arch in self.catalog.values():
            if arch.is_controllable:
                return arch
        return None

    def get_static_barriers(self) -> List[ObjectArchetype]:
        return [arch for arch in self.catalog.values() if arch.is_static and arch.pixel_count > 1]

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "object_id": a.object_id,
                "color": a.color,
                "layer": a.layer,
                "pixel_count": a.pixel_count,
                "bbox": list(a.bbox),
                "centroid": list(a.centroid),
                "is_controllable": a.is_controllable,
                "is_static": a.is_static,
                "is_collectible": a.is_collectible,
            }
            for a in self.catalog.values()
        ]
