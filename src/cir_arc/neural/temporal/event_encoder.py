"""Categorical Event Encoder and Compressed Temporal Event Memory for CIR-ARC.

Transforms frame-to-frame state and grid differences into discrete, typed SemanticEvents:
- MOVE: Centroid displacement above threshold
- APPEAR / SPAWN: Emergence of a previously unseen object entity
- DISAPPEAR / DESTROY: Vanishing of an active object entity
- COLLIDE: Motion leading to boundary contact or overlap
- TOGGLE: State mutation in interactive switches, gates, or buttons
- PICKUP: Collectible disappearance at agent coordinates
- SHIFT / ROTATE: Orientation or shape class mutation
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.neural.world_state import EVENT_TYPES, SemanticEvent, StructuredObject


class CategoricalEventEncoder:
    """Classifies discrete semantic events occurring between consecutive scene frames."""

    def __init__(
        self,
        motion_threshold: float = 0.05,
        contact_threshold: float = 0.08,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.contact_threshold = contact_threshold

    def encode_events(
        self,
        prev_objects: List[StructuredObject],
        curr_objects: List[StructuredObject],
        step: int = 0,
        action_id: Optional[int] = None,
        grid_diff: Optional[np.ndarray] = None,
    ) -> List[SemanticEvent]:
        """Compares previous and current object sets to deduce categorical events.

        Args:
            prev_objects: Objects from time t-1.
            curr_objects: Objects from time t.
            step: Current environment step count.
            action_id: Action executed between frames.
            grid_diff: Optional 2D binary grid difference map (H, W).

        Returns:
            List of detected SemanticEvent instances.
        """
        events: List[SemanticEvent] = []
        prev_by_id = {obj.slot_id: obj for obj in prev_objects}
        curr_by_id = {obj.slot_id: obj for obj in curr_objects}

        # 1. Detect SPAWN / APPEAR
        for slot_id, curr_obj in curr_by_id.items():
            if slot_id not in prev_by_id:
                events.append(
                    SemanticEvent(
                        event_type="SPAWN",
                        source_id=slot_id,
                        step=step,
                        confidence=curr_obj.confidence,
                        details={"color": curr_obj.color, "centroid": curr_obj.centroid},
                    )
                )

        # 2. Detect DESTROY / DISAPPEAR / PICKUP
        for slot_id, prev_obj in prev_by_id.items():
            if slot_id not in curr_by_id:
                # If an action was taken and another object is near, might be PICKUP
                is_pickup = False
                for other in curr_objects:
                    dr = abs(other.centroid[0] - prev_obj.centroid[0])
                    dc = abs(other.centroid[1] - prev_obj.centroid[1])
                    if dr <= self.contact_threshold and dc <= self.contact_threshold:
                        is_pickup = True
                        events.append(
                            SemanticEvent(
                                event_type="PICKUP",
                                source_id=other.slot_id,
                                target_id=slot_id,
                                step=step,
                                confidence=0.9,
                                details={"item_color": prev_obj.color},
                            )
                        )
                        break

                if not is_pickup:
                    events.append(
                        SemanticEvent(
                            event_type="DESTROY",
                            source_id=slot_id,
                            step=step,
                            confidence=prev_obj.confidence,
                            details={"color": prev_obj.color},
                        )
                    )

        # 3. Detect MOVE, COLLIDE, ROTATE, TOGGLE for surviving objects
        for slot_id, curr_obj in curr_by_id.items():
            if slot_id in prev_by_id:
                prev_obj = prev_by_id[slot_id]
                dr = curr_obj.centroid[0] - prev_obj.centroid[0]
                dc = curr_obj.centroid[1] - prev_obj.centroid[1]
                dist = np.sqrt(dr * dr + dc * dc)

                # Move event
                if dist >= self.motion_threshold:
                    events.append(
                        SemanticEvent(
                            event_type="MOVE",
                            source_id=slot_id,
                            step=step,
                            confidence=min(1.0, float(dist * 5.0)),
                            details={
                                "delta_r": float(dr),
                                "delta_c": float(dc),
                                "distance": float(dist),
                            },
                        )
                    )

                    # Collision check against other surviving objects
                    for other_id, other_obj in curr_by_id.items():
                        if other_id != slot_id:
                            sep_r = abs(curr_obj.centroid[0] - other_obj.centroid[0])
                            sep_c = abs(curr_obj.centroid[1] - other_obj.centroid[1])
                            if sep_r <= self.contact_threshold and sep_c <= self.contact_threshold:
                                events.append(
                                    SemanticEvent(
                                        event_type="COLLIDE",
                                        source_id=slot_id,
                                        target_id=other_id,
                                        step=step,
                                        confidence=0.85,
                                        details={"touching_slot": other_id},
                                    )
                                )

                # Orientation / Rotation event
                if curr_obj.orientation != prev_obj.orientation:
                    events.append(
                        SemanticEvent(
                            event_type="ROTATE",
                            source_id=slot_id,
                            step=step,
                            confidence=0.95,
                            details={
                                "prev_orient": prev_obj.orientation,
                                "curr_orient": curr_obj.orientation,
                            },
                        )
                    )

                # Color / State Toggle
                if curr_obj.color != prev_obj.color:
                    events.append(
                        SemanticEvent(
                            event_type="TOGGLE",
                            source_id=slot_id,
                            step=step,
                            confidence=0.98,
                            details={
                                "prev_color": prev_obj.color,
                                "curr_color": curr_obj.color,
                            },
                        )
                    )

        return events


class TemporalEventMemory:
    """Episodic compressed memory buffer storing action-event trajectories."""

    def __init__(self, max_history: int = 150) -> None:
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)

    def record_step(
        self,
        step: int,
        action_id: int,
        events: List[SemanticEvent],
        state_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append compressed step record."""
        entry = {
            "step": step,
            "action_id": action_id,
            "events": [e.to_dict() for e in events],
            "state_summary": state_summary or {},
        }
        self.history.append(entry)

    def get_recent_events(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent k step records."""
        items = list(self.history)
        return items[-k:] if k <= len(items) else items

    def count_event_types(self) -> Dict[str, int]:
        """Aggregate event frequencies across episodic memory."""
        counts: Dict[str, int] = {e: 0 for e in EVENT_TYPES}
        for entry in self.history:
            for ev in entry.get("events", []):
                t = ev.get("event_type")
                if t in counts:
                    counts[t] += 1
        return counts

    def clear(self) -> None:
        self.history.clear()
