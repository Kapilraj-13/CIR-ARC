from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cir_arc.environment.actions import ACTION_NAMES, Action
from cir_arc.environment.state_delta import StateDelta


@dataclass
class ActionEffect:
    action_id: int
    name: str
    total_probes: int = 0
    change_count: int = 0
    mean_pixel_diff: float = 0.0
    is_reversible: bool = False
    movement_vector: Optional[Tuple[float, float]] = None

    @property
    def effect_rate(self) -> float:
        return float(self.change_count) / max(self.total_probes, 1)

    @property
    def causes_frame_change(self) -> bool:
        return self.change_count > 0


class ActionEffectMatrix:
    """Maintains empirical effect probabilities, reversibility, and dynamics per action."""

    def __init__(self) -> None:
        self.effects: Dict[int, ActionEffect] = {}

    def record_probe(self, action: Action, delta: StateDelta) -> None:
        aid = action.action_id
        if aid not in self.effects:
            self.effects[aid] = ActionEffect(
                action_id=aid,
                name=ACTION_NAMES.get(aid, action.name),
            )

        eff = self.effects[aid]
        eff.total_probes += 1
        if delta.has_mutation:
            eff.change_count += 1

        # Running average pixel diff
        eff.mean_pixel_diff = (
            (eff.mean_pixel_diff * (eff.total_probes - 1) + delta.pixel_diff_count)
            / eff.total_probes
        )

        # Estimate movement vector if objects moved
        if delta.moved_objects:
            dr, dc = delta.moved_objects[0][2]
            eff.movement_vector = (dr, dc)

    def mark_reversibility(self, action_id: int, is_rev: bool) -> None:
        if action_id in self.effects:
            self.effects[action_id].is_reversible = is_rev

    def get_active_actions(self) -> List[int]:
        """Return action IDs that have caused a frame change."""
        return [aid for aid, eff in self.effects.items() if eff.causes_frame_change]

    def to_dict(self) -> Dict[str, Any]:
        return {
            str(aid): {
                "action_id": eff.action_id,
                "name": eff.name,
                "total_probes": eff.total_probes,
                "change_count": eff.change_count,
                "effect_rate": eff.effect_rate,
                "mean_pixel_diff": eff.mean_pixel_diff,
                "is_reversible": eff.is_reversible,
                "movement_vector": eff.movement_vector,
            }
            for aid, eff in self.effects.items()
        }
