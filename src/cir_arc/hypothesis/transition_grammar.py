from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cir_arc.environment.actions import Action, ActionType


@dataclass
class TransitionRule:
    rule_id: str
    action_trigger: int
    condition_type: str  # 'ALWAYS', 'OBJECT_PRESENT', 'NEAR_TARGET'
    effect_type: str     # 'TRANSLATE_PLAYER', 'REMOVE_OBJECT', 'CHANGE_COLOR', 'OPEN_DOOR'
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    support_count: int = 1
    refutation_count: int = 0

    def matches(self, action: Action, state_context: Optional[Dict[str, Any]] = None) -> bool:
        if action.action_id != self.action_trigger:
            return False
        if self.condition_type == "ALWAYS":
            return True
        if self.condition_type == "OBJECT_PRESENT":
            req_color = self.params.get("color")
            if state_context and "colors" in state_context:
                return req_color in state_context["colors"]
        return True

    def apply_to_grid(self, grid: np.ndarray, player_color: int = 9) -> np.ndarray:
        out = grid.copy()
        if self.effect_type == "TRANSLATE_PLAYER":
            dr = int(self.params.get("dr", 0))
            dc = int(self.params.get("dc", 0))
            player_locs = np.argwhere(out == player_color)
            if len(player_locs) > 0:
                pr, pc = player_locs[0]
                nr, nc = pr + dr, pc + dc
                h, w = out.shape
                if 0 <= nr < h and 0 <= nc < w:
                    out[pr, pc] = 0
                    out[nr, nc] = player_color
        elif self.effect_type == "REMOVE_OBJECT":
            target_color = self.params.get("target_color")
            if target_color is not None:
                out[out == target_color] = 0
        elif self.effect_type == "CHANGE_COLOR":
            src = self.params.get("src_color")
            dst = self.params.get("dst_color")
            if src is not None and dst is not None:
                out[out == src] = dst
        return out
