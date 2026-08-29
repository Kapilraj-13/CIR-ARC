"""Plan contradiction detection identifying broken causal assumptions during plan execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData


class PlanContradictionDetector:
    """Detects when an executed action failed to produce the expected spatial or state delta."""

    @staticmethod
    def check_movement_contradiction(
        before_frame: FrameData,
        action: Action,
        after_frame: FrameData,
        expected_movement: bool = True,
    ) -> bool:
        """Returns True if movement was expected but player remained in the same position."""
        if not expected_movement:
            return False

        before_grid_hash = before_frame.grid.hash()
        after_grid_hash = after_frame.grid.hash()

        # If grid hash is identical after taking a directional action (1..4), movement was blocked!
        if before_grid_hash == after_grid_hash and action.action_id in (1, 2, 3, 4):
            return True
        return False
