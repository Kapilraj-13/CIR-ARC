"""Cycle and deadlock detection with exploratory breakout rollbacks."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from cir_arc.environment.actions import Action, ActionType


class StateRollback:
    """Detects oscillation cycles and triggers breakout exploration actions."""

    def __init__(self, history_window: int = 6) -> None:
        self.history_window = history_window
        self.recent_positions: deque[Tuple[int, int]] = deque(maxlen=history_window)

    def record_position(self, pos: Tuple[int, int]) -> None:
        self.recent_positions.append(pos)

    def is_in_cycle(self) -> bool:
        """Returns True if agent is oscillating between a small set of positions repeatedly."""
        if len(self.recent_positions) < self.history_window:
            return False
        unique_positions = set(self.recent_positions)
        # If last 6 steps only touched <= 2 distinct coordinates, we are trapped/oscillating
        return len(unique_positions) <= 2

    def get_breakout_action(self, available_actions: List[int]) -> Action:
        """Selects a novel action to break out of oscillation cycle."""
        # Pick orthogonal action
        for act_code in [2, 3, 0, 1]:
            if act_code in available_actions:
                return Action.from_int(act_code)
        return Action(ActionType.ACTION1)
