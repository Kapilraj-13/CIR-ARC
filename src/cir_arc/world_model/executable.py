"""Executable World Model (Digital Twin) for forward state simulation in ARC-AGI-3."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import DIRECTION_VECTORS, Action, ActionType


class ExecutableWorldModel:
    """Simulates deterministic environment transitions (s, a) -> s' based on current belief state."""

    def __init__(
        self,
        player_color: int = 9,
        key_colors: Optional[List[int]] = None,
        door_colors: Optional[List[int]] = None,
        goal_colors: Optional[List[int]] = None,
    ) -> None:
        self.player_color = player_color
        self.key_colors = key_colors or [11, 6]
        self.door_colors = door_colors or [8, 7]
        self.goal_colors = goal_colors or [14, 3, 2]
        self.inventory: List[int] = []

    def simulate_step(
        self,
        comp_grid: np.ndarray,
        action: Action,
        belief: Optional[BeliefState] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulates one step forward given current composite grid and action.

        Returns:
            Tuple of (predicted_comp_grid, transition_metadata).
        """
        grid_next = comp_grid.copy()
        H, W = grid_next.shape
        metadata: Dict[str, Any] = {
            "player_moved": False,
            "collected_item": None,
            "unlocked_door": False,
            "reached_goal": False,
        }

        # Locate player
        player_locs = np.argwhere(grid_next == self.player_color)
        if len(player_locs) == 0:
            return grid_next, metadata

        pr, pc = int(player_locs[0][0]), int(player_locs[0][1])
        aid = action.action_id

        if aid in DIRECTION_VECTORS:
            dr, dc = DIRECTION_VECTORS[aid]
            nr, nc = pr + dr, pc + dc

            # Check boundary
            if 0 <= nr < H and 0 <= nc < W:
                target_color = int(grid_next[nr, nc])

                # Check passability
                is_passable = False
                if target_color == 0:
                    is_passable = True
                elif target_color in self.key_colors:
                    # Item collection
                    is_passable = True
                    self.inventory.append(target_color)
                    metadata["collected_item"] = target_color
                elif target_color in self.door_colors:
                    # Door unlock condition
                    if self.inventory:
                        is_passable = True
                        metadata["unlocked_door"] = True
                    else:
                        is_passable = False
                elif target_color in self.goal_colors:
                    is_passable = True
                    metadata["reached_goal"] = True
                elif belief is not None:
                    is_passable = target_color in belief.facts.get_known_passable_colors()
                else:
                    is_passable = False

                if is_passable:
                    grid_next[pr, pc] = 0
                    grid_next[nr, nc] = self.player_color
                    metadata["player_moved"] = True
                    metadata["new_pos"] = (nr, nc)

        return grid_next, metadata
