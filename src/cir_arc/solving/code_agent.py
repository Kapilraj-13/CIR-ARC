from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData
from cir_arc.solving.search_solvers import AStarSolver

logger = logging.getLogger(__name__)


class CodeAgentSolver:
    """Algorithmic solver synthesizing action sequences using structured spatial heuristics."""

    def __init__(self, player_color: int = 9, goal_color: int = 14, key_color: int = 11, door_color: int = 8) -> None:
        self.player_color = player_color
        self.goal_color = goal_color
        self.key_color = key_color
        self.door_color = door_color

    def plan_solution(self, frame: FrameData) -> List[Action]:
        """Synthesize plan from current frame."""
        comp = frame.grid.composite()

        player_locs = [tuple(p) for p in np.argwhere(comp == self.player_color)]
        goal_locs = [tuple(p) for p in np.argwhere(comp == self.goal_color)]
        key_locs = [tuple(p) for p in np.argwhere(comp == self.key_color)]
        door_locs = [tuple(p) for p in np.argwhere(comp == self.door_color)]

        if not player_locs or not goal_locs:
            return [Action(ActionType.ACTION1)]

        start = player_locs[0]
        goal = goal_locs[0]

        # Case 1: Key and door present -> navigate to key, interact, navigate to door, unlock, navigate to goal
        if key_locs and door_locs:
            key_pos = key_locs[0]
            door_pos = door_locs[0]

            path1 = AStarSolver.solve_grid_path(comp, start, key_pos, barrier_colors={5, 8})
            interact_action = Action(ActionType.ACTION5)
            path2 = AStarSolver.solve_grid_path(comp, key_pos, door_pos, barrier_colors={5})
            unlock_action = Action(ActionType.ACTION5)
            path3 = AStarSolver.solve_grid_path(comp, door_pos, goal, barrier_colors={5})

            full_plan = path1 + [interact_action] + path2 + [unlock_action] + path3
            return full_plan

        # Case 2: Simple maze navigation to goal
        plan = AStarSolver.solve_grid_path(comp, start, goal, barrier_colors={5, 8})
        if not plan:
            # Fallback direct step
            return [Action(ActionType.ACTION1)]
        return plan
