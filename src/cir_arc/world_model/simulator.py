"""Multi-step action sequence lookahead simulator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action
from cir_arc.world_model.executable import ExecutableWorldModel


class ActionSimulator:
    """Simulates multi-step hypothetical action rollouts."""

    def __init__(self, world_model: Optional[ExecutableWorldModel] = None) -> None:
        self.world_model = world_model or ExecutableWorldModel()

    def simulate_plan(
        self,
        initial_grid: np.ndarray,
        plan: List[Action],
        belief: Optional[BeliefState] = None,
    ) -> Dict[str, Any]:
        """Simulates sequence of actions and returns resulting state and rollout summary."""
        current_grid = initial_grid.copy()
        trajectory = [current_grid.copy()]
        events = []

        reached_goal = False
        total_movement = 0

        for idx, action in enumerate(plan):
            current_grid, meta = self.world_model.simulate_step(current_grid, action, belief=belief)
            trajectory.append(current_grid.copy())
            events.append(meta)

            if meta.get("player_moved", False):
                total_movement += 1
            if meta.get("reached_goal", False):
                reached_goal = True
                break

        return {
            "final_grid": current_grid,
            "trajectory": trajectory,
            "events": events,
            "reached_goal": reached_goal,
            "total_movement": total_movement,
            "plan_length": len(plan),
            "success_rate": 1.0 if reached_goal else (float(total_movement) / max(1, len(plan))),
        }
