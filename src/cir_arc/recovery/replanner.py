"""Dynamic Replanning engine handling plan invalidation and recovery."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action
from cir_arc.goals.manager import GoalManager
from cir_arc.planning.hierarchical import HierarchicalPlanner

logger = logging.getLogger(__name__)


class DynamicReplanner:
    """Handles on-the-fly plan invalidation and triggers immediate recovery replanning."""

    def __init__(
        self,
        planner: Optional[HierarchicalPlanner] = None,
        goal_manager: Optional[GoalManager] = None,
    ) -> None:
        self.planner = planner or HierarchicalPlanner(goal_manager=goal_manager)
        self.goal_manager = goal_manager or self.planner.goal_manager
        self.replan_count: int = 0

    def trigger_replan(
        self,
        belief: BeliefState,
        comp_grid: np.ndarray,
        available_actions: List[int],
        reason: str = "",
    ) -> List[Action]:
        """Invalidates current cached plan and generates fresh hierarchical plan."""
        self.replan_count += 1
        logger.info(f"Triggering dynamic replan (count={self.replan_count}) due to: {reason}")

        # Invalidate existing plan in hierarchical planner
        self.planner.current_plan.clear()

        # Replan fresh
        new_plan = self.planner.plan(belief, comp_grid, available_actions)
        return new_plan
