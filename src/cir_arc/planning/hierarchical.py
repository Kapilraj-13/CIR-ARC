"""4-Level Hierarchical Planner for ARC-AGI-3 Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action, ActionType
from cir_arc.goals.hypothesis import GoalHypothesis, GoalType
from cir_arc.goals.manager import GoalManager
from cir_arc.planning.action_cost import ActionCostModel
from cir_arc.planning.search import AStarGridPlanner
from cir_arc.probing.information_gain import InformationGainExplorer


class HierarchicalPlanner:
    """Hierarchically decomposes high-level goals into tactical navigation paths and primitive actions."""

    def __init__(
        self,
        goal_manager: Optional[GoalManager] = None,
        info_explorer: Optional[InformationGainExplorer] = None,
        cost_model: Optional[ActionCostModel] = None,
    ) -> None:
        self.goal_manager = goal_manager or GoalManager()
        self.info_explorer = info_explorer or InformationGainExplorer()
        self.cost_model = cost_model or ActionCostModel()
        self.current_plan: List[Action] = []
        self.current_target_coord: Optional[Tuple[int, int]] = None

    def plan(
        self,
        belief: BeliefState,
        comp_grid: np.ndarray,
        available_actions: List[int],
    ) -> List[Action]:
        """Synthesizes or advances the hierarchical action plan."""
        # Level 1: Update and select strategic goal / tactical subgoal
        active_goal = self.goal_manager.update_from_belief(belief, comp_grid)
        player_pos = belief.player_location

        if player_pos is None:
            return [Action(ActionType.ACTION1)]

        # Level 2 & 3: Tactical Target & Collision-Free Pathfinding
        target_coord = self.goal_manager.get_active_target_coordinate()

        if target_coord is not None:
            # If already at or adjacent to interactive subgoal (key or door)
            manhattan = abs(player_pos[0] - target_coord[0]) + abs(player_pos[1] - target_coord[1])
            is_interactive_subgoal = (
                active_goal is not None
                and active_goal.goal_type in (GoalType.COLLECT_OBJECT, GoalType.UNLOCK_BARRIER)
            )

            if manhattan <= 1 and is_interactive_subgoal and 5 in available_actions:
                # Perform interaction
                self.current_plan = [Action(ActionType.ACTION5)]
                self.current_target_coord = target_coord
                return self.current_plan

            if target_coord != player_pos:
                passable_mask = belief.get_passable_mask(comp_grid)
                path = AStarGridPlanner.find_path(passable_mask, player_pos, target_coord)

                if path and len(path) > 1:
                    # Level 4: Convert path to primitive actions
                    primitive_actions = AStarGridPlanner.path_to_actions(path)
                    # If navigating to an interactive object, append INTERACT at destination
                    if is_interactive_subgoal and 5 in available_actions:
                        primitive_actions.append(Action(ActionType.ACTION5))

                    self.current_plan = primitive_actions
                    self.current_target_coord = target_coord
                    return primitive_actions

        # If direct goal path is blocked or unknown -> Information-Gain Exploration
        exploratory_act = self.info_explorer.select_best_exploratory_action(
            belief=belief,
            comp_grid=comp_grid,
            available_actions=available_actions,
        )
        self.current_plan = [exploratory_act]
        return self.current_plan
