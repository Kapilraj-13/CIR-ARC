"""Unit tests for Hierarchical Planner and A* Grid Pathfinding."""

import numpy as np
import pytest

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, MultiLayerGrid
from cir_arc.goals.hypothesis import GoalHypothesis, GoalType
from cir_arc.goals.manager import GoalManager
from cir_arc.planning.hierarchical import HierarchicalPlanner
from cir_arc.planning.search import AStarGridPlanner


class TestHierarchicalPlanning:
    def test_astar_grid_planner_pathfinding(self):
        # 5x5 grid with wall barrier
        mask = np.ones((5, 5), dtype=bool)
        mask[1:4, 2] = False  # Vertical wall in column 2, rows 1..3

        start = (2, 0)
        goal = (2, 4)

        path = AStarGridPlanner.find_path(mask, start, goal)
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

        # Ensure wall is never crossed
        for r, c in path:
            assert mask[r, c] == True

        actions = AStarGridPlanner.path_to_actions(path)
        assert len(actions) == len(path) - 1

    def test_hierarchical_planner_execution(self):
        grid = np.zeros((4, 4), dtype=np.int16)
        grid[0, 0] = 9   # Player
        grid[3, 3] = 14  # Goal

        frame = FrameData(game_id="h_plan_test", grid=MultiLayerGrid([grid]))
        belief = BeliefState(game_id="h_plan_test", player_color=9)
        belief.update_from_frame(frame)

        planner = HierarchicalPlanner()
        actions = planner.plan(belief, grid, available_actions=[0, 1, 2, 3])

        assert len(actions) > 0
        assert all(isinstance(a, Action) for a in actions)
