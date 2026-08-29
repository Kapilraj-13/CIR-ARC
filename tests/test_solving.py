import pytest
import numpy as np

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import GameState
from cir_arc.environment.mock_engine import MockEngine
from cir_arc.solving.search_solvers import AStarSolver, BFSSolver
from cir_arc.solving.code_agent import CodeAgentSolver
from cir_arc.solving.cognitive_loop import CognitiveLoop
from cir_arc.solving.runtime import ScorecardReport, SolvingRuntime


class TestSearchSolvers:
    def test_astar_solver_finds_path(self):
        grid = np.zeros((6, 6), dtype=np.int16)
        grid[2, 0:5] = 5  # Barrier wall with opening at (2, 5)

        start = (1, 1)
        goal = (4, 1)

        actions = AStarSolver.solve_grid_path(grid, start, goal, barrier_colors={5})
        assert len(actions) > 0

        # Execute actions on mock environment or forward model
        curr = list(start)
        for act in actions:
            from cir_arc.environment.actions import DIRECTION_VECTORS
            dr, dc = DIRECTION_VECTORS[act.action_id]
            curr[0] += dr
            curr[1] += dc

        assert tuple(curr) == goal

    def test_bfs_solver_finds_path(self):
        grid = np.zeros((5, 5), dtype=np.int16)
        start = (0, 0)
        goal = (3, 3)

        actions = BFSSolver.solve(grid, start, goal)
        assert len(actions) == 6  # 3 down + 3 right = 6 steps


class TestCognitiveLoopAndRuntime:
    def test_solving_runtime_wins_maze_game(self):
        runtime = SolvingRuntime(max_actions=50, record=True)
        env = MockEngine("mock_maze_01")
        report = runtime.run_game(env)

        assert isinstance(report, ScorecardReport)
        assert report.is_win
        assert report.state == GameState.WIN
        assert report.levels_completed >= 1
        assert report.actions_taken > 0
        assert report.recording_path is not None
        assert "SESSION_END" in [e["name"] for e in report.telemetry_summary["events"]]

    def test_solving_runtime_locksmith_puzzle(self):
        runtime = SolvingRuntime(max_actions=60, record=False)
        env = MockEngine("mock_locksmith_01")
        report = runtime.run_game(env)

        assert isinstance(report, ScorecardReport)
        assert report.is_win
        assert report.state == GameState.WIN
        assert report.actions_taken <= 60
