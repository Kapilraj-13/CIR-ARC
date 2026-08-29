"""Unit tests for Executable World Model digital twin and replay verification."""

import numpy as np
import pytest

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, MultiLayerGrid
from cir_arc.hypothesis.counterexample import CounterexampleDetector
from cir_arc.world_model.executable import ExecutableWorldModel
from cir_arc.world_model.replay import ReplayVerifier
from cir_arc.world_model.simulator import ActionSimulator
from cir_arc.world_model.validator import WorldModelValidator


class TestWorldModelAndReplay:
    def test_executable_world_model_simulation(self):
        grid = np.array([
            [5, 5, 5, 5],
            [5, 9, 0, 14],
            [5, 5, 5, 5],
        ], dtype=np.int16)

        wm = ExecutableWorldModel(player_color=9)
        # Move Right (ACTION4)
        next_grid, meta = wm.simulate_step(grid, Action(ActionType.ACTION4))

        assert meta["player_moved"] is True
        assert next_grid[1, 1] == 0
        assert next_grid[1, 2] == 9

        # Move Right again onto goal (14)
        goal_grid, meta_goal = wm.simulate_step(next_grid, Action(ActionType.ACTION4))
        assert meta_goal["reached_goal"] is True
        assert goal_grid[1, 3] == 9

    def test_action_simulator_multi_step_rollout(self):
        grid = np.array([
            [5, 5, 5, 5, 5],
            [5, 9, 0, 0, 14],
            [5, 5, 5, 5, 5],
        ], dtype=np.int16)

        sim = ActionSimulator()
        plan = [
            Action(ActionType.ACTION4),  # Right
            Action(ActionType.ACTION4),  # Right
            Action(ActionType.ACTION4),  # Right into goal
        ]

        result = sim.simulate_plan(grid, plan)
        assert result["reached_goal"] is True
        assert result["total_movement"] == 3

    def test_replay_verifier_detects_consistency_and_counterexample(self):
        grid1 = np.array([[9, 0], [0, 0]], dtype=np.int16)
        grid2_consistent = np.array([[0, 9], [0, 0]], dtype=np.int16)
        grid2_contradict = np.array([[9, 0], [0, 0]], dtype=np.int16)  # Movement failed!

        f1 = FrameData(game_id="g1", grid=MultiLayerGrid([grid1]), step_count=1)
        f2_good = FrameData(game_id="g1", grid=MultiLayerGrid([grid2_consistent]), step_count=2)
        f2_bad = FrameData(game_id="g1", grid=MultiLayerGrid([grid2_contradict]), step_count=2)

        verifier = ReplayVerifier()
        is_good, ce_good = verifier.verify_transition(f1, Action(ActionType.ACTION4), f2_good)
        assert is_good is True
        assert ce_good is None

        is_bad, ce_bad = verifier.verify_transition(f1, Action(ActionType.ACTION4), f2_bad)
        assert is_bad is False
        assert ce_bad is not None
        assert len(ce_bad.mismatched_cells) > 0
