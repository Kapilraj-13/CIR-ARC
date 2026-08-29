"""Unit tests for Goal Inference Engine and Hierarchical Subgoals."""

import numpy as np
import pytest

from cir_arc.belief.state import BeliefState
from cir_arc.environment.frame import FrameData, MultiLayerGrid
from cir_arc.goals.detector import GoalDetector
from cir_arc.goals.hypothesis import GoalHypothesis, GoalType
from cir_arc.goals.manager import GoalManager
from cir_arc.goals.scorer import GoalScorer


class TestGoalInferenceEngine:
    def test_goal_hypothesis_progress_and_scoring(self):
        g = GoalHypothesis(
            goal_id="g1",
            goal_type=GoalType.REACH_LOCATION,
            target_coordinate=(5, 5),
            evidence=0.9,
        )
        assert g.progress == 0.0
        assert not g.is_satisfied

        g.update_progress(current_player_pos=(4, 5), grid_shape=(10, 10))
        assert g.progress > 0.8
        assert g.score > 0.5

        g.update_progress(current_player_pos=(5, 5), grid_shape=(10, 10))
        assert g.is_satisfied
        assert g.progress == 1.0

    def test_goal_detector_and_subgoal_decomposition(self):
        # Locksmith layout: player at (1, 1), key (11) at (1, 3), door (8) at (1, 5), goal (14) at (1, 7)
        grid_arr = np.zeros((3, 9), dtype=np.int16)
        grid_arr[1, 1] = 9   # Player
        grid_arr[1, 3] = 11  # Key
        grid_arr[1, 5] = 8   # Door
        grid_arr[1, 7] = 14  # Goal

        frame = FrameData(game_id="locksmith_test", grid=MultiLayerGrid([grid_arr]))
        belief = BeliefState(game_id="locksmith_test", player_color=9)
        belief.update_from_frame(frame)

        detector = GoalDetector()
        candidates = detector.detect_candidate_goals(belief, grid_arr)
        assert len(candidates) >= 3

        types = [g.goal_type for g in candidates]
        assert GoalType.COLLECT_OBJECT in types
        assert GoalType.UNLOCK_BARRIER in types
        assert GoalType.REACH_LOCATION in types

        # Test Manager Subgoal Decomposition
        mgr = GoalManager(detector=detector)
        active_goal = mgr.update_from_belief(belief, grid_arr)

        assert active_goal is not None
        assert active_goal.goal_type == GoalType.COLLECT_OBJECT  # Key must be collected first!
        assert active_goal.target_coordinate == (1, 3)
