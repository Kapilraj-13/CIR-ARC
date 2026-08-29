"""Unit tests for Failure Recovery, Dynamic Replanning, and Rollback."""

import numpy as np
import pytest

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, MultiLayerGrid
from cir_arc.hypothesis.counterexample import Counterexample
from cir_arc.hypothesis.induction_engine import HypothesisInductionEngine
from cir_arc.hypothesis.repair import HypothesisRepair
from cir_arc.recovery.contradiction import PlanContradictionDetector
from cir_arc.recovery.replanner import DynamicReplanner
from cir_arc.recovery.rollback import StateRollback


class TestRecoveryAndRepair:
    def test_plan_contradiction_detection(self):
        grid1 = np.array([[9, 5], [0, 0]], dtype=np.int16)
        f1 = FrameData(game_id="c_test", grid=MultiLayerGrid([grid1]), step_count=1)
        f2_blocked = FrameData(game_id="c_test", grid=MultiLayerGrid([grid1]), step_count=2)

        # Move Right (ACTION4 = 3) into wall (5) -> state hash unchanged
        act = Action(ActionType.ACTION4)
        is_contra = PlanContradictionDetector.check_movement_contradiction(f1, act, f2_blocked)
        assert is_contra is True

    def test_hypothesis_repair_from_counterexample(self):
        belief = BeliefState(game_id="r_test", player_color=9)
        engine = HypothesisInductionEngine()
        repair = HypothesisRepair()

        ce = Counterexample(
            counterexample_id="ce_1",
            action=Action(ActionType.ACTION4),
            predicted_state_hash="pred",
            actual_state_hash="actual",
            mismatched_cells=[(1, 2, 9, 8)],  # Expected player (9), found locked door (8)
            failed_assumption="Expected door to open",
        )

        record = repair.repair_from_counterexample(ce, belief, engine)
        assert len(record["repairs_applied"]) > 0
        # Color 8 should now be recognized as impassable
        assert 8 in belief.facts.get_known_blocked_colors()

    def test_state_rollback_cycle_detection(self):
        rollback = StateRollback(history_window=4)
        rollback.record_position((1, 1))
        rollback.record_position((1, 2))
        rollback.record_position((1, 1))
        rollback.record_position((1, 2))

        assert rollback.is_in_cycle() is True
        breakout = rollback.get_breakout_action([0, 1, 2, 3])
        assert isinstance(breakout, Action)
