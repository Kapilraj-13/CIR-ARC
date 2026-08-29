"""Unit tests for Epistemic Belief State and Provenance Tracking."""

import numpy as np
import pytest

from cir_arc.belief.facts import Fact, FactSet, FactType, Provenance
from cir_arc.belief.state import BeliefState
from cir_arc.belief.uncertainty import UncertaintyModel
from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, MultiLayerGrid


class TestBeliefStateAndFacts:
    def test_fact_creation_and_provenance(self):
        f = Fact(
            fact_id="PASS:color_5",
            fact_type=FactType.PASSABILITY,
            subject="color_5",
            predicate="is_passable",
            value=False,
            provenance=Provenance.OBSERVED,
            confidence=0.9,
        )
        assert f.is_verified
        assert f.provenance == Provenance.OBSERVED

        # Contradict fact
        f.contradict(penalty=0.7)
        assert not f.is_verified
        assert f.provenance == Provenance.HYPOTHESIS

    def test_fact_set_operations(self):
        fs = FactSet()
        fs.add_or_update(
            FactType.PASSABILITY,
            subject="color_0",
            predicate="is_passable",
            value=True,
            provenance=Provenance.FACT,
        )
        fs.add_or_update(
            FactType.PASSABILITY,
            subject="color_5",
            predicate="is_passable",
            value=False,
            provenance=Provenance.OBSERVED,
        )

        assert 0 in fs.get_known_passable_colors()
        assert 5 in fs.get_known_blocked_colors()

    def test_uncertainty_model_tracking(self):
        unc = UncertaintyModel(height=10, width=10)
        assert unc.get_color_uncertainty(5) == 1.0
        assert unc.get_color_uncertainty(0) == 0.0

        unc.record_visit(2, 3)
        unc.record_visit(2, 3)
        s_map = unc.get_spatial_uncertainty_map()
        assert s_map[2, 3] < s_map[0, 0]

        total_ent = unc.compute_total_epistemic_entropy()
        assert 0.0 <= total_ent <= 1.0

    def test_belief_state_update_from_frame(self):
        grid_arr = np.array([
            [5, 5, 5],
            [5, 9, 14],
            [5, 5, 5],
        ], dtype=np.int16)
        frame = FrameData(
            game_id="maze_test",
            grid=MultiLayerGrid([grid_arr]),
            step_count=1,
        )

        belief = BeliefState(game_id="maze_test", player_color=9)
        belief.update_from_frame(frame)

        assert belief.player_location == (1, 1)
        assert len(belief.observed_objects) >= 2
        assert "obj_color_14" in belief.observed_objects
        assert belief.observed_objects["obj_color_14"].role == "goal_candidate"

        mask = belief.get_passable_mask(grid_arr)
        assert mask[1, 1] == True  # Player
        assert mask[1, 2] == True  # Goal
        assert mask[0, 0] == False # Wall
