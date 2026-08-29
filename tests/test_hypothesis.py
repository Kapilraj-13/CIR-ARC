import pytest
import numpy as np

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, MultiLayerGrid
from cir_arc.environment.mock_engine import MockEngine
from cir_arc.hypothesis.delta_mapper import DeltaMapper
from cir_arc.hypothesis.induction_engine import HypothesisInductionEngine
from cir_arc.hypothesis.transition_grammar import TransitionRule
from cir_arc.dsl.interactive_primitives import step_translate, step_recolor, macro_navigate_path
from cir_arc.dsl.world_model import DSLWorldModel


class TestDeltaMapperAndRules:
    def test_transition_rule_application(self):
        grid = np.zeros((4, 4), dtype=np.int16)
        grid[1, 1] = 9

        rule = TransitionRule(
            rule_id="R_MOVE_DOWN",
            action_trigger=2,
            condition_type="ALWAYS",
            effect_type="TRANSLATE_PLAYER",
            params={"dr": 1, "dc": 0},
        )

        assert rule.matches(Action(ActionType.ACTION2))
        assert not rule.matches(Action(ActionType.ACTION1))

        sim = rule.apply_to_grid(grid, player_color=9)
        assert sim[1, 1] == 0
        assert sim[2, 1] == 9


class TestHypothesisInductionEngine:
    def test_rule_induction_from_transitions(self):
        engine = HypothesisInductionEngine()
        env = MockEngine("mock_maze_01")
        f0 = env.reset()

        a2 = Action(ActionType.ACTION2)  # Down
        f1 = env.step(a2)

        rule = engine.observe_transition(f0, a2, f1)
        assert rule is not None
        assert rule.action_trigger == 2
        assert rule.effect_type == "TRANSLATE_PLAYER"
        assert rule.params["dr"] == 1
        assert rule.params["dc"] == 0
        assert rule.confidence == 1.0


class TestDSLWorldModel:
    def test_forward_simulation(self):
        grid = np.zeros((6, 6), dtype=np.int16)
        grid[1, 1] = 9
        grid[1, 3] = 5  # Barrier wall

        wm = DSLWorldModel()
        act_right = Action(ActionType.ACTION4)  # (0, +1)
        
        # Step 1: (1, 1) -> (1, 2)
        s1 = wm.predict_step(grid, act_right, player_color=9)
        assert s1[1, 1] == 0
        assert s1[1, 2] == 9

        # Step 2: (1, 2) -> (1, 3) hits barrier, remains at (1, 2)
        s2 = wm.predict_step(s1, act_right, player_color=9)
        assert s2[1, 2] == 9
        assert s2[1, 3] == 5

    def test_prediction_loss(self):
        g1 = np.zeros((4, 4), dtype=np.int16)
        g2 = g1.copy()
        assert DSLWorldModel.evaluate_prediction_loss(g1, g2) == 0.0

        g2[0, 0] = 9
        loss = DSLWorldModel.evaluate_prediction_loss(g1, g2)
        assert loss == 1.0 / 16.0
