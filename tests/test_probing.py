import pytest
import numpy as np

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.mock_engine import MockEngine
from cir_arc.environment.state_delta import compute_state_delta
from cir_arc.probing.action_matrix import ActionEffectMatrix
from cir_arc.probing.object_catalog import DynamicObjectCatalog
from cir_arc.probing.resource_inspector import ResourceInspector
from cir_arc.probing.state_machine import GameStateMachine
from cir_arc.probing.inspector_agent import EnvironmentInspector, EnvironmentProfile


class TestActionEffectMatrix:
    def test_record_and_query_active_actions(self):
        aem = ActionEffectMatrix()
        env = MockEngine("mock_maze_01")
        f0 = env.reset()

        a1 = Action(ActionType.ACTION1)  # Up (hits boundary wall, no move)
        f1 = env.step(a1)
        d1 = compute_state_delta(f0, f1, a1)
        aem.record_probe(a1, d1)

        a2 = Action(ActionType.ACTION2)  # Down (moves agent)
        f2 = env.step(a2)
        d2 = compute_state_delta(f1, f2, a2)
        aem.record_probe(a2, d2)

        active = aem.get_active_actions()
        assert 2 in active
        assert aem.effects[2].causes_frame_change
        assert aem.effects[2].movement_vector is not None


class TestDynamicObjectCatalog:
    def test_catalog_player_detection(self):
        catalog = DynamicObjectCatalog()
        env = MockEngine("mock_maze_01")
        f0 = env.reset()

        a2 = Action(ActionType.ACTION2)
        f1 = env.step(a2)
        delta = compute_state_delta(f0, f1, a2)

        catalog.update_from_delta(delta)
        player = catalog.get_player()
        assert player is not None
        assert player.color == 9
        assert player.is_controllable


class TestResourceAndStateMachine:
    def test_resource_inspector(self):
        env = MockEngine("mock_maze_01")
        f0 = env.reset()
        resources = ResourceInspector.inspect([f0])
        names = {r.name for r in resources}
        assert "levels_completed" in names
        assert "grid_height" in names
        assert "grid_width" in names

    def test_state_machine_phases(self):
        gsm = GameStateMachine()
        env = MockEngine("mock_maze_01")
        f0 = env.reset()
        gsm.record_observation(0, f0)
        assert len(gsm.phases) == 1
        assert gsm.phases[0].phase_name == "active"


class TestEnvironmentInspector:
    def test_inspector_exploration_profile(self):
        env = MockEngine("mock_locksmith_01")
        inspector = EnvironmentInspector(env, max_probes=12)
        profile = inspector.inspect()

        assert isinstance(profile, EnvironmentProfile)
        assert profile.game_id == "mock_locksmith_01"
        assert profile.probe_count > 0
        assert len(profile.resources) > 0
        assert len(profile.phases) > 0
        assert profile.grid_shape == (2, 10, 10)

        # Profile serialization check
        p_json = profile.to_json()
        assert "mock_locksmith_01" in p_json
