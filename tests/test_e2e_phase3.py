import os
import tempfile
import numpy as np
import pytest
import torch

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, GameState, MultiLayerGrid
from cir_arc.environment.mock_engine import MockEngine
from cir_arc.environment.state_delta import compute_state_delta, find_objects
from cir_arc.neural.temporal.tracker import TemporalSlotTracker
from cir_arc.probing.inspector_agent import EnvironmentInspector
from cir_arc.hypothesis.induction_engine import HypothesisInductionEngine
from cir_arc.dsl.world_model import DSLWorldModel
from cir_arc.solving.runtime import SolvingRuntime
from cir_arc.recording.playback import PlaybackAgent


class TestE2EPhase3Acceptance:
    """Full End-to-End verification across all Phase 3 acceptance criteria."""

    def test_ac1_environment_and_8_actions(self):
        """AC1: Environment adapter handles all 8 action types and returns structured FrameData."""
        env = MockEngine("mock_locksmith_01")
        f0 = env.reset()
        assert f0.grid.num_layers == 2
        assert f0.grid.height == 10
        assert f0.grid.width == 10

        # Step through actions 0..7
        for aid in range(8):
            if aid == 6:
                act = Action.click(2, 2)
            else:
                act = Action(ActionType(aid))
            frame = env.step(act)
            assert isinstance(frame, FrameData)
            if aid > 0:
                assert frame.step_count > 0

    def test_ac2_state_delta_zero_false_positives(self):
        """AC2: Frame hashing and delta computation has zero false-positive mutations on identical frames."""
        env = MockEngine("mock_maze_01")
        f1 = env.reset()
        f2 = env.reset()
        assert f1.hash() == f2.hash()

        delta = compute_state_delta(f1, f2, Action(ActionType.RESET))
        assert not delta.has_mutation
        assert delta.pixel_diff_count == 0

    def test_ac3_temporal_slot_persistence_above_90_percent(self):
        """AC3: Temporal slot tracking maintains identity across frames with >= 90% persistence."""
        tracker = TemporalSlotTracker(objectness_threshold=0.3, slot_dim=64)
        K = 4
        slots_0 = torch.randn(K, 64)
        obj_0 = torch.tensor([0.95, 0.95, 0.1, 0.1])
        pos_0 = torch.tensor([[5.0 / 30.0, 5.0 / 30.0], [15.0 / 30.0, 15.0 / 30.0], [0, 0], [0, 0]])
        color_0 = torch.zeros(K, 16)
        color_0[0, 9] = 1.0
        color_0[1, 11] = 1.0

        t0 = tracker.update_from_perception(slots_0, obj_0, pos_preds=pos_0, color_preds=color_0, H=30, W=30)
        assert len(t0) == 2
        t_ids = {t.track_id for t in t0}

        matches = 0
        total_steps = 15
        for s in range(1, total_steps + 1):
            slots_s = slots_0 + 0.02 * torch.randn(K, 64)
            pos_s = pos_0.clone()
            pos_s[0, 0] += 0.01 * s
            t_s = tracker.update_from_perception(slots_s, obj_0, pos_preds=pos_s, color_preds=color_0, H=30, W=30)
            active_ids = {t.track_id for t in t_s if t.lifecycle_state == "ACTIVE"}
            if active_ids == t_ids:
                matches += 1

        persistence = matches / total_steps
        assert persistence >= 0.90, f"Slot persistence rate {persistence:.2f} < 0.90"

    def test_ac4_active_probing_environment_profile(self):
        """AC4: Inspector agent produces a structured EnvironmentProfile with actions, objects, resources, phases."""
        env = MockEngine("mock_maze_01")
        inspector = EnvironmentInspector(env, max_probes=15)
        profile = inspector.inspect()

        assert len(profile.action_matrix.get_active_actions()) > 0
        assert len(profile.object_catalog.catalog) > 0
        assert len(profile.resources) > 0
        assert len(profile.phases) > 0

    def test_ac5_hypothesis_rule_induction(self):
        """AC5: Hypothesis engine infers action effect rules from probe traces."""
        engine = HypothesisInductionEngine()
        env = MockEngine("mock_maze_01")
        f0 = env.reset()
        act_down = Action(ActionType.ACTION2)
        f1 = env.step(act_down)

        rule = engine.observe_transition(f0, act_down, f1)
        assert rule is not None
        assert rule.action_trigger == 2
        assert rule.effect_type == "TRANSLATE_PLAYER"

    def test_ac6_solving_and_recording_roundtrip(self):
        """AC6: Solving runtime completes games without crashes and produces valid recording replayable via Playback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_file = os.path.join(tmpdir, "e2e_session.recording.jsonl")
            runtime = SolvingRuntime(max_actions=50, record=True)
            env = MockEngine("mock_maze_01")
            report = runtime.run_game(env)

            assert report.is_win
            assert report.state == GameState.WIN
            assert report.actions_taken > 0
            assert report.recording_path is not None

            # Replay via PlaybackAgent
            playback = PlaybackAgent(report.recording_path)
            assert len(playback.actions) > 0
            replay_env = MockEngine("mock_maze_01")
            replayed_frames = playback.replay_all(replay_env)
            assert len(replayed_frames) > 1
            assert replayed_frames[-1].state == GameState.WIN
