import os
import tempfile
import numpy as np
import pytest

from cir_arc.environment.actions import Action, ActionSpec, ActionType, ACTION_NAMES, DIRECTION_VECTORS
from cir_arc.environment.frame import FrameData, GameState, MultiLayerGrid
from cir_arc.environment.state_delta import GridObject, PropertyMutation, StateDelta, compute_state_delta, find_objects
from cir_arc.environment.mock_engine import MockEngine
from cir_arc.environment.rc_adapter import RCEngineAdapter
from cir_arc.recording.recorder import SessionRecorder
from cir_arc.recording.playback import PlaybackAgent


class TestActions:
    def test_action_enum_values(self):
        assert ActionType.RESET == 0
        assert ActionType.ACTION1 == 1
        assert ActionType.ACTION2 == 2
        assert ActionType.ACTION3 == 3
        assert ActionType.ACTION4 == 4
        assert ActionType.ACTION5 == 5
        assert ActionType.ACTION6 == 6
        assert ActionType.ACTION7 == 7

    def test_action_creation(self):
        a = Action.from_id(1)
        assert a.action_type == ActionType.ACTION1
        assert a.action_id == 1
        assert a.name == "ACTION1"
        assert not a.is_complex

        click_a = Action.click(10, 20)
        assert click_a.is_complex
        assert click_a.data == {"x": 10, "y": 20}
        assert click_a.action_type == ActionType.ACTION6

    def test_action_to_dict(self):
        a = Action.from_id(5, data={"param": 1}, reasoning={"thought": "test"})
        d = a.to_dict()
        assert d["id"] == 5
        assert d["name"] == "ACTION5"
        assert d["data"] == {"param": 1}
        assert d["reasoning"] == {"thought": "test"}


class TestMultiLayerGrid:
    def test_create_and_composite(self):
        l0 = np.zeros((4, 4), dtype=np.int16)
        l0[0, 0] = 5
        l1 = np.zeros((4, 4), dtype=np.int16)
        l1[1, 1] = 9

        mlg = MultiLayerGrid([l0, l1])
        assert mlg.num_layers == 2
        assert mlg.height == 4
        assert mlg.width == 4
        assert mlg.shape == (2, 4, 4)

        comp = mlg.composite()
        assert comp[0, 0] == 5
        assert comp[1, 1] == 9
        assert comp[2, 2] == 0

    def test_hashing_stability(self):
        l0 = np.array([[1, 2], [3, 4]], dtype=np.int16)
        l1 = np.array([[0, 0], [5, 6]], dtype=np.int16)
        g1 = MultiLayerGrid([l0, l1])
        g2 = MultiLayerGrid([l0.copy(), l1.copy()])
        assert g1.hash() == g2.hash()
        assert len(g1.hash()) == 16

    def test_from_list_to_list(self):
        raw = [[[1, 2], [3, 4]], [[0, 5], [6, 0]]]
        mlg = MultiLayerGrid.from_list(raw)
        assert mlg.num_layers == 2
        assert mlg.height == 2
        assert mlg.width == 2
        assert mlg.to_list() == raw


class TestStateDeltaAndObjects:
    def test_find_objects_single_layer(self):
        grid = np.zeros((6, 6), dtype=np.int16)
        grid[1:3, 1:3] = 4  # 2x2 square of color 4
        grid[4, 4] = 9      # 1x1 dot of color 9

        objs = find_objects(grid)
        assert len(objs) == 2
        assert objs[0].color == 4
        assert objs[0].pixel_count == 4
        assert objs[0].bbox == (1, 1, 2, 2)
        assert objs[1].color == 9
        assert objs[1].pixel_count == 1

    def test_find_objects_multilayer(self):
        l0 = np.zeros((4, 4), dtype=np.int16)
        l0[0, :] = 5  # Wall of 4 pixels
        l1 = np.zeros((4, 4), dtype=np.int16)
        l1[2, 2] = 9  # Agent of 1 pixel

        mlg = MultiLayerGrid([l0, l1])
        objs = find_objects(mlg)
        assert len(objs) == 2
        assert objs[0].layer == 0
        assert objs[0].color == 5
        assert objs[1].layer == 1
        assert objs[1].color == 9

    def test_compute_state_delta_zero_mutation_on_identical(self):
        grid = MultiLayerGrid.from_list([[[1, 2], [3, 4]]])
        f1 = FrameData(game_id="g1", grid=grid, state=GameState.NOT_FINISHED)
        f2 = FrameData(game_id="g1", grid=grid, state=GameState.NOT_FINISHED)

        delta = compute_state_delta(f1, f2, Action(ActionType.ACTION1))
        assert not delta.has_mutation
        assert delta.pixel_diff_count == 0
        assert delta.is_identity

    def test_compute_state_delta_movement(self):
        g1 = np.zeros((5, 5), dtype=np.int16)
        g1[1, 1] = 9
        g2 = np.zeros((5, 5), dtype=np.int16)
        g2[2, 1] = 9  # Moved down by 1 row

        f1 = FrameData(game_id="g1", grid=MultiLayerGrid([g1]), state=GameState.NOT_FINISHED)
        f2 = FrameData(game_id="g1", grid=MultiLayerGrid([g2]), state=GameState.NOT_FINISHED)

        delta = compute_state_delta(f1, f2, Action(ActionType.ACTION2))
        assert delta.has_mutation
        assert delta.pixel_diff_count == 2
        assert len(delta.moved_objects) == 1
        dr, dc = delta.moved_objects[0][2]
        assert (dr, dc) == (1.0, 0.0)


class TestMockEngine:
    def test_mock_engine_maze_solve(self):
        env = MockEngine("mock_maze_01")
        initial_frame = env.reset()
        assert initial_frame.state == GameState.NOT_FINISHED
        assert env.player_pos == [1, 1]

        # Step towards goal
        env.step(Action(ActionType.ACTION2))  # Down to (2, 1)
        assert env.player_pos == [2, 1]

        env.step(Action(ActionType.ACTION4))  # Right blocked by wall (2, 2)
        assert env.player_pos == [2, 1]

    def test_mock_engine_locksmith_key_door(self):
        env = MockEngine("mock_locksmith_01")
        env.reset()
        assert not env.has_key
        assert not env.door_open

        # Teleport near key and pick it up
        env.player_pos = [1, 7]
        env.step(Action(ActionType.ACTION5))
        assert env.has_key

        # Move near door and unlock
        env.player_pos = [4, 5]
        env.step(Action(ActionType.ACTION5))
        assert env.door_open


class TestRecordingAndPlayback:
    def test_recorder_and_playback_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, "test.recording.jsonl")
            recorder = SessionRecorder(game_id="mock_maze_01", output_path=rec_path)

            env = MockEngine("mock_maze_01")
            f0 = env.reset()
            recorder.record(f0)

            a1 = Action(ActionType.ACTION2)
            f1 = env.step(a1)
            recorder.record(a1)
            recorder.record(f1)

            saved_path = recorder.flush()
            assert os.path.exists(saved_path)

            playback = PlaybackAgent(saved_path)
            assert len(playback.actions) == 1
            assert playback.actions[0].action_type == ActionType.ACTION2

            replayed_frames = playback.replay_all(MockEngine("mock_maze_01"))
            assert len(replayed_frames) == 2
            assert replayed_frames[1].step_count == 1
