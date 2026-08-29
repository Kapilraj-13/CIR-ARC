"""Unit tests for Episodic Memory, Environment Schemas, and Novelty Detection."""

import numpy as np
import pytest

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, GameState, MultiLayerGrid
from cir_arc.memory.episodic import Episode, EpisodicMemory
from cir_arc.memory.schema import EnvironmentSchema, NoveltyDetector


class TestEpisodicMemoryAndSchema:
    def test_episodic_memory_storage_and_retrieval(self):
        mem = EpisodicMemory()
        grid = np.array([[9, 14]], dtype=np.int16)
        f0 = FrameData(game_id="maze_01", grid=MultiLayerGrid([grid]))

        ep = Episode(
            episode_id="ep_01",
            game_id="maze_01",
            initial_frame=f0,
            actions=[Action(ActionType.ACTION4)],
            outcome=GameState.WIN,
            total_steps=1,
        )
        mem.store_episode(ep)

        assert mem.get_episode("ep_01") is not None
        similar = mem.find_similar_episodes("maze_01")
        assert len(similar) == 1

    def test_environment_schema_novelty_detection(self):
        schema = EnvironmentSchema(
            schema_id="locksmith_schema",
            game_type="locksmith_puzzle",
            required_colors={9, 11, 8, 14},  # Player, Key, Door, Goal
        )

        belief = BeliefState(game_id="locksmith_test", player_color=9)

        # Matching grid
        grid_match = np.array([[9, 11, 8, 14]], dtype=np.int16)
        is_match, score = NoveltyDetector.match_schema(schema, belief, grid_match)
        assert is_match is True
        assert score == 1.0

        # Non-matching grid (missing door and goal)
        grid_diff = np.array([[9, 0, 0, 0]], dtype=np.int16)
        is_match_diff, score_diff = NoveltyDetector.match_schema(schema, belief, grid_diff)
        assert is_match_diff is False
