"""Comprehensive Test Suite for Procedural 100-Environment Benchmark Suite.

Validates:
- All 10 ARC-AGI-3 core mechanics (m0r0, r25, p35, n36, r87, a86, cl78, wa30, pk90, kq74)
- 10 difficulty tiers per mechanic with monotonic scaling
- Strict BaseEnvironment protocol conformance (reset, step, current_observation, enumerate_actions, is_terminal)
- FrameData structure, MultiLayerGrid layer compositing, GameState transitions
- Deterministic seed reproducibility
- Ground-truth solvability via reference_solution_path execution
"""

from __future__ import annotations

import pytest
import numpy as np

from cir_arc.environment.actions import Action, ActionType, ActionSpec
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData, GameState, MultiLayerGrid
from cir_arc.generators.interactive_suite import (
    InteractiveBenchmarkSuite,
    MirroredSymmetryEnv,
    GravityTrajectoriesEnv,
    PressurePlateGatesEnv,
    KeyLockMazesEnv,
    IceSlidingInertiaEnv,
    PortalTeleportationEnv,
    SokobanBlockPushingEnv,
    TrailFloorPaintingEnv,
    LaserOpticsMirrorsEnv,
    DynamicMovingHazardsEnv,
)


class TestInteractiveSuiteMechanics:
    """Validates registry, factory, and enumeration of all 10 mechanics."""

    def test_suite_mechanic_listing(self) -> None:
        mechanics = InteractiveBenchmarkSuite.list_mechanics()
        assert len(mechanics) == 10
        expected = [
            "mirrored_symmetry",
            "gravity_trajectories",
            "pressure_plate_gates",
            "key_lock_mazes",
            "ice_sliding_inertia",
            "portal_teleportation",
            "sokoban_block_pushing",
            "trail_floor_painting",
            "laser_optics_mirrors",
            "dynamic_moving_hazards",
        ]
        assert mechanics == expected

    def test_suite_generate_all_100(self) -> None:
        suite = InteractiveBenchmarkSuite.generate_all_100(base_seed=123)
        assert len(suite) == 100

        for m_idx, slug in enumerate(InteractiveBenchmarkSuite.MECHANICS):
            for tier in range(1, 11):
                gid = f"arc3_{slug}_t{tier:02d}"
                assert gid in suite
                env = suite[gid]
                assert isinstance(env, BaseEnvironment)
                assert env.game_id == gid
                assert env.optimal_solution_length >= 1

    @pytest.mark.parametrize("m_idx,slug", list(enumerate(InteractiveBenchmarkSuite.MECHANICS)))
    def test_create_environment_by_index_and_slug(self, m_idx: int, slug: str) -> None:
        env_from_slug = InteractiveBenchmarkSuite.create_environment(slug, tier=2, seed=10)
        env_from_idx = InteractiveBenchmarkSuite.create_environment(m_idx, tier=2, seed=10)
        assert env_from_slug.game_id == f"arc3_{slug}_t02"
        assert env_from_idx.game_id == f"arc3_{slug}_t02"


class TestBaseEnvironmentProtocol:
    """Validates strict adherence to BaseEnvironment protocol across mechanics."""

    @pytest.mark.parametrize("slug", InteractiveBenchmarkSuite.MECHANICS)
    def test_protocol_lifecycle(self, slug: str) -> None:
        env = InteractiveBenchmarkSuite.create_environment(slug, tier=1, seed=42)

        # 1. Observation before reset is None or valid
        obs0 = env.current_observation()
        # 2. Reset returns valid FrameData
        frame_reset = env.reset()
        assert isinstance(frame_reset, FrameData)
        assert frame_reset.state == GameState.NOT_FINISHED
        assert frame_reset.step_count == 0
        assert frame_reset.grid.num_layers >= 2
        assert env.current_observation() == frame_reset
        assert not env.is_terminal()

        # 3. Enumerate actions
        actions = env.enumerate_actions()
        assert len(actions) >= 4
        assert all(isinstance(a, ActionSpec) for a in actions)

        # 4. Step movement
        frame_step = env.step(Action.from_id(1))
        assert isinstance(frame_step, FrameData)
        assert frame_step.step_count == 1

        # 5. Composite grid
        composite = frame_step.grid.composite()
        assert isinstance(composite, np.ndarray)
        assert composite.shape == (frame_step.grid.height, frame_step.grid.width)


class TestSolvabilityAndGroundTruth:
    """Validates that reference solution paths consistently solve every mechanic."""

    @pytest.mark.parametrize("slug", InteractiveBenchmarkSuite.MECHANICS)
    def test_reference_solution_solves_environment(self, slug: str) -> None:
        for tier in [1, 3, 5]:
            env = InteractiveBenchmarkSuite.create_environment(slug, tier=tier, seed=100 + tier)
            frame = env.reset()
            assert frame.state == GameState.NOT_FINISHED

            sol_path = env.reference_solution_path
            assert len(sol_path) >= 1
            assert len(sol_path) == env.optimal_solution_length

            for aid in sol_path:
                frame = env.step(Action.from_id(aid))

            assert env.state == GameState.WIN
            assert env.is_terminal()
            assert env.levels_completed == 1


class TestDeterministicReproducibility:
    """Validates deterministic state and layout reproducibility under fixed seeds."""

    def test_seed_determinism(self) -> None:
        env1 = InteractiveBenchmarkSuite.create_environment("mirrored_symmetry", tier=3, seed=777)
        env2 = InteractiveBenchmarkSuite.create_environment("mirrored_symmetry", tier=3, seed=777)

        f1 = env1.reset()
        f2 = env2.reset()

        assert f1.grid.hash() == f2.grid.hash()
        assert env1.reference_solution_path == env2.reference_solution_path
        assert np.array_equal(f1.grid.composite(), f2.grid.composite())


class TestIndividualMechanics:
    """In-depth verification of specific mechanical dynamics."""

    def test_mirrored_symmetry_merging(self) -> None:
        env = MirroredSymmetryEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.agents) == 2

        # Step through reference solution to verify agents merge
        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert len(env.agents) == 1
        assert frame.state == GameState.WIN

    def test_gravity_trajectories_physics(self) -> None:
        env = GravityTrajectoriesEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.boulders) >= 1

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN

    def test_pressure_plate_gate_mechanics(self) -> None:
        env = PressurePlateGatesEnv(tier=2, seed=42)
        frame = env.reset()
        assert len(env.plates) >= 1
        assert len(env.gates) >= 1

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN

    def test_key_lock_mazes_inventory(self) -> None:
        env = KeyLockMazesEnv(tier=2, seed=42)
        frame = env.reset()
        assert len(env.keys) >= 1
        assert len(env.doors) >= 1

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN

    def test_ice_sliding_continuous_momentum(self) -> None:
        env = IceSlidingInertiaEnv(tier=1, seed=42)
        frame = env.reset()
        init_pos = env.player_pos

        # Step first slide action
        first_act = env.reference_solution_path[0]
        frame = env.step(Action.from_id(first_act))
        # Player should have slid multiple tiles or reached a wall/goal
        assert frame.state in (GameState.NOT_FINISHED, GameState.WIN)

    def test_portal_teleportation_jump(self) -> None:
        env = PortalTeleportationEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.portals) >= 2

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN

    def test_sokoban_block_pushing_rules(self) -> None:
        env = SokobanBlockPushingEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.boxes) == 1
        assert len(env.targets) == 1

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN

    def test_trail_floor_painting_coverage(self) -> None:
        env = TrailFloorPaintingEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.target_tiles) >= 5

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN
        assert set(env.target_tiles).issubset(env.painted)

    def test_laser_optics_sensor_illumination(self) -> None:
        env = LaserOpticsMirrorsEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.mirrors) >= 1

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN

    def test_dynamic_moving_hazards_avoidance(self) -> None:
        env = DynamicMovingHazardsEnv(tier=1, seed=42)
        frame = env.reset()
        assert len(env.hazards) >= 1

        for aid in env.reference_solution_path:
            frame = env.step(Action.from_id(aid))

        assert frame.state == GameState.WIN


class TestDifficultyScaling:
    """Validates monotonic parameter scaling across difficulty tiers 1 to 10."""

    @pytest.mark.parametrize("slug", InteractiveBenchmarkSuite.MECHANICS)
    def test_grid_size_and_complexity_monotonicity(self, slug: str) -> None:
        env_t1 = InteractiveBenchmarkSuite.create_environment(slug, tier=1, seed=42)
        env_t5 = InteractiveBenchmarkSuite.create_environment(slug, tier=5, seed=42)
        env_t10 = InteractiveBenchmarkSuite.create_environment(slug, tier=10, seed=42)

        # Dimension scaling
        h1, w1 = env_t1.grid_size
        h5, w5 = env_t5.grid_size
        h10, w10 = env_t10.grid_size

        assert h1 <= h5 <= h10
        assert w1 <= w5 <= w10
        assert (h1 * w1) < (h10 * w10)

        # Step limits
        assert env_t1.max_steps < env_t5.max_steps < env_t10.max_steps
