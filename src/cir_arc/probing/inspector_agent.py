from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cir_arc.environment.actions import Action, ActionSpec, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData
from cir_arc.environment.state_delta import StateDelta, compute_state_delta, find_objects
from cir_arc.probing.action_matrix import ActionEffectMatrix
from cir_arc.probing.object_catalog import DynamicObjectCatalog
from cir_arc.probing.resource_inspector import ResourceIndicator, ResourceInspector
from cir_arc.probing.state_machine import GamePhase, GameStateMachine

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentProfile:
    game_id: str
    action_matrix: ActionEffectMatrix
    object_catalog: DynamicObjectCatalog
    resources: List[ResourceIndicator]
    phases: List[GamePhase]
    probe_count: int = 0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "actions": self.action_matrix.to_dict(),
            "objects": self.object_catalog.to_list(),
            "resources": [
                {"name": r.name, "value": r.value, "source": r.source}
                for r in self.resources
            ],
            "phases": [
                {
                    "phase": p.phase_name,
                    "state": p.state.value,
                    "levels_completed": p.levels_completed,
                    "observation_index": p.observation_index,
                }
                for p in self.phases
            ],
            "probe_count": self.probe_count,
            "grid_shape": list(self.grid_shape),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class EnvironmentInspector:
    """Active exploratory probing agent that inspects actions, objects, resources, and phases."""

    CLICK_POINTS = [(1, 1), (2, 2), (4, 4), (5, 5), (8, 8)]

    def __init__(self, env: BaseEnvironment, max_probes: int = 20) -> None:
        self.env = env
        self.max_probes = max_probes
        self.action_matrix = ActionEffectMatrix()
        self.object_catalog = DynamicObjectCatalog()
        self.state_machine = GameStateMachine()

    def inspect(self) -> EnvironmentProfile:
        observations: List[FrameData] = []
        probe_count = 0

        # Step 1: Baseline RESET
        baseline = self.env.reset()
        observations.append(baseline)
        self.state_machine.record_observation(0, baseline)
        self.object_catalog.register_objects(find_objects(baseline))
        probe_count += 1

        available_specs = self.env.enumerate_actions(baseline)

        # Step 2: Probe discrete actions
        for spec in available_specs:
            if spec.action_id == 0:  # Skip redundant RESET
                continue
            if probe_count >= self.max_probes:
                break

            if not spec.is_complex:
                before = self.env.reset()
                action = Action(ActionType(spec.action_id))
                after = self.env.step(action)
                observations.append(after)
                probe_count += 1

                self.state_machine.record_observation(probe_count - 1, after)
                delta = compute_state_delta(before, after, action)
                self.action_matrix.record_probe(action, delta)
                self.object_catalog.update_from_delta(delta)

        # Step 3: Probe complex / click actions if available
        has_click = any(s.is_complex for s in available_specs)
        if has_click:
            for cx, cy in self.CLICK_POINTS:
                if probe_count >= self.max_probes:
                    break
                before = self.env.reset()
                action = Action.click(cx, cy)
                after = self.env.step(action)
                observations.append(after)
                probe_count += 1

                self.state_machine.record_observation(probe_count - 1, after)
                delta = compute_state_delta(before, after, action)
                self.action_matrix.record_probe(action, delta)
                self.object_catalog.update_from_delta(delta)

        # Step 4: Aggregate resources
        resources = ResourceInspector.inspect(observations)

        profile = EnvironmentProfile(
            game_id=self.env.game_id,
            action_matrix=self.action_matrix,
            object_catalog=self.object_catalog,
            resources=resources,
            phases=self.state_machine.phases,
            probe_count=probe_count,
            grid_shape=baseline.grid.shape,
        )

        logger.info(
            "Inspector completed for %s: %d active actions, %d objects, %d phases",
            self.env.game_id,
            len(self.action_matrix.get_active_actions()),
            len(self.object_catalog.catalog),
            len(self.state_machine.phases),
        )

        return profile
