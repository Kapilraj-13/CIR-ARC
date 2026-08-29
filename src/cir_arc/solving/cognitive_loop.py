from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData, GameState
from cir_arc.hypothesis.induction_engine import HypothesisInductionEngine
from cir_arc.probing.inspector_agent import EnvironmentInspector, EnvironmentProfile
from cir_arc.solving.code_agent import CodeAgentSolver

logger = logging.getLogger(__name__)


class CognitiveStage(str, Enum):
    INSPECT = "INSPECT"
    INDUCE = "INDUCE"
    PLAN = "PLAN"
    ACT = "ACT"
    DONE = "DONE"


@dataclass
class CognitiveContext:
    stage: CognitiveStage = CognitiveStage.INSPECT
    profile: Optional[EnvironmentProfile] = None
    action_plan: List[Action] = field(default_factory=list)
    plan_index: int = 0
    step_count: int = 0
    max_steps: int = 80
    memory: Dict[str, Any] = field(default_factory=dict)


class CognitiveLoop:
    """State graph cognitive loop coordinating Inspection -> Hypothesis Induction -> Planning -> Action Execution."""

    def __init__(self, env: BaseEnvironment, max_steps: int = 80) -> None:
        self.env = env
        self.max_steps = max_steps
        self.context = CognitiveContext(max_steps=max_steps)
        self.induction_engine = HypothesisInductionEngine()
        self.solver = CodeAgentSolver()

    def step(self) -> Tuple[Action, FrameData]:
        """Perform one cognitive loop step and return (action, next_frame)."""
        current_obs = self.env.current_observation()
        if current_obs is None or current_obs.state == GameState.NOT_PLAYED:
            frame = self.env.reset()
            return Action(ActionType.RESET), frame

        self.context.step_count += 1

        # Stage 1: Inspect if not already done
        if self.context.stage == CognitiveStage.INSPECT:
            inspector = EnvironmentInspector(self.env, max_probes=10)
            self.context.profile = inspector.inspect()
            self.context.stage = CognitiveStage.PLAN
            # Reset environment after probing to start clean solve run
            frame = self.env.reset()
            return Action(ActionType.RESET), frame

        # Stage 2: Plan if plan is empty or completed
        if self.context.stage == CognitiveStage.PLAN or self.context.plan_index >= len(self.context.action_plan):
            current_obs = self.env.current_observation() or self.env.reset()
            self.context.action_plan = self.solver.plan_solution(current_obs)
            self.context.plan_index = 0
            self.context.stage = CognitiveStage.ACT

        # Stage 3: Act
        if self.context.plan_index < len(self.context.action_plan):
            action = self.context.action_plan[self.context.plan_index]
            self.context.plan_index += 1
        else:
            action = Action(ActionType.ACTION1)

        before = self.env.current_observation() or self.env.reset()
        after = self.env.step(action)

        # Update hypotheses with observed transition
        self.induction_engine.observe_transition(before, action, after)

        if after.is_terminal or self.context.step_count >= self.max_steps:
            self.context.stage = CognitiveStage.DONE

        return action, after
