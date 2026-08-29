"""Model-based cognitive loop orchestrating Belief -> Goal -> World Model -> Planning -> Execution -> Replay -> Repair."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData, GameState
from cir_arc.goals.manager import GoalManager
from cir_arc.hypothesis.counterexample import Counterexample, CounterexampleDetector
from cir_arc.hypothesis.induction_engine import HypothesisInductionEngine
from cir_arc.hypothesis.repair import HypothesisRepair
from cir_arc.planning.hierarchical import HierarchicalPlanner
from cir_arc.probing.information_gain import InformationGainExplorer
from cir_arc.recovery.contradiction import PlanContradictionDetector
from cir_arc.recovery.replanner import DynamicReplanner
from cir_arc.recovery.rollback import StateRollback
from cir_arc.world_model.executable import ExecutableWorldModel
from cir_arc.world_model.replay import ReplayVerifier
from cir_arc.world_model.validator import WorldModelValidator

logger = logging.getLogger(__name__)


class CognitiveStage(str, Enum):
    OBSERVE = "OBSERVE"
    INFER = "INFER"
    PLAN = "PLAN"
    ACT = "ACT"
    REPLAY_VERIFY = "REPLAY_VERIFY"
    REPAIR = "REPAIR"
    DONE = "DONE"


@dataclass
class CognitiveContext:
    stage: CognitiveStage = CognitiveStage.OBSERVE
    step_count: int = 0
    max_steps: int = 80
    action_plan: List[Action] = field(default_factory=list)
    plan_index: int = 0
    telemetry: Dict[str, Any] = field(default_factory=dict)


class CognitiveLoop:
    """Closed-loop model-based agent integrating belief, goal inference, world model digital twin, replay verification, and repair."""

    def __init__(self, env: BaseEnvironment, max_steps: int = 80) -> None:
        self.env = env
        self.max_steps = max_steps
        self.context = CognitiveContext(max_steps=max_steps)

        # Architectural Core Modules
        self.belief = BeliefState(game_id=getattr(env, "game_id", "unknown_game"))
        self.goal_manager = GoalManager()
        self.world_model = ExecutableWorldModel()
        self.induction_engine = HypothesisInductionEngine()
        self.info_explorer = InformationGainExplorer()
        self.planner = HierarchicalPlanner(goal_manager=self.goal_manager, info_explorer=self.info_explorer)
        self.replanner = DynamicReplanner(planner=self.planner, goal_manager=self.goal_manager)
        self.rollback = StateRollback()

        # Verification & Digital Twin Subsystem
        self.replay_verifier = ReplayVerifier(world_model=self.world_model)
        self.validator = WorldModelValidator()
        self.repair = HypothesisRepair()

        self.last_observation: Optional[FrameData] = None

    def step(self) -> Tuple[Action, FrameData]:
        """Performs one complete cycle of the model-based cognitive loop."""
        current_obs = self.env.current_observation()
        if current_obs is None or current_obs.state == GameState.NOT_PLAYED:
            frame = self.env.reset()
            self.belief.update_from_frame(frame)
            self.last_observation = frame
            return Action(ActionType.RESET), frame

        self.context.step_count += 1
        comp_grid = current_obs.grid.composite()

        # Step 1: Update Belief State
        self.belief.update_from_frame(current_obs)
        if self.belief.player_location:
            self.rollback.record_position(self.belief.player_location)

        # Step 2: Goal Inference & Ranking
        self.goal_manager.update_from_belief(self.belief, comp_grid)

        # Step 3: Check for cycle or deadlock -> Trigger Breakout Action if stuck
        if self.rollback.is_in_cycle():
            action = self.rollback.get_breakout_action(current_obs.available_actions)
            self.context.action_plan.clear()
            self.context.plan_index = 0
        else:
            # Step 4: Hierarchical Planning (Strategic -> Tactical -> Spatial Path -> Action)
            if self.context.plan_index >= len(self.context.action_plan):
                self.context.action_plan = self.planner.plan(
                    belief=self.belief,
                    comp_grid=comp_grid,
                    available_actions=current_obs.available_actions,
                )
                self.context.plan_index = 0

            # Select next action from plan
            if self.context.plan_index < len(self.context.action_plan):
                action = self.context.action_plan[self.context.plan_index]
                self.context.plan_index += 1
            else:
                action = self.info_explorer.select_best_exploratory_action(
                    self.belief, comp_grid, current_obs.available_actions
                )

        # Step 5: Execute Action in Environment
        before = current_obs
        after = self.env.step(action)
        self.last_observation = after

        # Step 6: Replay Verification against Digital Twin World Model
        is_consistent, counterexample = self.replay_verifier.verify_transition(before, action, after)
        self.validator.record_evaluation(is_consistent)

        # Step 7: Contradiction Detection & Model Repair
        movement_contradiction = PlanContradictionDetector.check_movement_contradiction(before, action, after)
        if not is_consistent and counterexample is not None:
            self.repair.repair_from_counterexample(counterexample, self.belief, self.induction_engine)
            # Trigger immediate dynamic replan
            self.context.action_plan = self.replanner.trigger_replan(
                self.belief, comp_grid, after.available_actions, reason=counterexample.failed_assumption
            )
            self.context.plan_index = 0
        elif movement_contradiction:
            # Player tried to move into blocked cell -> trigger dynamic replan
            self.context.action_plan = self.replanner.trigger_replan(
                self.belief, comp_grid, after.available_actions, reason="Movement blocked by obstacle"
            )
            self.context.plan_index = 0

        # Step 8: Update Hypothesis Induction Engine
        self.induction_engine.observe_transition(before, action, after)

        # Check termination
        if after.is_terminal or self.context.step_count >= self.max_steps:
            self.context.stage = CognitiveStage.DONE

        return action, after
