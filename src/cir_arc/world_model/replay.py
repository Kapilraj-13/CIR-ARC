"""Replay verification for empirical world model validation against recorded traces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData
from cir_arc.hypothesis.counterexample import Counterexample, CounterexampleDetector
from cir_arc.world_model.executable import ExecutableWorldModel


class ReplayVerifier:
    """Replays executed actions in digital twin world model and verifies against actual observations."""

    def __init__(
        self,
        world_model: Optional[ExecutableWorldModel] = None,
        detector: Optional[CounterexampleDetector] = None,
    ) -> None:
        self.world_model = world_model or ExecutableWorldModel()
        self.detector = detector or CounterexampleDetector()

    def verify_transition(
        self,
        before_frame: FrameData,
        action: Action,
        after_frame: FrameData,
    ) -> Tuple[bool, Optional[Counterexample]]:
        """Compares single forward simulation step against actual observed after_frame."""
        before_comp = before_frame.grid.composite()
        predicted_grid, meta = self.world_model.simulate_step(before_comp, action)

        ce = self.detector.detect_counterexample(
            predicted_grid=predicted_grid,
            actual_frame=after_frame,
            action=action,
        )
        is_consistent = (ce is None)
        return is_consistent, ce

    def replay_history(
        self,
        frame_history: List[FrameData],
        action_history: List[Action],
    ) -> Dict[str, Any]:
        """Replays a full trajectory of actions and counts discrepancies."""
        if len(frame_history) < 2 or len(action_history) < 1:
            return {"verified": True, "counterexamples": []}

        counterexamples = []
        for i in range(len(action_history)):
            if i + 1 >= len(frame_history):
                break
            before_f = frame_history[i]
            act = action_history[i]
            after_f = frame_history[i + 1]

            is_consistent, ce = self.verify_transition(before_f, act, after_f)
            if not is_consistent and ce is not None:
                counterexamples.append(ce)

        total_steps = len(action_history)
        match_rate = float(total_steps - len(counterexamples)) / max(1, total_steps)

        return {
            "verified": len(counterexamples) == 0,
            "total_steps": total_steps,
            "counterexamples": counterexamples,
            "match_rate": match_rate,
        }
