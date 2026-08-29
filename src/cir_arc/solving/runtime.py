from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData, GameState
from cir_arc.recording.recorder import SessionRecorder
from cir_arc.solving.cognitive_loop import CognitiveLoop
from cir_arc.solving.telemetry import AgentOpsTelemetry

logger = logging.getLogger(__name__)


@dataclass
class ScorecardReport:
    game_id: str
    state: GameState
    levels_completed: int
    win_levels: int
    actions_taken: int
    elapsed_seconds: float
    is_win: bool
    recording_path: Optional[str] = None
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "state": self.state.value,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "actions_taken": self.actions_taken,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "is_win": self.is_win,
            "recording_path": self.recording_path,
        }


class SolvingRuntime:
    """Solving runtime managing environment lifecycle, cognitive loops, action budgets, and scorecards."""

    def __init__(self, max_actions: int = 80, record: bool = True) -> None:
        self.max_actions = max_actions
        self.record = record

    def run_game(self, env: BaseEnvironment) -> ScorecardReport:
        """Execute solving loop on the provided environment."""
        start_time = time.time()
        telemetry = AgentOpsTelemetry(agent_name="CIR-ARC-Solver")
        recorder = SessionRecorder(env.game_id) if self.record else None

        loop = CognitiveLoop(env, max_steps=self.max_actions)
        action_count = 0
        latest_frame: Optional[FrameData] = None

        telemetry.log_event("SESSION_START", {"game_id": env.game_id})

        while action_count < self.max_actions:
            action, frame = loop.step()
            action_count += 1
            latest_frame = frame

            if recorder:
                recorder.record(action)
                recorder.record(frame)

            telemetry.log_event(
                "STEP",
                {
                    "step": action_count,
                    "action": action.name,
                    "state": frame.state.value,
                    "levels_completed": frame.levels_completed,
                },
            )

            if frame.is_terminal:
                break

        elapsed = time.time() - start_time
        final_state = latest_frame.state if latest_frame else GameState.NOT_FINISHED
        levels = latest_frame.levels_completed if latest_frame else 0
        win_levels = latest_frame.win_levels if latest_frame else 1
        is_win = (final_state == GameState.WIN)

        rec_path = recorder.flush() if recorder else None

        telemetry.log_event(
            "SESSION_END",
            {
                "final_state": final_state.value,
                "is_win": is_win,
                "total_actions": action_count,
                "elapsed": elapsed,
            },
        )

        return ScorecardReport(
            game_id=env.game_id,
            state=final_state,
            levels_completed=levels,
            win_levels=win_levels,
            actions_taken=action_count,
            elapsed_seconds=elapsed,
            is_win=is_win,
            recording_path=rec_path,
            telemetry_summary=telemetry.get_summary(),
        )
