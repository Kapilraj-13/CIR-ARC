from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from cir_arc.environment.frame import FrameData, GameState


@dataclass
class GamePhase:
    phase_name: str
    state: GameState
    levels_completed: int
    observation_index: int


class GameStateMachine:
    """Tracks phase state transitions across game lifecycle."""

    def __init__(self) -> None:
        self.phases: List[GamePhase] = []
        self._seen_transitions: Set[Tuple[str, int]] = set()

    def record_observation(self, index: int, frame: FrameData) -> Optional[GamePhase]:
        state_key = (frame.state.value, frame.levels_completed)
        if state_key in self._seen_transitions:
            return None

        self._seen_transitions.add(state_key)
        phase_name = self._classify_phase(frame.state, frame.levels_completed)
        phase = GamePhase(
            phase_name=phase_name,
            state=frame.state,
            levels_completed=frame.levels_completed,
            observation_index=index,
        )
        self.phases.append(phase)
        return phase

    @staticmethod
    def _classify_phase(state: GameState, levels_completed: int) -> str:
        if state == GameState.NOT_PLAYED:
            return "not_played"
        if state == GameState.WIN:
            return "win"
        if state == GameState.GAME_OVER:
            return "game_over"
        if levels_completed > 0:
            return f"level_{levels_completed}"
        return "active"

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "phase": p.phase_name,
                "state": p.state.value,
                "levels_completed": p.levels_completed,
                "observation_index": p.observation_index,
            }
            for p in self.phases
        ]
