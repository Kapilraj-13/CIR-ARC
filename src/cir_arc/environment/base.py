from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cir_arc.environment.actions import ACTION_NAMES, Action, ActionSpec, ActionType
from cir_arc.environment.frame import FrameData


class BaseEnvironment(ABC):
    """Abstract interface for ARC-AGI-3 environments."""

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id

    @abstractmethod
    def reset(self) -> FrameData:
        """Reset environment to initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> FrameData:
        """Apply action and return resulting FrameData."""
        raise NotImplementedError

    @abstractmethod
    def current_observation(self) -> Optional[FrameData]:
        """Return latest observation or None if not started."""
        raise NotImplementedError

    def enumerate_actions(self, frame: Optional[FrameData] = None) -> List[ActionSpec]:
        """Enumerate available action specifications."""
        target_frame = frame or self.current_observation()
        available_ids = target_frame.available_actions if target_frame else [0, 1, 2, 3, 4, 5]
        specs: List[ActionSpec] = []
        for aid in sorted(set(available_ids)):
            try:
                atype = ActionType(aid)
                specs.append(
                    ActionSpec(
                        action_id=aid,
                        name=atype.name,
                        label=ACTION_NAMES.get(aid, atype.name),
                        is_complex=(atype == ActionType.ACTION6),
                    )
                )
            except ValueError:
                continue
        return specs

    def is_terminal(self) -> bool:
        obs = self.current_observation()
        return obs.is_terminal if obs else False

    def close(self) -> None:
        """Clean up resources."""
        pass
