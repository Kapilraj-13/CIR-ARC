from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


class ActionType(IntEnum):
    RESET = 0
    ACTION1 = 1  # Up
    ACTION2 = 2  # Down
    ACTION3 = 3  # Left
    ACTION4 = 4  # Right
    ACTION5 = 5  # Interact / Action
    ACTION6 = 6  # Parameterized Click (x, y)
    ACTION7 = 7  # Undo

    # Semantic aliases
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5
    CLICK = 6
    UNDO = 7


ACTION_NAMES: Dict[int, str] = {
    0: "RESET",
    1: "ACTION1 (Up)",
    2: "ACTION2 (Down)",
    3: "ACTION3 (Left)",
    4: "ACTION4 (Right)",
    5: "ACTION5 (Interact)",
    6: "ACTION6 (Click)",
    7: "ACTION7 (Undo)",
}

DIRECTION_VECTORS: Dict[int, tuple[int, int]] = {
    1: (-1, 0),  # Up (row - 1)
    2: (1, 0),   # Down (row + 1)
    3: (0, -1),  # Left (col - 1)
    4: (0, 1),   # Right (col + 1)
}


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    name: str
    label: str
    is_complex: bool


@dataclass
class Action:
    action_type: ActionType
    data: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)

    @property
    def action_id(self) -> int:
        return int(self.action_type)

    @property
    def name(self) -> str:
        return self.action_type.name

    @property
    def is_complex(self) -> bool:
        return self.action_type == ActionType.ACTION6

    @classmethod
    def from_id(cls, action_id: int, data: Optional[Dict[str, Any]] = None, reasoning: Optional[Dict[str, Any]] = None) -> Action:
        return cls(
            action_type=ActionType(action_id),
            data=data or {},
            reasoning=reasoning or {},
        )

    @classmethod
    def from_int(cls, action_id: int, data: Optional[Dict[str, Any]] = None, reasoning: Optional[Dict[str, Any]] = None) -> Action:
        return cls.from_id(action_id, data=data, reasoning=reasoning)

    @classmethod
    def click(cls, x: int, y: int, extra: Optional[Dict[str, Any]] = None) -> Action:
        payload = {"x": x, "y": y}
        if extra:
            payload.update(extra)
        return cls(action_type=ActionType.ACTION6, data=payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.action_id,
            "name": self.name,
            "data": self.data,
            "reasoning": self.reasoning,
        }
