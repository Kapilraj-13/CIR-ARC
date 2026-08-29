from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from cir_arc.environment.actions import Action


class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"

    @classmethod
    def from_value(cls, val: Any) -> GameState:
        if isinstance(val, GameState):
            return val
        s = str(val).upper()
        if hasattr(val, "name"):
            s = val.name.upper()
        elif hasattr(val, "value"):
            s = str(val.value).upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.NOT_FINISHED


@dataclass
class MultiLayerGrid:
    layers: List[np.ndarray]  # each layer is (H, W) dtype int16/uint8

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [np.zeros((1, 1), dtype=np.int16)]
        validated: List[np.ndarray] = []
        for l in self.layers:
            arr = np.asarray(l, dtype=np.int16)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim > 2:
                # If 3D, take slices
                for i in range(arr.shape[0]):
                    validated.append(arr[i])
                continue
            validated.append(arr)
        self.layers = validated

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def height(self) -> int:
        return self.layers[0].shape[0] if self.layers else 0

    @property
    def width(self) -> int:
        return self.layers[0].shape[1] if self.layers else 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_layers, self.height, self.width)

    def composite(self, background_color: int = 0) -> np.ndarray:
        """Composite all layers from bottom (layer 0) to top. Non-background overrides."""
        if not self.layers:
            return np.zeros((0, 0), dtype=np.int16)
        comp = self.layers[0].copy()
        for layer in self.layers[1:]:
            mask = (layer != background_color)
            comp[mask] = layer[mask]
        return comp

    def to_list(self) -> List[List[List[int]]]:
        return [layer.tolist() for layer in self.layers]

    @classmethod
    def from_list(cls, data: Sequence[Any]) -> MultiLayerGrid:
        if not data:
            return cls([np.zeros((1, 1), dtype=np.int16)])
        arr = np.asarray(data, dtype=np.int16)
        if arr.ndim == 2:
            return cls([arr])
        elif arr.ndim == 3:
            return cls([arr[i] for i in range(arr.shape[0])])
        elif arr.ndim == 1:
            return cls([arr.reshape(1, -1)])
        return cls([np.zeros((1, 1), dtype=np.int16)])

    def hash(self) -> str:
        arr_bytes = b"".join(layer.tobytes() for layer in self.layers)
        return hashlib.sha256(arr_bytes).hexdigest()[:16]


@dataclass
class FrameData:
    game_id: str
    grid: MultiLayerGrid
    state: GameState = GameState.NOT_PLAYED
    levels_completed: int = 0
    win_levels: int = 1
    action_input: Optional[Action] = None
    available_actions: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    guid: str = ""
    full_reset: bool = False
    step_count: int = 0

    @property
    def is_win(self) -> bool:
        return self.state == GameState.WIN

    @property
    def is_game_over(self) -> bool:
        return self.state == GameState.GAME_OVER

    @property
    def is_terminal(self) -> bool:
        return self.is_win or self.is_game_over

    def hash(self) -> str:
        payload = {
            "grid_hash": self.grid.hash(),
            "state": self.state.value,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "available_actions": sorted(self.available_actions),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "frame": self.grid.to_list(),
            "state": self.state.value,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "guid": self.guid,
            "full_reset": self.full_reset,
            "available_actions": self.available_actions,
            "step_count": self.step_count,
            "action_input": self.action_input.to_dict() if self.action_input else None,
        }
