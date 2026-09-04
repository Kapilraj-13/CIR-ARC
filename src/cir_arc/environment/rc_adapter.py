from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData, GameState, MultiLayerGrid

logger = logging.getLogger(__name__)

DEFAULT_ENV_DIRS = [
    r"D:\Data_and_Models\DATASETS\arc-prize-2026-arc-agi-3\environment_files",
    "environment_files",
    "data/environment_files",
]


class RCEngineAdapter(BaseEnvironment):
    """Adapter bridging native arcengine / arc_agi Arcade environments into CIR-ARC."""

    def __init__(
        self,
        game_id: str,
        scorecard_id: Optional[str] = None,
        environments_dir: Optional[str] = None,
    ) -> None:
        super().__init__(game_id)
        self.scorecard_id = scorecard_id
        self.environments_dir = environments_dir
        self._env: Any = None
        self._arc: Any = None
        self._owns_scorecard = False
        self._latest_obs: Optional[FrameData] = None
        self.step_count = 0

        self._init_native_engine()

    def _init_native_engine(self) -> None:
        # Resolve environments_dir
        resolved_dir = None
        if self.environments_dir and os.path.exists(self.environments_dir):
            resolved_dir = self.environments_dir
        else:
            for candidate in DEFAULT_ENV_DIRS:
                if os.path.exists(candidate):
                    resolved_dir = candidate
                    break

        try:
            from arc_agi import Arcade
            if resolved_dir:
                self._arc = Arcade(environments_dir=resolved_dir)
            else:
                self._arc = Arcade()

            if self.scorecard_id is None:
                self.scorecard_id = self._arc.open_scorecard(tags=["cir_arc", "phase3"])
                self._owns_scorecard = True
            self._env = self._arc.make(self.game_id, scorecard_id=self.scorecard_id)
            logger.info("Initialized native Arcade environment for %s (from %s)", self.game_id, resolved_dir)
        except Exception as e:
            logger.warning("Arcade/arcengine not available (%s); falling back to MockEngine mode.", e)
            from cir_arc.environment.mock_engine import MockEngine
            self._env = MockEngine(self.game_id)

    def _to_cir_frame(self, raw: Any, action_input: Optional[Action] = None) -> FrameData:
        if isinstance(raw, FrameData):
            return raw

        if raw is None:
            return FrameData(
                game_id=self.game_id,
                grid=MultiLayerGrid([np.zeros((1, 1), dtype=np.int16)]),
                state=GameState.NOT_PLAYED,
            )

        # raw is arcengine FrameDataRaw, FrameData, or pydantic response
        raw_layers = getattr(raw, "frame", [[]])
        grid = MultiLayerGrid.from_list(raw_layers)
        state_val = getattr(raw, "state", GameState.NOT_FINISHED)
        levels_completed = getattr(raw, "levels_completed", 0) or 0
        win_levels = getattr(raw, "win_levels", 1) or 1
        guid = getattr(raw, "guid", "") or ""
        full_reset = getattr(raw, "full_reset", False)
        avail = getattr(raw, "available_actions", [0, 1, 2, 3, 4, 5, 6, 7])

        return FrameData(
            game_id=self.game_id,
            grid=grid,
            state=GameState.from_value(state_val),
            levels_completed=levels_completed,
            win_levels=win_levels,
            action_input=action_input,
            available_actions=list(avail),
            guid=guid,
            full_reset=full_reset,
            step_count=self.step_count,
        )

    def reset(self) -> FrameData:
        self.step_count = 0
        if hasattr(self._env, "reset") and callable(self._env.reset):
            res = self._env.reset()
            self._latest_obs = self._to_cir_frame(res, action_input=Action(ActionType.RESET))
        else:
            from arcengine import GameAction
            res = self._env.step(GameAction.RESET, data={"game_id": self.game_id})
            self._latest_obs = self._to_cir_frame(res, action_input=Action(ActionType.RESET))
        return self._latest_obs

    def step(self, action: Action) -> FrameData:
        self.step_count += 1
        if hasattr(self._env, "step"):
            try:
                # arc_agi step accepts (action: int, data: dict, reasoning: str)
                act_val = int(action.action_id)
                act_data = dict(action.data) if action.data else {}
                res = self._env.step(action=act_val, data=act_data, reasoning=action.reasoning)
            except Exception as e:
                try:
                    # Fallback to direct action object
                    res = self._env.step(action)
                except Exception:
                    raise e
            self._latest_obs = self._to_cir_frame(res, action_input=action)
            return self._latest_obs
        raise RuntimeError("Underlying environment has no step method")

    def current_observation(self) -> Optional[FrameData]:
        if self._latest_obs is not None:
            return self._latest_obs
        if hasattr(self._env, "observation_space"):
            raw = self._env.observation_space
            if raw is not None:
                self._latest_obs = self._to_cir_frame(raw)
        return self._latest_obs

    def close(self) -> None:
        if self._owns_scorecard and self._arc and self.scorecard_id:
            try:
                self._arc.close_scorecard(self.scorecard_id)
            except Exception as e:
                logger.warning("Error closing scorecard: %s", e)
