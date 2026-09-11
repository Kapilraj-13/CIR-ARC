"""Codex Analytical Solver Hub.

Provides a unified interface for the 10 exact analytical ARC-AGI-3 solvers:
cd82, ft09, lp85, ls20, s5i5, sb26, sc25, tr87, tu93, wa30.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("CodexSolverHub")

SUPPORTED_GAMES = {
    "cd82", "ft09", "lp85", "ls20", "s5i5",
    "sb26", "sc25", "tr87", "tu93", "wa30",
}


def _normalize_frame_to_grid(frame: Any) -> list[list[int]]:
    """Convert any frame representation (FrameData, MultiLayerGrid, ndarray, or nested list)
    into a clean 2D list[list[int]].
    """
    if hasattr(frame, "grid") and hasattr(frame.grid, "layers"):
        arr = np.asarray(frame.grid.layers)
    elif hasattr(frame, "layers"):
        arr = np.asarray(frame.layers)
    elif hasattr(frame, "frame"):
        arr = np.asarray(frame.frame)
    else:
        arr = np.asarray(frame)

    if arr.ndim == 3:
        arr = arr[-1]
    elif arr.ndim == 1 and arr.size > 0 and isinstance(arr[0], (list, np.ndarray)):
        arr = np.asarray(arr[-1])

    return [[int(cell) for cell in row] for row in arr.tolist()]


class CodexSolverHub:
    """Manages active analytical policies for games with known deterministic solutions."""

    def __init__(self) -> None:
        self._current_game_key: Optional[str] = None
        self._active_policy: Any = None

    def match_game_key(self, game_id: str) -> Optional[str]:
        """Check if game_id corresponds to one of the 10 analytical games."""
        clean_id = game_id.lower().strip()
        for key in SUPPORTED_GAMES:
            if key in clean_id:
                return key
        return None

    def is_supported(self, game_id: str) -> bool:
        return self.match_game_key(game_id) is not None

    def reset_for_game(self, game_id: str) -> None:
        key = self.match_game_key(game_id)
        self._current_game_key = key
        self._active_policy = None

        if key is None:
            return

        try:
            if key == "cd82":
                from cir_arc.solving.codex.kaggle_cd82_submit import CD82KagglePolicy
                self._active_policy = CD82KagglePolicy()
            elif key == "ft09":
                from cir_arc.solving.codex.kaggle_ft09_submit import FT09KagglePolicy
                self._active_policy = FT09KagglePolicy()
            elif key == "lp85":
                from cir_arc.solving.codex.kaggle_lp85_submit import LP85KagglePolicy
                self._active_policy = LP85KagglePolicy()
            elif key == "ls20":
                from cir_arc.solving.codex.kaggle_ls20_submit import LS20KagglePolicy
                self._active_policy = LS20KagglePolicy()
            elif key == "s5i5":
                from cir_arc.solving.codex.kaggle_s5i5_submit import S5I5KagglePolicy
                self._active_policy = S5I5KagglePolicy()
            elif key == "sb26":
                from cir_arc.solving.codex.kaggle_sb26_submit import SB26KagglePolicy
                self._active_policy = SB26KagglePolicy()
            elif key == "sc25":
                from cir_arc.solving.codex.kaggle_sc25_submit import SC25KagglePolicy
                self._active_policy = SC25KagglePolicy()
            elif key == "tr87":
                from cir_arc.solving.codex.kaggle_tr87_submit import TR87KagglePolicy
                self._active_policy = TR87KagglePolicy()
            elif key == "tu93":
                from cir_arc.solving.codex.kaggle_tu93_submit import TU93KagglePolicy
                self._active_policy = TU93KagglePolicy()
            elif key == "wa30":
                from cir_arc.solving.codex.kaggle_wa30_submit import WA30KagglePolicy
                self._active_policy = WA30KagglePolicy()

            logger.info("Activated Codex Analytical Policy for %s (%s)", game_id, key)
        except Exception as e:
            logger.error("Failed to initialize Codex policy for %s: %s", game_id, e)
            self._active_policy = None

    def get_action(
        self,
        game_id: str,
        frame: Any,
        levels_completed: int = 0,
    ) -> Optional[Tuple[int, Optional[Dict[str, Any]]]]:
        """Query the analytical solver for the next action.
        Returns (action_id, data_dict) or None if not supported or exhausted.
        """
        key = self.match_game_key(game_id)
        if key is None:
            return None

        if self._current_game_key != key or self._active_policy is None:
            self.reset_for_game(game_id)

        if self._active_policy is None:
            return None

        try:
            grid = _normalize_frame_to_grid(frame)
            action_dict = self._active_policy.choose_action_dict(grid, levels_completed=levels_completed)
            if not action_dict:
                return None

            action_id = int(action_dict.get("id", 1))
            data = action_dict.get("data")
            data_dict = dict(data) if data else None
            return (action_id, data_dict)
        except Exception as e:
            logger.warning("Codex policy exception for %s (level %d): %s. Falling back.", game_id, levels_completed, e)
            return None
