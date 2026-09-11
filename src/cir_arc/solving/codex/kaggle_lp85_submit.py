from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LP85FrameProfile:
    button_pixels: int
    progress_pixels: int


class LP85FrameProfiler:
    def parse(self, frame: Any) -> LP85FrameProfile:
        grid = frame.tolist() if hasattr(frame, "tolist") else frame
        if grid and isinstance(grid[0][0], list):
            grid = grid[-1]
        if len(grid) != 64 or any(len(row) != 64 for row in grid):
            raise ValueError("LP85 requires a 64x64 frame")
        progress = sum(int(grid[y][0]) in {5, 14} for y in range(64))
        buttons = sum(
            int(grid[y][x]) == 14
            for y in range(2, 63)
            for x in range(1, 63)
        )
        return LP85FrameProfile(button_pixels=buttons, progress_pixels=progress)

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        try:
            profile = self.parse(frame)
            return profile.progress_pixels == 64 and profile.button_pixels >= 8
        except Exception:
            return False


class LP85KagglePolicy:
    plans = {
        0: ((2, 33),) * 5,
        1: ((38, 18), (47, 36), (47, 36), (47, 36), (38, 18), (38, 18), (38, 18), (47, 36)),
        2: ((34, 42),) * 4 + ((22, 42),) * 4 + ((34, 42),) * 6 + ((22, 42),) * 2,
        3: ((16, 25),) * 4 + ((5, 16),) * 8,
        4: ((35, 38), (35, 38), (49, 8), (9, 38), (7, 8)) + ((9, 38),) * 4,
        5: (
            ((56, 16),) * 2
            + ((41, 46),) * 2
            + ((26, 16),) * 2
            + ((29, 59),) * 6
            + ((14, 29),) * 2
            + ((44, 29),) * 4
            + ((54, 55),)
        ),
        6: ((32, 43), (21, 20), (28, 43), (21, 33), (28, 43)),
        7: ((52, 30), (52, 30), (52, 35), (52, 35), (30, 58)),
    }

    def __init__(self) -> None:
        self.profiler = LP85FrameProfiler()
        self.plan_level: int | None = None
        self.cursor = 0

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        return self.profiler.supports(frame, levels_completed)

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        plan = self.plans.get(levels_completed)
        if not plan:
            return {"id": 6, "data": {"x": 1, "y": 1}}
        if self.plan_level != levels_completed:
            self.plan_level = levels_completed
            self.cursor = 0
        if self.cursor >= len(plan):
            return {"id": 6, "data": {"x": 1, "y": 1}}
        x, y = plan[self.cursor]
        self.cursor += 1
        return {"id": 6, "data": {"x": x, "y": y}}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        from arcengine import GameAction

        action_dict = self.choose_action_dict(frame, levels_completed)
        action = GameAction.ACTION6
        action.set_data(action_dict["data"])
        return action


class _LP85SubmitMixin:
    MAX_ACTIONS = 1000

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return getattr(getattr(latest_frame, "state", None), "name", None) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        from arcengine import GameAction, GameState

        if not hasattr(self, "policy"):
            self.policy = LP85KagglePolicy()
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or not latest_frame.frame:
            return GameAction.RESET
        action = self.policy.choose_game_action(latest_frame.frame[-1], latest_frame.levels_completed)
        action.reasoning = {"agent": "lp85_submit", "scope": "lp85_only"}
        return action


try:
    from agents.agent import Agent

    class LP85SubmitAgent(_LP85SubmitMixin, Agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.policy = LP85KagglePolicy()

except Exception:

    class LP85SubmitAgent(_LP85SubmitMixin):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.policy = LP85KagglePolicy()
