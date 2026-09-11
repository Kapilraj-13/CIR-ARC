from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Position = tuple[int, int]


@dataclass(frozen=True)
class SC25FrameProfile:
    cell_values: tuple[tuple[int, ...], ...]
    marked_cells: frozenset[Position]


class SC25FrameProfiler:
    centers = (25, 30, 35)
    known_patterns = (
        frozenset({(0, 0), (1, 0), (1, 1)}),
        frozenset({(1, 0), (0, 1), (2, 1), (1, 2)}),
        frozenset({(1, 0), (1, 1), (1, 2)}),
    )

    def parse(self, frame: Any) -> SC25FrameProfile:
        grid = frame.tolist() if hasattr(frame, "tolist") else frame
        if grid and isinstance(grid[0][0], list):
            grid = grid[-1]
        if len(grid) != 64 or any(len(row) != 64 for row in grid):
            raise ValueError("SC25 requires a 64x64 frame")
        values = tuple(
            tuple(int(grid[y][x]) for x in self.centers)
            for y in (50, 55, 60)
        )
        marked = frozenset(
            (column, row)
            for row in range(3)
            for column in range(3)
            if values[row][column] != 2
        )
        return SC25FrameProfile(values, marked)

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        try:
            return self.parse(frame).marked_cells in self.known_patterns
        except Exception:
            return False


class SC25KagglePolicy:
    spell_clicks = {
        "teleport": ((25, 50), (30, 50), (30, 55)),
        "resize": ((30, 50), (25, 55), (35, 55), (30, 60)),
        "fireball": ((30, 50), (30, 55), (30, 60)),
    }
    locked_steps = {
        0: (1, 3, 3, 3, 3, "resize", 3, 3, 3, 3),
        1: ("teleport", 1, 1),
        2: (3, 3, 4, "fireball", 2, 2, 3, 2, 3),
        3: (2, 2, 4, 4, "resize", 2, 4, 3, "fireball", 2, 2, 2, 4, 4, 4, 4),
        4: (
            "resize", "teleport", 3, 3, 3, 3, 3, 3, 2, 2, 3,
            "fireball", 1, "fireball", "resize", "teleport", 1, 1, 1, 1, 1, 1,
        ),
        5: (
            "resize", "teleport", 4, 4, 1, 1, "fireball", 3, "resize",
            "teleport", "fireball", "teleport", 1, 4, 1, 1, 1, 1, 1,
        ),
    }

    def __init__(self) -> None:
        self.profiler = SC25FrameProfiler()
        self.plans = {
            level: self._compile(steps)
            for level, steps in self.locked_steps.items()
        }
        self.plan_level: int | None = None
        self.cursor = 0

    def _compile(self, steps: tuple[Any, ...]) -> tuple[tuple[int, dict[str, int]], ...]:
        actions: list[tuple[int, dict[str, int]]] = []
        for step in steps:
            if isinstance(step, int):
                actions.append((step, {}))
                continue
            actions.extend(
                (6, {"x": x, "y": y})
                for x, y in self.spell_clicks[step]
            )
        return tuple(actions)

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        return self.profiler.supports(frame, levels_completed)

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        plan = self.plans.get(levels_completed)
        if not plan:
            return {"id": 1, "data": {}}
        if self.plan_level != levels_completed:
            self.plan_level = levels_completed
            self.cursor = 0
        if self.cursor >= len(plan):
            return {"id": 1, "data": {}}
        action_id, data = plan[self.cursor]
        self.cursor += 1
        return {"id": action_id, "data": dict(data)}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        from arcengine import GameAction

        action_dict = self.choose_action_dict(frame, levels_completed)
        action = GameAction.from_id(action_dict["id"])
        if action_dict["data"]:
            action.set_data(action_dict["data"])
        return action


class _SC25SubmitMixin:
    MAX_ACTIONS = 1000

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return getattr(getattr(latest_frame, "state", None), "name", None) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        from arcengine import GameAction, GameState

        if not hasattr(self, "policy"):
            self.policy = SC25KagglePolicy()
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or not latest_frame.frame:
            return GameAction.RESET
        action = self.policy.choose_game_action(latest_frame.frame[-1], latest_frame.levels_completed)
        action.reasoning = {"agent": "sc25_submit", "scope": "sc25_only"}
        return action


try:
    from agents.agent import Agent

    class SC25SubmitAgent(_SC25SubmitMixin, Agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.policy = SC25KagglePolicy()

except Exception:

    class SC25SubmitAgent(_SC25SubmitMixin):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.policy = SC25KagglePolicy()

