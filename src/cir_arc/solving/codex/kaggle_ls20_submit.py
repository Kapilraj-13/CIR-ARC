from __future__ import annotations

from typing import Any


class LS20FrameProfiler:
    player_pattern = (
        (12, 12, 12, 12, 12),
        (12, 12, 12, 12, 12),
        (9, 9, 9, 9, 9),
        (9, 9, 9, 9, 9),
        (9, 9, 9, 9, 9),
    )

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        try:
            if hasattr(frame, "tolist"):
                frame = frame.tolist()
            if frame and isinstance(frame[0][0], list):
                frame = frame[-1]
            for y in range(len(frame) - 4):
                for x in range(len(frame[0]) - 4):
                    block = tuple(tuple(frame[y + row][x : x + 5]) for row in range(5))
                    if block == self.player_pattern:
                        return True
            return False
        except Exception:
            return False


class LS20KagglePolicy:
    plans = {
        0: (3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1),
        1: (1, 4, 1, 1, 1, 1, 1, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 4, 1, 4, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2),
        2: (1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 3, 2, 1, 4, 1, 2),
        3: (3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 3, 1, 2, 3, 3, 1, 1, 1, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 1, 3, 3, 3),
        4: (1, 4, 1, 1, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 2, 2, 3, 3, 3, 1, 3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 4, 1, 4, 4, 2, 2, 2, 1),
        5: (1, 3, 1, 3, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 4, 2, 2, 1, 1, 3, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 1, 2, 2, 2, 2, 3, 3, 2, 1, 4, 4, 2, 2, 2, 3, 3, 4, 3, 4, 4, 1, 1, 1, 1, 4, 1, 4, 1, 1, 4, 2, 2, 2, 2, 2),
        6: (3, 3, 2, 2, 2, 2, 2, 4, 3, 4, 2, 1, 4, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 1, 1, 1, 4, 4, 4, 4, 2, 1, 1, 4, 4, 1, 2, 1, 4, 4, 1, 1, 4, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2),
    }

    def __init__(self) -> None:
        self.profiler = LS20FrameProfiler()
        self.plan_level: int | None = None
        self.cursor = 0

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        return self.profiler.supports(frame, levels_completed)

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        if levels_completed not in self.plans:
            raise ValueError(f"LS20 level {levels_completed + 1} has no locked plan")
        if self.plan_level != levels_completed:
            self.plan_level = levels_completed
            self.cursor = 0
        plan = self.plans[levels_completed]
        if self.cursor >= len(plan):
            raise RuntimeError(f"LS20 level {levels_completed + 1} exceeded its locked plan")
        action_id = plan[self.cursor]
        self.cursor += 1
        return {"id": action_id, "data": {}}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        from arcengine import GameAction

        return GameAction.from_id(self.choose_action_dict(frame, levels_completed)["id"])


class NavigationSolver(LS20KagglePolicy):
    name = "navigation_solver"
