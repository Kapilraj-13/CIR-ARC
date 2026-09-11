from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable


Grid = list[list[int]]
Mask = frozenset[tuple[int, int]]


@dataclass(frozen=True)
class PaintOperation:
    mask_index: int
    color: int


@dataclass(frozen=True)
class CompiledAction:
    id: int
    data: dict[str, int] | None = None

    def to_action_input(self) -> Any:
        from arcengine import ActionInput, GameAction

        return ActionInput(id=GameAction.from_id(self.id), data=dict(self.data or {}))

    def to_game_action(self) -> Any:
        from arcengine import GameAction

        action = GameAction.from_id(self.id)
        if self.data:
            action.set_data(dict(self.data))
        return action


class CD82CellPlanner:
    size = 10
    initial_color = 0
    target_origin = (3, 3)
    action_budget = 100
    ring_positions = {
        0: (0, 1),
        1: (0, 2),
        2: (1, 2),
        3: (2, 2),
        4: (2, 1),
        5: (2, 0),
        6: (1, 0),
        7: (0, 0),
    }
    action_deltas = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    center_mask_to_ring = {
        8: 0,
        9: 2,
        10: 4,
        11: 6,
    }
    arrow_centers = {
        0: (32, 20),
        2: (51, 38),
        4: (32, 57),
        6: (14, 38),
    }

    def __init__(self) -> None:
        self.masks = self._build_masks()
        self.diagonal_cells = {
            (i, i) for i in range(self.size)
        } | {
            (i, self.size - 1 - i) for i in range(self.size)
        }

    def compile_from_frame(self, frame: Any) -> list[CompiledAction]:
        grid = self._coerce_frame(frame)
        target = self.extract_target(grid)
        palette = self.detect_palette(grid)
        operations = self.plan_paint_operations(target)
        if operations is None:
            return []
        actions = self.compile_operations(operations, palette)
        if len(actions) > self.action_budget:
            return []
        return actions

    def extract_target(self, frame: Grid) -> Grid:
        x0, y0 = self.target_origin
        return [
            [int(frame[y0 + row][x0 + col]) for col in range(self.size)]
            for row in range(self.size)
        ]

    def detect_palette(self, frame: Grid) -> dict[int, tuple[int, int]]:
        palette: dict[int, tuple[int, int]] = {}
        height = len(frame)
        width = len(frame[0]) if height else 0
        for y in range(0, min(12, height - 4)):
            for x in range(0, max(0, width - 4)):
                if not self._is_palette_swatch(frame, x, y):
                    continue
                color = int(frame[y + 2][x + 2])
                palette[color] = (x + 2, y + 2)
        return palette

    def plan_paint_operations(self, target: Grid) -> list[PaintOperation] | None:
        remaining = {
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in self.diagonal_cells
        }
        reverse_plan = self._search_reverse_plan(target, frozenset(remaining), {})
        if reverse_plan is None:
            return None
        return list(reversed(reverse_plan))

    def compile_operations(
        self,
        operations: Iterable[PaintOperation],
        palette: dict[int, tuple[int, int]],
    ) -> list[CompiledAction]:
        actions: list[CompiledAction] = []
        current_mask = 0
        current_color = 15
        for operation in operations:
            target_mask = self.center_mask_to_ring.get(operation.mask_index, operation.mask_index)
            actions.extend(CompiledAction(id=action_id) for action_id in self._mask_path(current_mask, target_mask))
            current_mask = target_mask

            if operation.color != current_color:
                if operation.color not in palette:
                    return []
                x, y = palette[operation.color]
                actions.append(CompiledAction(id=6, data={"x": x, "y": y}))
                current_color = operation.color

            if operation.mask_index in self.center_mask_to_ring:
                x, y = self.arrow_centers[target_mask]
                actions.append(CompiledAction(id=6, data={"x": x, "y": y}))
            else:
                actions.append(CompiledAction(id=5))
        return actions

    def apply_operations(self, operations: Iterable[PaintOperation]) -> Grid:
        canvas = [[self.initial_color for _ in range(self.size)] for _ in range(self.size)]
        for operation in operations:
            for row, col in self.masks[operation.mask_index]:
                canvas[row][col] = operation.color
        return canvas

    def matches_target(self, canvas: Grid, target: Grid) -> bool:
        for row in range(self.size):
            for col in range(self.size):
                if (row, col) in self.diagonal_cells:
                    continue
                if canvas[row][col] != target[row][col]:
                    return False
        return True

    def _search_reverse_plan(
        self,
        target: Grid,
        remaining: frozenset[tuple[int, int]],
        memo: dict[frozenset[tuple[int, int]], list[PaintOperation] | None],
    ) -> list[PaintOperation] | None:
        if not remaining or all(target[row][col] == self.initial_color for row, col in remaining):
            return []
        if remaining in memo:
            return memo[remaining]

        for candidate in self._reverse_candidates(target, remaining):
            result = self._search_reverse_plan(
                target,
                frozenset(remaining - self.masks[candidate.mask_index]),
                memo,
            )
            if result is not None:
                memo[remaining] = [candidate] + result
                return memo[remaining]
        memo[remaining] = None
        return None

    def _reverse_candidates(self, target: Grid, remaining: frozenset[tuple[int, int]]) -> list[PaintOperation]:
        candidates: list[tuple[int, int, int]] = []
        for mask_index, mask in self.masks.items():
            cells = sorted(remaining & mask)
            if not cells:
                continue
            colors = {target[row][col] for row, col in cells}
            if len(colors) != 1:
                continue
            color = next(iter(colors))
            if color == self.initial_color:
                continue
            candidates.append((len(cells), mask_index, color))
        candidates.sort(reverse=True)
        return [PaintOperation(mask_index=mask_index, color=color) for _, mask_index, color in candidates]

    def _mask_path(self, start: int, goal: int) -> list[int]:
        if start == goal:
            return []
        start_pos = self.ring_positions[start]
        goal_pos = self.ring_positions[goal]
        position_to_index = {position: index for index, position in self.ring_positions.items()}
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start_pos, [])])
        seen = {start_pos}
        while queue:
            position, path = queue.popleft()
            if position == goal_pos:
                return path
            x, y = position
            for action_id, (dx, dy) in self.action_deltas.items():
                next_position = (x + dx, y + dy)
                if next_position not in position_to_index or next_position == (1, 1):
                    continue
                if next_position in seen:
                    continue
                seen.add(next_position)
                queue.append((next_position, path + [action_id]))
        return []

    def _build_masks(self) -> dict[int, Mask]:
        size = self.size
        masks: dict[int, set[tuple[int, int]]] = {
            0: {(row, col) for row in range(0, 5) for col in range(size)},
            2: {(row, col) for row in range(size) for col in range(5, 10)},
            4: {(row, col) for row in range(5, 10) for col in range(size)},
            6: {(row, col) for row in range(size) for col in range(0, 5)},
            1: {(row, col) for row in range(size) for col in range(row, size)},
            3: {(row, col) for row in range(size) for col in range(size - 1 - row, size)},
            5: {(row, col) for row in range(size) for col in range(0, row + 1)},
            7: {(row, col) for row in range(size) for col in range(0, size - row)},
            8: {(row, col) for row in range(0, 3) for col in range(3, 7)},
            9: {(row, col) for row in range(3, 7) for col in range(7, 10)},
            10: {(row, col) for row in range(7, 10) for col in range(3, 7)},
            11: {(row, col) for row in range(3, 7) for col in range(0, 3)},
        }
        return {index: frozenset(cells) for index, cells in masks.items()}

    def _is_palette_swatch(self, frame: Grid, x: int, y: int) -> bool:
        try:
            top = all(frame[y][x + col] == 4 for col in range(5))
            bottom = all(frame[y + 4][x + col] == 4 for col in range(5))
            left = all(frame[y + row][x] == 4 for row in range(5))
            right = all(frame[y + row][x + 4] == 4 for row in range(5))
            inner = {frame[y + row][x + col] for row in range(1, 4) for col in range(1, 4)}
        except IndexError:
            return False
        return top and bottom and left and right and len(inner) == 1

    def _coerce_frame(self, frame: Any) -> Grid:
        import numpy as np
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = arr[-1]
        return [[int(cell) for cell in row] for row in arr.tolist()]


class CD82KagglePolicy:
    def __init__(self) -> None:
        self.planner = CD82CellPlanner()
        self.plan: list[CompiledAction] = []
        self.cursor = 0
        self.plan_level: int | None = None

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        if self.plan_level != levels_completed or self.cursor >= len(self.plan):
            self.plan = self.planner.compile_from_frame(frame)
            self.cursor = 0
            self.plan_level = levels_completed

        if not self.plan:
            return {"id": 5, "data": {}}

        action = self.plan[self.cursor]
        self.cursor += 1
        return {"id": action.id, "data": dict(action.data or {})}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        action_dict = self.choose_action_dict(frame, levels_completed)
        action = CompiledAction(id=action_dict["id"], data=action_dict.get("data") or None)
        return action.to_game_action()


class _CD82SubmitMixin:
    MAX_ACTIONS = 100

    def _ensure_cd82_policy(self) -> None:
        if not hasattr(self, "policy"):
            self.policy = CD82KagglePolicy()

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return getattr(getattr(latest_frame, "state", None), "name", None) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        from arcengine import GameAction, GameState

        self._ensure_cd82_policy()
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or not latest_frame.frame:
            return GameAction.RESET
        action = self.policy.choose_game_action(
            latest_frame.frame[-1],
            latest_frame.levels_completed,
        )
        action.reasoning = {"agent": "cd82_submit", "scope": "cd82_only"}
        return action


try:
    from agents.agent import Agent

    class CD82SubmitAgent(_CD82SubmitMixin, Agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.policy = CD82KagglePolicy()

except Exception:

    class CD82SubmitAgent(_CD82SubmitMixin):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.policy = CD82KagglePolicy()
