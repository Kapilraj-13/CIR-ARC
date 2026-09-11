from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


Grid = list[list[int]]
Position = tuple[int, int]


@dataclass(frozen=True)
class Node:
    position: Position
    color: int
    effects: tuple[Position, ...] = ((0, 0),)


@dataclass(frozen=True)
class Clue:
    position: Position
    color: int
    pattern: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Board:
    nodes: tuple[Node, ...]
    clues: tuple[Clue, ...]
    palette: tuple[int, ...]
    scale: int


@dataclass(frozen=True)
class CompiledAction:
    id: int
    data: dict[str, int]

    def to_game_action(self) -> Any:
        from arcengine import GameAction

        action = GameAction.from_id(self.id)
        action.set_data(dict(self.data))
        return action


class FT09CellPlanner:
    size = 32
    structural = frozenset({0, 2, 3, 4, 5, 6, 7})
    action_budget = 128

    def compile_from_frame(self, frame: Any) -> list[CompiledAction]:
        board = self.parse_frame(frame)
        allowed = self._allowed(board)
        counts = self._click_counts(board, allowed)
        actions = [
            CompiledAction(6, {"x": node.position[0] * board.scale, "y": node.position[1] * board.scale})
            for node, count in zip(board.nodes, counts)
            for _ in range(count)
        ]
        if len(actions) > self.action_budget:
            return []
        return actions

    def parse_frame(self, frame: Any) -> Board:
        display = self._grid(frame)
        scale = max(1, min(len(display), len(display[0])) // self.size)
        source = [[display[y * scale][x * scale] for x in range(self.size)] for y in range(self.size)]
        nodes = self._nodes(source)
        clues = self._clues(source)
        palette = self._palette(source, nodes, clues)
        nodes, clues = self._drop_examples(nodes, clues, palette)
        if not nodes or len(palette) < 2:
            raise ValueError("not an FT09 puzzle frame")
        return Board(
            tuple(sorted(nodes, key=lambda node: (node.position[1], node.position[0]))),
            tuple(sorted(clues, key=lambda clue: (clue.position[1], clue.position[0]))),
            palette,
            scale,
        )

    def _nodes(self, grid: Grid) -> list[Node]:
        found: dict[Position, Node] = {}
        for y in range(self.size - 2):
            for x in range(self.size - 2):
                block = [row[x : x + 3] for row in grid[y : y + 3]]
                center = block[1][1]
                if center in self.structural:
                    continue
                values = {value for row in block for value in row}
                if values == {center} and self._isolated(grid, x, y, center):
                    found[(x, y)] = Node((x, y), center)
                elif 6 in values and values <= {6, center}:
                    effects = ((0, 0),) + tuple(
                        ((column - 1) * 4, (row - 1) * 4)
                        for row in range(3)
                        for column in range(3)
                        if block[row][column] == 6
                    )
                    found[(x, y)] = Node((x, y), center, effects)
        return list(found.values())

    def _clues(self, grid: Grid) -> list[Clue]:
        found: dict[Position, Clue] = {}
        for y in range(self.size - 2):
            for x in range(self.size - 2):
                block = [row[x : x + 3] for row in grid[y : y + 3]]
                center = block[1][1]
                surrounding = {
                    block[row][column]
                    for row in range(3)
                    for column in range(3)
                    if (row, column) != (1, 1)
                }
                if center not in self.structural and surrounding <= {0, 2, 3}:
                    found[(x, y)] = Clue((x, y), center, tuple(tuple(row) for row in block))
        return list(found.values())

    def _palette(self, grid: Grid, nodes: Iterable[Node], clues: Iterable[Clue]) -> tuple[int, ...]:
        swatches: list[tuple[int, int, int]] = []
        for color, component in self._components(grid):
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            if (
                color not in self.structural
                and len(component) == 4
                and max(xs) - min(xs) == 1
                and max(ys) - min(ys) == 1
                and min(xs) >= 27
                and max(ys) < 8
            ):
                swatches.append((min(ys), min(xs), color))
        if swatches:
            return tuple(color for _, _, color in sorted(swatches))
        ordered: list[int] = []
        for color in [node.color for node in nodes] + [clue.color for clue in clues]:
            if color not in ordered:
                ordered.append(color)
        return tuple(ordered)

    def _drop_examples(
        self, nodes: list[Node], clues: list[Clue], palette: tuple[int, ...]
    ) -> tuple[list[Node], list[Clue]]:
        if not palette or all(node.color == palette[0] for node in nodes):
            return nodes, clues
        by_position = {node.position: node for node in nodes}
        active_clues: list[Clue] = []
        active_nodes: set[Position] = set()
        for clue in clues:
            adjacent = {
                (clue.position[0] + (column - 1) * 4, clue.position[1] + (row - 1) * 4)
                for row in range(3)
                for column in range(3)
                if (row, column) != (1, 1)
            }
            present = [by_position[position] for position in adjacent if position in by_position]
            if present and all(node.color == palette[0] for node in present):
                active_clues.append(clue)
                active_nodes.update(node.position for node in present)
        if not active_clues:
            return nodes, clues
        return [node for node in nodes if node.position in active_nodes], active_clues

    def _allowed(self, board: Board) -> dict[Position, set[int]]:
        indices = {color: index for index, color in enumerate(board.palette)}
        nodes = {node.position: node for node in board.nodes}
        allowed = {position: set(range(len(board.palette))) for position in nodes}
        for clue in board.clues:
            if clue.color not in indices:
                continue
            clue_index = indices[clue.color]
            for row in range(3):
                for column in range(3):
                    if (row, column) == (1, 1):
                        continue
                    position = (
                        clue.position[0] + (column - 1) * 4,
                        clue.position[1] + (row - 1) * 4,
                    )
                    if position not in nodes:
                        continue
                    choices = {clue_index} if clue.pattern[row][column] == 0 else set(range(len(board.palette))) - {clue_index}
                    allowed[position] &= choices
        if any(not choices for choices in allowed.values()):
            raise ValueError("inconsistent FT09 constraints")
        return allowed

    def _click_counts(self, board: Board, allowed: dict[Position, set[int]]) -> list[int]:
        modulus = len(board.palette)
        indices = {color: index for index, color in enumerate(board.palette)}
        positions = {node.position: index for index, node in enumerate(board.nodes)}
        effects = [[0 for _ in board.nodes] for _ in board.nodes]
        for click, node in enumerate(board.nodes):
            for dx, dy in node.effects:
                affected = positions.get((node.position[0] + dx, node.position[1] + dy))
                if affected is not None:
                    effects[affected][click] = (effects[affected][click] + 1) % modulus
        if modulus == 2:
            equations = []
            for index, node in enumerate(board.nodes):
                if len(allowed[node.position]) == 1:
                    target = next(iter(allowed[node.position]))
                    equations.append((effects[index], (target - indices[node.color]) % 2))
            return self._binary(equations, len(board.nodes))
        if any(
            effects[row][column] != (1 if row == column else 0)
            for row in range(len(board.nodes))
            for column in range(len(board.nodes))
        ):
            return []
        return [
            min((target - indices[node.color]) % modulus for target in allowed[node.position])
            for node in board.nodes
        ]

    def _binary(self, equations: list[tuple[list[int], int]], count: int) -> list[int]:
        rows = [list(coefficients) + [rhs] for coefficients, rhs in equations]
        pivots: list[int] = []
        pivot = 0
        for column in range(count):
            candidate = next((row for row in range(pivot, len(rows)) if rows[row][column]), None)
            if candidate is None:
                continue
            rows[pivot], rows[candidate] = rows[candidate], rows[pivot]
            for row in range(len(rows)):
                if row != pivot and rows[row][column]:
                    rows[row] = [left ^ right for left, right in zip(rows[row], rows[pivot])]
            pivots.append(column)
            pivot += 1
            if pivot == len(rows):
                break
        if any(not any(row[:count]) and row[-1] for row in rows):
            return []
        solution = [0] * count
        for row, column in enumerate(pivots):
            solution[column] = rows[row][-1]
        return solution

    def _components(self, grid: Grid) -> list[tuple[int, set[Position]]]:
        seen: set[Position] = set()
        result: list[tuple[int, set[Position]]] = []
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                if (x, y) in seen:
                    continue
                pending = [(x, y)]
                seen.add((x, y))
                component: set[Position] = set()
                while pending:
                    px, py = pending.pop()
                    component.add((px, py))
                    for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                        if not (0 <= nx < len(row) and 0 <= ny < len(grid)):
                            continue
                        if (nx, ny) not in seen and grid[ny][nx] == color:
                            seen.add((nx, ny))
                            pending.append((nx, ny))
                result.append((color, component))
        return result

    def _isolated(self, grid: Grid, x: int, y: int, color: int) -> bool:
        border = (
            [(column, y - 1) for column in range(x, x + 3)]
            + [(column, y + 3) for column in range(x, x + 3)]
            + [(x - 1, row) for row in range(y, y + 3)]
            + [(x + 3, row) for row in range(y, y + 3)]
        )
        return all(
            not (0 <= bx < self.size and 0 <= by < self.size) or grid[by][bx] != color
            for bx, by in border
        )

    def _grid(self, frame: Any) -> Grid:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        if frame and isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
            frame = frame[-1]
        return [[int(value) for value in row] for row in frame]


class FT09KagglePolicy:
    def __init__(self) -> None:
        self.planner = FT09CellPlanner()
        self.plan: list[CompiledAction] = []
        self.cursor = 0
        self.plan_level: int | None = None

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        if self.plan_level != levels_completed or self.cursor >= len(self.plan):
            self.plan = self.planner.compile_from_frame(frame)
            self.cursor = 0
            self.plan_level = levels_completed
        if not self.plan:
            return {"id": 6, "data": {"x": 0, "y": 0}}
        action = self.plan[self.cursor]
        self.cursor += 1
        return {"id": action.id, "data": dict(action.data)}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        action = self.choose_action_dict(frame, levels_completed)
        return CompiledAction(action["id"], action["data"]).to_game_action()


class _FT09SubmitMixin:
    MAX_ACTIONS = 100

    def _ensure_policy(self) -> None:
        if not hasattr(self, "policy"):
            self.policy = FT09KagglePolicy()

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return getattr(getattr(latest_frame, "state", None), "name", None) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        from arcengine import GameAction, GameState

        self._ensure_policy()
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or not latest_frame.frame:
            return GameAction.RESET
        action = self.policy.choose_game_action(latest_frame.frame[-1], latest_frame.levels_completed)
        action.reasoning = {"agent": "ft09_submit", "scope": "ft09_only"}
        return action


try:
    from agents.agent import Agent

    class FT09SubmitAgent(_FT09SubmitMixin, Agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.policy = FT09KagglePolicy()

except Exception:

    class FT09SubmitAgent(_FT09SubmitMixin):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.policy = FT09KagglePolicy()
