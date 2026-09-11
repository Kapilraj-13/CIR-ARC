from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


Grid = list[list[int]]


@dataclass(frozen=True)
class Symbol:
    family: str
    index: int

    def shifted(self, delta: int) -> "Symbol":
        return Symbol(self.family, (self.index - 1 + delta) % 7 + 1)


@dataclass(frozen=True)
class Rule:
    left: tuple[Symbol, ...]
    right: tuple[Symbol, ...]


@dataclass(frozen=True)
class Puzzle:
    rules: tuple[Rule, ...]
    source: tuple[Symbol, ...]
    target: tuple[Symbol, ...]
    mode: str
    action_budget: int


class TR87CellPlanner:
    family_colors = {10: "A", 7: "B", 11: "C"}
    templates = {
        "A": (
            "00100/00100/01110/00100/11111",
            "00001/00101/11111/10100/10000",
            "11111/10100/10000/10100/11111",
            "00100/11111/10101/10101/00100",
            "10001/11111/00100/11111/10001",
            "11111/00100/11100/10000/11111",
            "10001/11111/10001/10001/11011",
        ),
        "B": (
            "10000/11110/10010/10010/11111",
            "11111/10001/11101/10101/11111",
            "00111/00101/11111/10100/11100",
            "01110/01010/11111/10001/11111",
            "11111/10001/10001/11111/00100",
            "11110/10011/10001/11001/01111",
            "00100/11111/10101/11111/00100",
        ),
        "C": (
            "10101/10000/10101/10000/11111",
            "11101/10000/00000/00001/10111",
            "10111/00100/00100/00100/11101",
            "00100/10101/00100/10101/00100",
            "11011/10001/10101/10001/11011",
            "10111/10101/11101/00000/10101",
            "10001/00000/10001/11011/01010",
        ),
    }

    def __init__(self) -> None:
        self.patterns: dict[tuple[str, tuple[tuple[int, ...], ...]], int] = {}
        for family, templates in self.templates.items():
            for index, template in enumerate(templates, 1):
                pattern = tuple(tuple(int(cell) for cell in row) for row in template.split("/"))
                for rotation in self._rotations(pattern):
                    self.patterns[(family, rotation)] = index

    def compile_from_frame(self, frame: Any, levels_completed: int = 0) -> list[int]:
        puzzle = self.parse_frame(frame, levels_completed)
        if puzzle.mode in {"standard", "double"}:
            desired = self._translation(puzzle)
            if desired is None or len(desired) != len(puzzle.target):
                return []
            deltas = tuple((desired.index - current.index) % 7 for current, desired in zip(puzzle.target, desired))
        else:
            shifts = self._rule_shifts(puzzle, puzzle.mode == "tree_alter")
            if shifts is None:
                return []
            deltas = tuple(shift or 0 for shift in shifts)
        actions = self._actions(deltas)
        return actions if len(actions) <= puzzle.action_budget else []

    def parse_frame(self, frame: Any, levels_completed: int = 0) -> Puzzle:
        grid = self._grid(frame)
        tiles = self._tiles(grid)
        rows = sorted({y for x, y, symbol in tiles})
        if len(rows) < 3:
            raise ValueError("not a TR87 puzzle frame")
        source_y, target_y = rows[-2:]
        source = tuple(symbol for x, y, symbol in sorted((tile for tile in tiles if tile[1] == source_y)))
        target = tuple(symbol for x, y, symbol in sorted((tile for tile in tiles if tile[1] == target_y)))
        rules = self._rules([tile for tile in tiles if tile[1] < source_y])
        mode = "double" if levels_completed == 3 else "alter" if levels_completed == 4 else "tree_alter" if levels_completed >= 5 else "standard"
        if not rules or not source or not target:
            raise ValueError("TR87 frame is incomplete")
        return Puzzle(rules, source, target, mode, 256 if levels_completed >= 5 else 128)

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        try:
            self.parse_frame(frame, levels_completed)
            return True
        except Exception:
            return False

    def _tiles(self, grid: Grid) -> list[tuple[int, int, Symbol]]:
        found: dict[tuple[int, int], tuple[int, int, Symbol]] = {}
        for y in range(len(grid) - 6):
            for x in range(len(grid[0]) - 6):
                for color, family in self.family_colors.items():
                    border = [grid[y][x + offset] for offset in range(7)]
                    border += [grid[y + 6][x + offset] for offset in range(7)]
                    border += [grid[y + offset][x] for offset in range(1, 6)]
                    border += [grid[y + offset][x + 6] for offset in range(1, 6)]
                    if not all(value == color for value in border):
                        continue
                    pattern = tuple(
                        tuple(1 if grid[y + row + 1][x + column + 1] == 5 else 0 for column in range(5))
                        for row in range(5)
                    )
                    index = self.patterns.get((family, pattern))
                    if index is not None:
                        found[(x, y)] = (x, y, Symbol(family, index))
        return sorted(found.values(), key=lambda tile: (tile[1], tile[0]))

    def _rules(self, tiles: list[tuple[int, int, Symbol]]) -> tuple[Rule, ...]:
        sides: list[tuple[Symbol, ...]] = []
        for y in sorted({tile[1] for tile in tiles}):
            row = sorted((tile for tile in tiles if tile[1] == y), key=lambda tile: tile[0])
            current: list[tuple[int, int, Symbol]] = []
            for tile in row:
                if current and (tile[2].family != current[-1][2].family or tile[0] - current[-1][0] > 7):
                    sides.append(tuple(item[2] for item in current))
                    current = []
                current.append(tile)
            if current:
                sides.append(tuple(item[2] for item in current))
        if len(sides) % 2:
            raise ValueError("unpaired TR87 rule sides")
        return tuple(Rule(sides[index], sides[index + 1]) for index in range(0, len(sides), 2))

    def _translation(self, puzzle: Puzzle) -> tuple[Symbol, ...] | None:
        if puzzle.mode == "double":
            mappings = [
                (first.left, second.right)
                for first in puzzle.rules
                for second in puzzle.rules
                if first.right == second.left
            ]
        else:
            mappings = [(rule.left, rule.right) for rule in puzzle.rules]
        memo: dict[int, tuple[Symbol, ...] | None] = {}

        def translate(position: int) -> tuple[Symbol, ...] | None:
            if position == len(puzzle.source):
                return ()
            if position in memo:
                return memo[position]
            for left, right in mappings:
                if puzzle.source[position : position + len(left)] == left:
                    suffix = translate(position + len(left))
                    if suffix is not None:
                        memo[position] = right + suffix
                        return memo[position]
            memo[position] = None
            return None

        return translate(0)

    def _rule_shifts(self, puzzle: Puzzle, tree: bool) -> tuple[int | None, ...] | None:
        rules = puzzle.rules

        def bind(pattern: Sequence[Symbol], segment: Sequence[Symbol], shifts: tuple[int | None, ...], index: int):
            delta = self._shift(pattern, segment, shifts[index])
            if delta is None:
                return None
            updated = list(shifts)
            updated[index] = delta
            return tuple(updated)

        def expand(symbols: Sequence[Symbol], target_position: int, shifts: tuple[int | None, ...]):
            if not symbols:
                return [(target_position, shifts)]
            outcomes = []
            for index, rule in enumerate(rules):
                if not rule.left or rule.left[0].family != symbols[0].family:
                    continue
                left = bind(rule.left[:1], symbols[:1], shifts, index * 2)
                end = target_position + len(rule.right)
                if left is None or end > len(puzzle.target):
                    continue
                right = bind(rule.right, puzzle.target[target_position:end], left, index * 2 + 1)
                if right is not None:
                    outcomes.extend(expand(symbols[1:], end, right))
            return outcomes

        def search(source_position: int, target_position: int, shifts: tuple[int | None, ...]):
            if source_position == len(puzzle.source) and target_position == len(puzzle.target):
                return shifts
            if source_position >= len(puzzle.source) or target_position > len(puzzle.target):
                return None
            for index, rule in enumerate(rules):
                source_end = source_position + len(rule.left)
                if source_end > len(puzzle.source):
                    continue
                left = bind(rule.left, puzzle.source[source_position:source_end], shifts, index * 2)
                if left is None:
                    continue
                if tree:
                    existing = left[index * 2 + 1]
                    candidates: Iterable[int] = range(7) if existing is None else (existing,)
                    for delta in candidates:
                        shifted = list(left)
                        shifted[index * 2 + 1] = delta
                        symbols = tuple(symbol.shifted(delta) for symbol in rule.right)
                        for next_target, expanded in expand(symbols, target_position, tuple(shifted)):
                            result = search(source_end, next_target, expanded)
                            if result is not None:
                                return result
                else:
                    target_end = target_position + len(rule.right)
                    if target_end > len(puzzle.target):
                        continue
                    right = bind(rule.right, puzzle.target[target_position:target_end], left, index * 2 + 1)
                    if right is not None:
                        result = search(source_end, target_end, right)
                        if result is not None:
                            return result
            return None

        return search(0, 0, (None,) * (len(rules) * 2))

    def _shift(self, pattern: Sequence[Symbol], segment: Sequence[Symbol], assigned: int | None) -> int | None:
        if len(pattern) != len(segment):
            return None
        deltas = []
        for source, target in zip(pattern, segment):
            if source.family != target.family:
                return None
            deltas.append((target.index - source.index) % 7)
        if not deltas or len(set(deltas)) != 1:
            return None
        return deltas[0] if assigned is None or assigned == deltas[0] else None

    def _actions(self, deltas: Sequence[int]) -> list[int]:
        actions: list[int] = []
        selector = 0
        for index, delta in enumerate(deltas):
            delta %= 7
            if not delta:
                continue
            forward = (index - selector) % len(deltas)
            backward = (selector - index) % len(deltas)
            actions.extend([4] * forward if forward <= backward else [3] * backward)
            selector = index
            actions.extend([2] * delta if delta <= 7 - delta else [1] * (7 - delta))
        return actions

    def _rotations(self, pattern: tuple[tuple[int, ...], ...]):
        rotations = []
        current = pattern
        for _ in range(4):
            rotations.append(current)
            current = tuple(tuple(current[4 - column][row] for column in range(5)) for row in range(5))
        return rotations

    def _grid(self, frame: Any) -> Grid:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        if frame and isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
            frame = frame[-1]
        if not frame or not frame[0]:
            raise ValueError("frame is empty")
        return [[int(value) for value in row] for row in frame]


class TR87KagglePolicy:
    def __init__(self) -> None:
        self.planner = TR87CellPlanner()
        self.plan: list[int] = []
        self.cursor = 0
        self.plan_level: int | None = None

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        return self.planner.supports(frame, levels_completed)

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        if self.plan_level != levels_completed or self.cursor >= len(self.plan):
            self.plan = self.planner.compile_from_frame(frame, levels_completed)
            self.cursor = 0
            self.plan_level = levels_completed
        if not self.plan:
            return {"id": 2, "data": {}}
        action_id = self.plan[self.cursor]
        self.cursor += 1
        return {"id": action_id, "data": {}}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        from arcengine import GameAction

        return GameAction.from_id(self.choose_action_dict(frame, levels_completed)["id"])


class _TR87SubmitMixin:
    MAX_ACTIONS = 1000

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return getattr(getattr(latest_frame, "state", None), "name", None) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        from arcengine import GameAction, GameState

        if not hasattr(self, "policy"):
            self.policy = TR87KagglePolicy()
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or not latest_frame.frame:
            return GameAction.RESET
        action = self.policy.choose_game_action(latest_frame.frame[-1], latest_frame.levels_completed)
        action.reasoning = {"agent": "tr87_submit", "scope": "tr87_only"}
        return action


try:
    from agents.agent import Agent

    class TR87SubmitAgent(_TR87SubmitMixin, Agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.policy = TR87KagglePolicy()

except Exception:

    class TR87SubmitAgent(_TR87SubmitMixin):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.policy = TR87KagglePolicy()
