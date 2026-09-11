from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


Grid = list[list[int]]


@dataclass(frozen=True)
class Instruction:
    kind: str
    color: int


@dataclass(frozen=True)
class Token:
    instruction: Instruction
    x: int
    y: int


@dataclass(frozen=True)
class Slot:
    frame_color: int
    index: int
    x: int
    y: int
    instruction: Instruction | None


@dataclass(frozen=True)
class Procedure:
    color: int
    slots: tuple[Slot, ...]
    x: int
    y: int


@dataclass(frozen=True)
class Puzzle:
    target: tuple[int, ...]
    procedures: tuple[Procedure, ...]
    tokens: tuple[Token, ...]


@dataclass(frozen=True)
class CompiledAction:
    id: int
    data: dict[str, int]

    def to_game_action(self) -> Any:
        from arcengine import GameAction

        action = GameAction.from_id(self.id)
        if self.data:
            action.set_data(dict(self.data))
        return action


class SB26CellPlanner:
    background = 4
    colors = frozenset({6, 8, 9, 11, 12, 14, 15})
    action_budget = 64

    def compile_from_frame(self, frame: Any) -> list[CompiledAction]:
        puzzle = self.parse_frame(frame)
        empty = tuple(
            slot
            for procedure in puzzle.procedures
            for slot in procedure.slots
            if slot.instruction is None
        )
        assignment = self._assignment(puzzle, empty)
        if assignment is None:
            return []
        used: set[int] = set()
        actions: list[CompiledAction] = []
        for slot, instruction in zip(empty, assignment):
            token_index = next(
                index
                for index, token in enumerate(puzzle.tokens)
                if index not in used and token.instruction == instruction
            )
            used.add(token_index)
            token = puzzle.tokens[token_index]
            actions.append(CompiledAction(6, {"x": token.x + 1, "y": token.y + 1}))
            actions.append(CompiledAction(6, {"x": slot.x + 1, "y": slot.y + 1}))
        actions.append(CompiledAction(5, {}))
        return actions if len(actions) <= self.action_budget else []

    def parse_frame(self, frame: Any) -> Puzzle:
        grid = self._grid(frame)
        detected, targets = self._scan(grid)
        middle = sorted((item for item in detected if item[1] < 53), key=lambda item: (item[1], item[0]))
        tokens = tuple(
            Token(item[2], item[0], item[1])
            for item in sorted(
                (item for item in detected if item[1] > 53 and item[2] is not None),
                key=lambda item: item[0],
            )
        )
        procedures = self._procedures(grid, middle)
        target = tuple(color for _, _, color in sorted(targets, key=lambda item: (item[1], item[0])))
        if not target or not procedures or not tokens:
            raise ValueError("not an SB26 puzzle frame")
        if sum(slot.instruction is None for procedure in procedures for slot in procedure.slots) != len(tokens):
            raise ValueError("SB26 slots and movable tokens do not match")
        return Puzzle(target, procedures, tokens)

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        try:
            self.parse_frame(frame)
            return True
        except Exception:
            return False

    def _scan(
        self,
        grid: Grid,
    ) -> tuple[list[tuple[int, int, Instruction | None]], list[tuple[int, int, int]]]:
        found: list[tuple[int, int, Instruction | None]] = []
        targets: list[tuple[int, int, int]] = []
        for y in range(len(grid) - 5):
            for x in range(len(grid[0]) - 5):
                window = [row[x : x + 6] for row in grid[y : y + 6]]
                outer = window[0] + window[5]
                outer += [window[row][0] for row in range(1, 5)]
                outer += [window[row][5] for row in range(1, 5)]
                inner = [row[1:5] for row in window[1:5]]
                values = {value for row in inner for value in row}

                if all(value == self.background for value in outer) and len(values) == 1:
                    color = next(iter(values))
                    if color in self.colors:
                        found.append((x, y, Instruction("literal", color)))
                        continue

                color = inner[0][0]
                if (
                    color in self.colors
                    and set(outer) <= {self.background, color}
                    and all(
                        inner[row][column] == (self.background if 1 <= row <= 2 and 1 <= column <= 2 else color)
                        for row in range(4)
                        for column in range(4)
                    )
                ):
                    found.append((x, y, Instruction("call", color)))
                    continue

                if all(value == self.background for value in outer) and all(
                    inner[row][column] == (2 if 1 <= row <= 2 and 1 <= column <= 2 else self.background)
                    for row in range(4)
                    for column in range(4)
                ):
                    found.append((x, y, None))

                for target_color in self.colors:
                    if (
                        all(window[0][offset] == target_color and window[5][offset] == target_color for offset in range(6))
                        and all(
                            window[offset][0] == target_color and window[offset][5] == target_color
                            for offset in range(1, 5)
                        )
                        and all(inner[row][column] == 5 for row in range(4) for column in range(4))
                    ):
                        targets.append((x, y, target_color))
        return found, targets

    def _procedures(
        self,
        grid: Grid,
        middle: list[tuple[int, int, Instruction | None]],
    ) -> tuple[Procedure, ...]:
        groups: list[list[tuple[int, int, Instruction | None]]] = []
        for y in sorted({item[1] for item in middle}):
            row = sorted((item for item in middle if item[1] == y), key=lambda item: item[0])
            current: list[tuple[int, int, Instruction | None]] = []
            for item in row:
                if current and item[0] - current[-1][0] != 6:
                    groups.append(current)
                    current = []
                current.append(item)
            if current:
                groups.append(current)

        procedures = []
        for group in groups:
            frame_x = group[0][0] - 2
            frame_y = group[0][1] - 2
            color = int(grid[frame_y][frame_x])
            slots = tuple(
                Slot(color, index, x, y, instruction)
                for index, (x, y, instruction) in enumerate(group)
            )
            procedures.append(Procedure(color, slots, frame_x, frame_y))
        return tuple(sorted(procedures, key=lambda procedure: (procedure.y, procedure.x)))

    def _assignment(
        self,
        puzzle: Puzzle,
        empty: tuple[Slot, ...],
    ) -> tuple[Instruction, ...] | None:
        programs = {
            procedure.color: [slot.instruction for slot in procedure.slots]
            for procedure in puzzle.procedures
        }
        values = tuple(token.instruction for token in puzzle.tokens)
        for permutation in self._permutations(values):
            for slot, instruction in zip(empty, permutation):
                programs[slot.frame_color][slot.index] = instruction
            if self._matches(programs, puzzle.procedures[0].color, puzzle.target):
                return permutation
        return None

    def _matches(
        self,
        programs: dict[int, list[Instruction | None]],
        root: int,
        target: tuple[int, ...],
    ) -> bool:
        stack: list[tuple[int, int]] = [(root, 0)]
        output = 0
        for _ in range(max(128, len(target) * 16)):
            if not stack:
                return False
            frame_color, position = stack[-1]
            instruction = programs[frame_color][position]
            if instruction is None:
                return False
            if instruction.kind == "literal":
                if instruction.color != target[output]:
                    return False
                output += 1
                if output == len(target):
                    return True
                while stack:
                    caller, caller_position = stack[-1]
                    if caller_position + 1 < len(programs[caller]):
                        stack[-1] = (caller, caller_position + 1)
                        break
                    stack.pop()
            else:
                if instruction.color not in programs:
                    return False
                if (
                    position == 0
                    and (frame_color, position) in stack[:-1]
                    and len(stack) > 1
                    and stack[-2][1] == 0
                ):
                    return False
                stack.append((instruction.color, 0))
        return False

    def _permutations(self, values: tuple[Instruction, ...]) -> Iterable[tuple[Instruction, ...]]:
        counts: dict[Instruction, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        order = tuple(counts)
        current: list[Instruction] = []

        def generate() -> Iterable[tuple[Instruction, ...]]:
            if len(current) == len(values):
                yield tuple(current)
                return
            for value in order:
                if counts[value] == 0:
                    continue
                counts[value] -= 1
                current.append(value)
                yield from generate()
                current.pop()
                counts[value] += 1

        yield from generate()

    def _grid(self, frame: Any) -> Grid:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        if frame and isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
            frame = frame[-1]
        if not frame or not frame[0]:
            raise ValueError("frame is empty")
        return [[int(value) for value in row] for row in frame]


class SB26KagglePolicy:
    def __init__(self) -> None:
        self.planner = SB26CellPlanner()
        self.plan: list[CompiledAction] = []
        self.cursor = 0
        self.plan_level: int | None = None

    def supports(self, frame: Any, levels_completed: int = 0) -> bool:
        return self.planner.supports(frame, levels_completed)

    def choose_action_dict(self, frame: Any, levels_completed: int = 0) -> dict[str, Any]:
        if self.plan_level != levels_completed or self.cursor >= len(self.plan):
            self.plan = self.planner.compile_from_frame(frame)
            self.cursor = 0
            self.plan_level = levels_completed
        if not self.plan:
            return {"id": 5, "data": {}}
        action = self.plan[self.cursor]
        self.cursor += 1
        return {"id": action.id, "data": dict(action.data)}

    def choose_game_action(self, frame: Any, levels_completed: int = 0) -> Any:
        action = self.choose_action_dict(frame, levels_completed)
        return CompiledAction(action["id"], action["data"]).to_game_action()


class _SB26SubmitMixin:
    MAX_ACTIONS = 1000

    def _ensure_policy(self) -> None:
        if not hasattr(self, "policy"):
            self.policy = SB26KagglePolicy()

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return getattr(getattr(latest_frame, "state", None), "name", None) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        from arcengine import GameAction, GameState

        self._ensure_policy()
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or not latest_frame.frame:
            return GameAction.RESET
        action = self.policy.choose_game_action(latest_frame.frame[-1], latest_frame.levels_completed)
        action.reasoning = {"agent": "sb26_submit", "scope": "sb26_only"}
        return action


try:
    from agents.agent import Agent

    class SB26SubmitAgent(_SB26SubmitMixin, Agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.policy = SB26KagglePolicy()

except Exception:

    class SB26SubmitAgent(_SB26SubmitMixin):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.policy = SB26KagglePolicy()
