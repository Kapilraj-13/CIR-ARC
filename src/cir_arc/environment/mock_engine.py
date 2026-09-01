from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cir_arc.environment.actions import DIRECTION_VECTORS, Action, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData, GameState, MultiLayerGrid


class MockEngine(BaseEnvironment):
    """Deterministic simulated ARC-AGI-3 environment for testing and offline evaluation."""

    def __init__(self, game_id: str = "mock_maze_01", win_levels: int = 1) -> None:
        super().__init__(game_id)
        self.win_levels = win_levels
        self.step_count = 0
        self.levels_completed = 0
        self.state = GameState.NOT_PLAYED
        self._current_frame: Optional[FrameData] = None

        # Game state internals
        self.player_pos = [1, 1]
        self.player_color = 9  # Blue
        self.goal_pos = [5, 5]
        self.key_pos = [2, 4]
        self.door_pos = [4, 4]
        self.has_key = False
        self.door_open = False
        self.grid_size = (8, 8)
        self.walls: set[Tuple[int, int]] = set()

        self._init_game_layout()

    def _init_game_layout(self) -> None:
        """Setup initial game layout based on game_id."""
        self.walls = set()
        if self.game_id == "mock_maze_02":
            self.grid_size = (9, 9)
            self.player_pos = [1, 1]
            self.goal_pos = [7, 7]
            for r in range(9):
                self.walls.add((r, 0))
                self.walls.add((r, 8))
            for c in range(9):
                self.walls.add((0, c))
                self.walls.add((8, c))
            # S-curve internal walls
            for c in range(1, 6):
                self.walls.add((3, c))
            for c in range(3, 8):
                self.walls.add((6, c))
        elif "maze" in self.game_id:
            self.grid_size = (8, 8)
            self.player_pos = [1, 1]
            self.goal_pos = [6, 6]
            # Outer boundary walls (color 5: black/wall)
            for r in range(8):
                self.walls.add((r, 0))
                self.walls.add((r, 7))
            for c in range(8):
                self.walls.add((0, c))
                self.walls.add((7, c))
            # Internal obstacle
            self.walls.add((2, 2))
            self.walls.add((3, 2))
            self.walls.add((4, 2))
            self.walls.add((2, 4))
            self.walls.add((3, 4))
            self.walls.add((4, 4))
        elif self.game_id == "mock_locksmith_02":
            self.grid_size = (9, 9)
            self.player_pos = [1, 1]
            self.key_pos = [1, 7]
            self.door_pos = [4, 4]
            self.goal_pos = [7, 7]
            self.has_key = False
            self.door_open = False
            for r in range(9):
                self.walls.add((r, 0))
                self.walls.add((r, 8))
                self.walls.add((4, r))
            for c in range(9):
                self.walls.add((0, c))
                self.walls.add((8, c))
            self.walls.remove((4, 4))
        elif "locksmith" in self.game_id or "ls20" in self.game_id:
            self.grid_size = (10, 10)
            self.player_pos = [1, 1]
            self.key_pos = [1, 8]
            self.door_pos = [5, 5]
            self.goal_pos = [8, 8]
            self.has_key = False
            self.door_open = False
            for r in range(10):
                self.walls.add((r, 0))
                self.walls.add((r, 9))
                self.walls.add((5, r))
            for c in range(10):
                self.walls.add((0, c))
                self.walls.add((9, c))
            # Door is in wall dividing room
            self.walls.remove((5, 5))
        else:
            self.grid_size = (8, 8)
            self.player_pos = [1, 1]
            self.goal_pos = [5, 5]

    def _render_grid(self) -> MultiLayerGrid:
        """Render multi-layer grid: Layer 0 background/walls/doors, Layer 1 agent/items."""
        h, w = self.grid_size
        layer0 = np.zeros((h, w), dtype=np.int16)  # Background/world
        layer1 = np.zeros((h, w), dtype=np.int16)  # Entities/agent

        # Draw walls (color 4/5)
        for r, c in self.walls:
            layer0[r, c] = 5

        # Draw goal (color 14: Green)
        if self.goal_pos:
            layer0[self.goal_pos[0], self.goal_pos[1]] = 14

        if "locksmith" in self.game_id or "ls20" in self.game_id:
            # Draw door (color 8: Red if locked, 0 if open)
            if not self.door_open:
                layer0[self.door_pos[0], self.door_pos[1]] = 8
            # Draw key (color 11: Yellow)
            if not self.has_key:
                layer1[self.key_pos[0], self.key_pos[1]] = 11

        # Draw player (color 9: Blue)
        layer1[self.player_pos[0], self.player_pos[1]] = self.player_color

        return MultiLayerGrid([layer0, layer1])

    def reset(self) -> FrameData:
        self.step_count = 0
        self.levels_completed = 0
        self.state = GameState.NOT_FINISHED
        self._init_game_layout()
        grid = self._render_grid()
        self._current_frame = FrameData(
            game_id=self.game_id,
            grid=grid,
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            action_input=Action(ActionType.RESET),
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            full_reset=True,
            step_count=self.step_count,
        )
        return self._current_frame

    def step(self, action: Action) -> FrameData:
        self.step_count += 1

        if action.action_type == ActionType.RESET:
            return self.reset()

        if self.state in (GameState.WIN, GameState.GAME_OVER):
            return self._current_frame or self.reset()

        prev_pos = list(self.player_pos)

        # Handle movement actions
        if action.action_id in DIRECTION_VECTORS:
            dr, dc = DIRECTION_VECTORS[action.action_id]
            nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
            h, w = self.grid_size

            # Check boundaries & walls
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in self.walls:
                # Check door in locksmith games
                if "locksmith" in self.game_id or "ls20" in self.game_id:
                    if [nr, nc] == self.door_pos and not self.door_open:
                        pass  # Blocked by door
                    else:
                        self.player_pos = [nr, nc]
                else:
                    self.player_pos = [nr, nc]

        # Handle interact (Action 5)
        elif action.action_type == ActionType.ACTION5:
            # Check key pickup
            if ("locksmith" in self.game_id or "ls20" in self.game_id) and not self.has_key:
                if self.player_pos == self.key_pos or abs(self.player_pos[0] - self.key_pos[0]) + abs(self.player_pos[1] - self.key_pos[1]) <= 1:
                    self.has_key = True
            # Check unlock door
            if ("locksmith" in self.game_id or "ls20" in self.game_id) and self.has_key and not self.door_open:
                if abs(self.player_pos[0] - self.door_pos[0]) + abs(self.player_pos[1] - self.door_pos[1]) <= 1:
                    self.door_open = True

        # Handle click (Action 6)
        elif action.action_type == ActionType.ACTION6:
            cx = action.data.get("x", 0)
            cy = action.data.get("y", 0)
            # Click near door with key unlocks door
            if ("locksmith" in self.game_id or "ls20" in self.game_id) and self.has_key:
                if abs(cy - self.door_pos[0]) <= 1 and abs(cx - self.door_pos[1]) <= 1:
                    self.door_open = True

        # Check win condition
        if self.player_pos == self.goal_pos:
            self.levels_completed += 1
            if self.levels_completed >= self.win_levels:
                self.state = GameState.WIN
            else:
                self._init_game_layout()

        # Check step limit
        if self.step_count >= 100 and self.state != GameState.WIN:
            self.state = GameState.GAME_OVER

        grid = self._render_grid()
        self._current_frame = FrameData(
            game_id=self.game_id,
            grid=grid,
            state=self.state,
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            action_input=action,
            available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
            step_count=self.step_count,
        )
        return self._current_frame

    def current_observation(self) -> Optional[FrameData]:
        return self._current_frame
