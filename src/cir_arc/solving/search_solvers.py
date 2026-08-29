from __future__ import annotations

import heapq
from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np

from cir_arc.environment.actions import Action, ActionType, DIRECTION_VECTORS
from cir_arc.environment.frame import FrameData, MultiLayerGrid


class AStarSolver:
    """A* pathfinding solver finding optimal action sequences over discrete grid states."""

    @staticmethod
    def solve_grid_path(
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        barrier_colors: Optional[Set[int]] = None,
    ) -> List[Action]:
        barriers = barrier_colors or {5, 8}  # Walls and locked doors
        h, w = grid.shape

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority queue: (f_score, g_score, current_pos, path_actions)
        open_set: List[Tuple[int, int, Tuple[int, int], List[Action]]] = []
        heapq.heappush(open_set, (heuristic(start_pos, goal_pos), 0, start_pos, []))
        visited: Set[Tuple[int, int]] = {start_pos}

        while open_set:
            f, g, curr, actions = heapq.heappop(open_set)

            if curr == goal_pos:
                return actions

            for aid, (dr, dc) in DIRECTION_VECTORS.items():
                nr, nc = curr[0] + dr, curr[1] + dc
                next_pos = (nr, nc)

                if 0 <= nr < h and 0 <= nc < w:
                    if next_pos not in visited and (grid[nr, nc] not in barriers or next_pos == goal_pos):
                        visited.add(next_pos)
                        new_actions = actions + [Action(ActionType(aid))]
                        new_g = g + 1
                        new_f = new_g + heuristic(next_pos, goal_pos)
                        heapq.heappush(open_set, (new_f, new_g, next_pos, new_actions))

        return []


class BFSSolver:
    """Breadth-First Search solver for shortest-path discrete grid navigation."""

    @staticmethod
    def solve(
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        barrier_colors: Optional[Set[int]] = None,
    ) -> List[Action]:
        barriers = barrier_colors or {5, 8}
        h, w = grid.shape

        queue: deque[Tuple[Tuple[int, int], List[Action]]] = deque([(start_pos, [])])
        visited: Set[Tuple[int, int]] = {start_pos}

        while queue:
            curr, actions = queue.popleft()

            if curr == goal_pos:
                return actions

            for aid, (dr, dc) in DIRECTION_VECTORS.items():
                nr, nc = curr[0] + dr, curr[1] + dc
                next_pos = (nr, nc)

                if 0 <= nr < h and 0 <= nc < w:
                    if next_pos not in visited and (grid[nr, nc] not in barriers or next_pos == goal_pos):
                        visited.add(next_pos)
                        queue.append((next_pos, actions + [Action(ActionType(aid))]))

        return []
