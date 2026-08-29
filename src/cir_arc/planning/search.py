"""A* search pathfinding algorithm for discrete grid navigation."""

from __future__ import annotations

import heapq
from typing import List, Optional, Set, Tuple
import numpy as np

from cir_arc.environment.actions import DIRECTION_VECTORS, Action, ActionType


class AStarGridPlanner:
    """Computes shortest collision-free paths on 2D discrete grids."""

    @staticmethod
    def find_path(
        passable_mask: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Finds coordinate path from start to goal on boolean 2D grid using A*."""
        H, W = passable_mask.shape
        if not (0 <= start[0] < H and 0 <= start[1] < W):
            return []
        if not (0 <= goal[0] < H and 0 <= goal[1] < W):
            return []

        # If start is already goal
        if start == goal:
            return [start]

        # Priority queue entries: (f_score, h_score, (r, c))
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (heuristic(start, goal), 0.0, start))

        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], float] = {start: 0.0}

        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            cr, cc = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                if not (0 <= nr < H and 0 <= nc < W):
                    continue

                # Cell must be passable, unless it is the goal itself
                if not passable_mask[nr, nc] and neighbor != goal:
                    continue

                tentative_g = current_g + 1.0
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        return []

    @staticmethod
    def path_to_actions(path: List[Tuple[int, int]]) -> List[Action]:
        """Converts coordinate path [(r0, c0), (r1, c1), ...] to directional Action list."""
        if len(path) < 2:
            return []

        actions: List[Action] = []
        dir_to_action = {
            (-1, 0): Action(ActionType.ACTION1),  # UP
            (1, 0): Action(ActionType.ACTION2),   # DOWN
            (0, -1): Action(ActionType.ACTION3),  # LEFT
            (0, 1): Action(ActionType.ACTION4),   # RIGHT
        }

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            delta = (r2 - r1, c2 - c1)
            if delta in dir_to_action:
                actions.append(dir_to_action[delta])
        return actions
