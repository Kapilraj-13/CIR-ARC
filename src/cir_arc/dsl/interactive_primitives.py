from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from cir_arc.environment.actions import Action, ActionType, DIRECTION_VECTORS


def step_translate(
    grid: np.ndarray,
    obj_color: int,
    dr: int,
    dc: int,
    barrier_colors: Optional[Set[int]] = None,
) -> np.ndarray:
    """Translate all pixels of obj_color by (dr, dc) if destination is not in barrier_colors."""
    out = grid.copy()
    barriers = barrier_colors or {5}
    locs = np.argwhere(grid == obj_color)
    if len(locs) == 0:
        return out

    h, w = grid.shape
    new_locs = [(r + dr, c + dc) for r, c in locs]

    # Check collision with borders or barrier colors
    for nr, nc in new_locs:
        if not (0 <= nr < h and 0 <= nc < w):
            return out  # Hit boundary
        if grid[nr, nc] in barriers:
            return out  # Hit wall

    # Apply translation
    for r, c in locs:
        out[r, c] = 0
    for nr, nc in new_locs:
        out[nr, nc] = obj_color

    return out


def step_recolor(grid: np.ndarray, src_color: int, dst_color: int) -> np.ndarray:
    """Recolor all pixels of src_color to dst_color."""
    out = grid.copy()
    out[grid == src_color] = dst_color
    return out


def macro_navigate_path(
    grid: np.ndarray,
    path: List[Tuple[int, int]],
    player_color: int = 9,
) -> Tuple[np.ndarray, List[Action]]:
    """Generate actions and simulated final grid along a waypoint path."""
    out = grid.copy()
    actions: List[Action] = []

    if not path or len(path) < 2:
        return out, actions

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dr, dc = r2 - r1, c2 - c1

        aid = None
        for k, v in DIRECTION_VECTORS.items():
            if v == (dr, dc):
                aid = k
                break

        if aid is not None:
            actions.append(Action(ActionType(aid)))
            out = step_translate(out, player_color, dr, dc)

    return out, actions
