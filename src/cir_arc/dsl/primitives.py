from __future__ import annotations
import numpy as np
from typing import Dict, Any, Callable, Tuple
from cir_arc.core.grid import Grid

# Type alias
RuleFunction = Callable[[Grid, Dict[str, Any]], Grid]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TIER 1 â€” Highest ARC frequency
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def reflect_horizontal(grid: Grid, params: Dict = {}) -> Grid:
    """Flip grid top-to-bottom."""
    return grid.reflect_horizontal()


def reflect_vertical(grid: Grid, params: Dict = {}) -> Grid:
    """Flip grid left-to-right."""
    return grid.reflect_vertical()


def reflect_diagonal(grid: Grid, params: Dict = {}) -> Grid:
    """Transpose grid (flip along main diagonal)."""
    return grid.reflect_diagonal()


def reflect_antidiagonal(grid: Grid, params: Dict = {}) -> Grid:
    """Flip along anti-diagonal."""
    return grid.reflect_antidiagonal()


def rotate_90(grid: Grid, params: Dict = {}) -> Grid:
    return grid.rotate_90()


def rotate_180(grid: Grid, params: Dict = {}) -> Grid:
    return grid.rotate_180()


def rotate_270(grid: Grid, params: Dict = {}) -> Grid:
    return grid.rotate_270()


def color_swap_all(grid: Grid, params: Dict) -> Grid:
    """
    Swap two colors everywhere in the grid.
    params: {"color_a": int, "color_b": int}
    """
    a, b = params["color_a"], params["color_b"]
    return grid.recolor({a: b, b: a})


def color_remap(grid: Grid, params: Dict) -> Grid:
    """
    Remap multiple colors at once.
    params: {"mapping": {src_color: dst_color, ...}}
    """
    return grid.recolor(params["mapping"])


def move_objects(grid: Grid, params: Dict) -> Grid:
    """
    Move all non-background objects by (dr, dc).
    params: {"dr": int, "dc": int, "background": int}
    """
    dr = params["dr"]
    dc = params["dc"]
    bg = params.get("background", 0)

    data = grid.data
    new_data = np.full_like(data, bg)
    h, w = data.shape

    for r in range(h):
        for c in range(w):
            if data[r, c] != bg:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    new_data[nr, nc] = data[r, c]
    return Grid(new_data)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TIER 2
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def gravity(grid: Grid, params: Dict) -> Grid:
    """
    Drop all non-background cells in a direction until they hit bottom/wall.
    params: {"direction": "down"|"up"|"left"|"right", "background": int}
    """
    direction = params["direction"]
    bg = params.get("background", 0)
    data = grid.data.copy()
    h, w = data.shape

    if direction == "down":
        for c in range(w):
            col = data[:, c]
            non_bg = col[col != bg]
            new_col = np.full(h, bg, dtype=np.int8)
            new_col[h - len(non_bg):] = non_bg
            data[:, c] = new_col
    elif direction == "up":
        for c in range(w):
            col = data[:, c]
            non_bg = col[col != bg]
            new_col = np.full(h, bg, dtype=np.int8)
            new_col[:len(non_bg)] = non_bg
            data[:, c] = new_col
    elif direction == "right":
        for r in range(h):
            row = data[r, :]
            non_bg = row[row != bg]
            new_row = np.full(w, bg, dtype=np.int8)
            new_row[w - len(non_bg):] = non_bg
            data[r, :] = new_row
    elif direction == "left":
        for r in range(h):
            row = data[r, :]
            non_bg = row[row != bg]
            new_row = np.full(w, bg, dtype=np.int8)
            new_row[:len(non_bg)] = non_bg
            data[r, :] = new_row

    return Grid(data)


def scale_up(grid: Grid, params: Dict) -> Grid:
    """
    Scale grid by integer factor using nearest neighbor.
    params: {"factor": int}  (2 or 3)
    """
    factor = params["factor"]
    data = grid.data
    new_data = np.kron(data, np.ones((factor, factor), dtype=np.int8))
    return Grid(new_data)


def draw_border(grid: Grid, params: Dict) -> Grid:
    """
    Draw a 1-cell border of given color around grid edges.
    params: {"color": int}
    """
    color = params["color"]
    data = grid.data.copy()
    data[0, :] = color
    data[-1, :] = color
    data[:, 0] = color
    data[:, -1] = color
    return Grid(data)


def tile_pattern(grid: Grid, params: Dict) -> Grid:
    """
    Tile the grid n_rows x n_cols times.
    params: {"n_rows": int, "n_cols": int}
    Max output size enforced at 30x30.
    """
    n_rows = params["n_rows"]
    n_cols = params["n_cols"]
    tiled = np.tile(grid.data, (n_rows, n_cols))
    tiled = tiled[:30, :30]
    return Grid(tiled.astype(np.int8))


def fill_enclosed(grid: Grid, params: Dict) -> Grid:
    """
    Flood-fill all background cells enclosed by non-background cells.
    params: {"fill_color": int, "background": int}
    """
    from scipy.ndimage import label
    bg = params.get("background", 0)
    fill_color = params["fill_color"]
    data = grid.data.copy()
    h, w = data.shape

    border_mask = (data == bg)
    labeled, _ = label(border_mask)

    border_labels = set()
    border_labels.update(labeled[0, :].tolist())
    border_labels.update(labeled[-1, :].tolist())
    border_labels.update(labeled[:, 0].tolist())
    border_labels.update(labeled[:, -1].tolist())
    border_labels.discard(0)

    for r in range(h):
        for c in range(w):
            if data[r, c] == bg and labeled[r, c] not in border_labels:
                data[r, c] = fill_color

    return Grid(data)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# REGISTRY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

RULE_REGISTRY: Dict[str, Tuple[RuleFunction, Dict]] = {
    # Tier 1
    "reflect_horizontal":   (reflect_horizontal, {}),
    "reflect_vertical":     (reflect_vertical, {}),
    "reflect_diagonal":     (reflect_diagonal, {}),
    "reflect_antidiagonal": (reflect_antidiagonal, {}),
    "rotate_90":            (rotate_90, {}),
    "rotate_180":           (rotate_180, {}),
    "rotate_270":           (rotate_270, {}),
    "color_swap_all":       (color_swap_all,
                             {"color_a": "int 0-9", "color_b": "int 0-9"}),
    "color_remap":          (color_remap,
                             {"mapping": "dict[int->int]"}),
    "move_objects":         (move_objects,
                             {"dr": "int", "dc": "int", "background": "int"}),
    # Tier 2
    "gravity":              (gravity,
                             {"direction": "down|up|left|right",
                              "background": "int"}),
    "scale_up":             (scale_up, {"factor": "int 2|3"}),
    "draw_border":          (draw_border, {"color": "int 0-9"}),
    "tile_pattern":         (tile_pattern,
                             {"n_rows": "int", "n_cols": "int"}),
    "fill_enclosed":        (fill_enclosed,
                             {"fill_color": "int", "background": "int"}),
}


def apply_rule(rule_name: str, grid: Grid, params: Dict = {}) -> Grid:
    """Apply a named rule from the registry."""
    if rule_name not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule: {rule_name}. "
                         f"Available: {list(RULE_REGISTRY.keys())}")
    fn, _ = RULE_REGISTRY[rule_name]
    return fn(grid, params)
