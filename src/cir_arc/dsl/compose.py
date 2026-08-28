from __future__ import annotations
from typing import List, Tuple, Dict, Any

from cir_arc.core.grid import Grid
from cir_arc.dsl.primitives import apply_rule


def compose_rules(
    grid: Grid,
    rules: List[Tuple[str, Dict[str, Any]]]
) -> Grid:
    """
    Apply a sequence of rules to a grid in order.

    Parameters
    ----------
    grid  : Starting grid
    rules : List of (rule_name, params) tuples

    Returns
    -------
    Final grid after all rules applied.

    Example
    -------
    result = compose_rules(grid, [
        ("rotate_90", {}),
        ("color_swap_all", {"color_a": 1, "color_b": 2}),
    ])
    """
    current = grid
    for rule_name, params in rules:
        current = apply_rule(rule_name, current, params)
    return current
