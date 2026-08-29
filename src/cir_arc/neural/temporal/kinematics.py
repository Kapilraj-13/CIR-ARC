from __future__ import annotations

import math
from typing import Optional, Tuple


def compute_velocity(
    pos_before: Tuple[float, float],
    pos_after: Tuple[float, float],
    dt: float = 1.0,
) -> Tuple[float, float]:
    """Compute (d_row / dt, d_col / dt)."""
    dr = (pos_after[0] - pos_before[0]) / max(dt, 1e-6)
    dc = (pos_after[1] - pos_before[1]) / max(dt, 1e-6)
    return (dr, dc)


def classify_motion_direction(dr: float, dc: float, threshold: float = 0.1) -> str:
    """Classify (dr, dc) into 8-way compass direction or STILL."""
    mag = math.sqrt(dr * dr + dc * dc)
    if mag < threshold:
        return "STILL"

    angle = math.atan2(-dr, dc)  # In grid: row increases downwards, so -dr is up
    deg = math.degrees(angle) % 360  # [0, 360) where 0 is East, 90 is North, 180 is West, 270 is South

    if 337.5 <= deg or deg < 22.5:
        return "E"
    elif 22.5 <= deg < 67.5:
        return "NE"
    elif 67.5 <= deg < 112.5:
        return "N"
    elif 112.5 <= deg < 157.5:
        return "NW"
    elif 157.5 <= deg < 202.5:
        return "W"
    elif 202.5 <= deg < 247.5:
        return "SW"
    elif 247.5 <= deg < 292.5:
        return "S"
    else:
        return "SE"
