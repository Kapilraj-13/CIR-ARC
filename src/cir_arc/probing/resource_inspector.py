from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from cir_arc.environment.frame import FrameData
from cir_arc.environment.state_delta import find_objects


@dataclass
class ResourceIndicator:
    name: str
    value: Any
    source: str


class ResourceInspector:
    """Detects HUD resources, score counters, level indicators, and color distributions."""

    @staticmethod
    def inspect(frames: List[FrameData]) -> List[ResourceIndicator]:
        if not frames:
            return []

        latest = frames[-1]
        resources: List[ResourceIndicator] = [
            ResourceIndicator(
                name="levels_completed",
                value=latest.levels_completed,
                source="frame_metadata",
            ),
            ResourceIndicator(
                name="win_levels",
                value=latest.win_levels,
                source="frame_metadata",
            ),
            ResourceIndicator(
                name="grid_height",
                value=latest.grid.height,
                source="grid_geometry",
            ),
            ResourceIndicator(
                name="grid_width",
                value=latest.grid.width,
                source="grid_geometry",
            ),
            ResourceIndicator(
                name="num_layers",
                value=latest.grid.num_layers,
                source="grid_geometry",
            ),
        ]

        # Color pixel aggregate across frames
        color_counts: Dict[int, int] = {}
        for f in frames:
            for obj in find_objects(f):
                color_counts[obj.color] = color_counts.get(obj.color, 0) + obj.pixel_count

        for color, count in sorted(color_counts.items(), key=lambda item: -item[1])[:5]:
            resources.append(
                ResourceIndicator(
                    name=f"color_{color}_pixels",
                    value=count,
                    source="grid_aggregate",
                )
            )

        return resources
