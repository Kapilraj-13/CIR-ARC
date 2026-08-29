"""Abstract relational environment schemas and safe transfer novelty detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.memory.episodic import Episode


@dataclass
class EnvironmentSchema:
    """Compressed causal abstraction distilled from an episode."""
    schema_id: str
    game_type: str                            # e.g. "maze_navigation", "locksmith_puzzle", "sokoban_push"
    required_colors: Set[int] = field(default_factory=set)
    interaction_sequence: List[str] = field(default_factory=list)  # e.g. ["collect_key", "unlock_door", "reach_goal"]
    invariants: Dict[str, Any] = field(default_factory=dict)


class NoveltyDetector:
    """Checks whether a new puzzle matches assumptions of a known schema to prevent negative transfer."""

    @staticmethod
    def match_schema(schema: EnvironmentSchema, belief: BeliefState, comp_grid: np.ndarray) -> Tuple[bool, float]:
        """Calculates compatibility score between known schema and current belief state."""
        grid_colors = set(np.unique(comp_grid))

        # Check if all required colors for the schema are present in the grid
        missing_colors = schema.required_colors - grid_colors
        if missing_colors:
            return False, 0.0

        # Compatibility score based on shared colors
        overlap = len(schema.required_colors.intersection(grid_colors))
        score = float(overlap) / max(1, len(schema.required_colors))
        is_match = (score >= 0.8)
        return is_match, score
