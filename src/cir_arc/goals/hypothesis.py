"""Goal hypothesis representations and scoring metrics for ARC-AGI-3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GoalType(str, Enum):
    """Categorical types of inferred agent goals."""
    REACH_LOCATION = "REACH_LOCATION"          # Navigate to specific coordinate (r, c)
    COLLECT_OBJECT = "COLLECT_OBJECT"          # Collect item (key, coin, token)
    UNLOCK_BARRIER = "UNLOCK_BARRIER"          # Unlock / clear blocking door or wall
    ACTIVATE_MECHANISM = "ACTIVATE_MECHANISM"  # Press button / toggle switch
    MAXIMIZE_RESOURCE = "MAXIMIZE_RESOURCE"    # Maximize quantity of collected objects
    REACH_STABLE_STATE = "REACH_STABLE_STATE"  # Eliminate moving objects or reach terminal state


@dataclass
class GoalHypothesis:
    """An inferred candidate goal with evidence and scoring metrics."""
    goal_id: str
    goal_type: GoalType
    target_object_id: Optional[str] = None
    target_coordinate: Optional[Tuple[int, int]] = None
    target_color: Optional[int] = None
    description: str = ""

    # Multi-factor Scoring Attributes
    evidence: float = 0.5           # Initial visual / prior saliency [0.0, 1.0]
    progress: float = 0.0           # Metric of spatial or state progress towards goal [0.0, 1.0]
    consistency: float = 0.5        # Consistency with known environment invariants [0.0, 1.0]
    persistence: float = 0.5        # Stability of goal across state transitions [0.0, 1.0]
    contradiction: float = 0.0      # Penalties from counterexamples or failure [0.0, 1.0]
    action_cost: float = 0.0        # Estimated pathfinding / action distance

    # Execution State
    is_active: bool = False
    is_satisfied: bool = False
    subgoals: List[GoalHypothesis] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Composite heuristic goal score."""
        # Normalized score combining evidence, progress, and cost
        base = 0.35 * self.evidence + 0.25 * self.progress + 0.20 * self.consistency + 0.20 * self.persistence
        penalty = 0.40 * self.contradiction + 0.10 * min(1.0, self.action_cost / 50.0)
        return max(0.0, min(1.0, base - penalty))

    def update_progress(self, current_player_pos: Optional[Tuple[int, int]], grid_shape: Tuple[int, int]) -> None:
        """Update progress based on Manhattan proximity to target coordinate."""
        if self.target_coordinate is None or current_player_pos is None:
            return
        pr, pc = current_player_pos
        tr, tc = self.target_coordinate
        max_dist = float(grid_shape[0] + grid_shape[1])
        dist = float(abs(pr - tr) + abs(pc - tc))
        self.action_cost = dist
        # Progress increases as distance decreases
        self.progress = max(0.0, min(1.0, 1.0 - (dist / max(max_dist, 1.0))))
        if dist <= 0.5:
            self.is_satisfied = True
            self.progress = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "target_coordinate": self.target_coordinate,
            "target_color": self.target_color,
            "score": round(self.score, 4),
            "evidence": round(self.evidence, 4),
            "progress": round(self.progress, 4),
            "is_satisfied": self.is_satisfied,
            "description": self.description,
        }
