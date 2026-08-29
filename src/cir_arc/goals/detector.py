"""Active Goal Detector generating candidate hypotheses from belief state observations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.goals.hypothesis import GoalHypothesis, GoalType


class GoalDetector:
    """Detects and initializes candidate goal hypotheses from visual layouts and belief state."""

    def __init__(
        self,
        goal_color_priorities: Optional[Dict[int, float]] = None,
        key_colors: Optional[Set[int]] = None,
        door_colors: Optional[Set[int]] = None,
    ) -> None:
        # Default color prior evidence values
        self.goal_color_priorities = goal_color_priorities or {
            14: 0.95,   # Bright Gold / Goal Flag
            3: 0.85,    # Green exit / target
            2: 0.75,    # Red goal
            11: 0.70,   # Cyan / Key
            6: 0.65,    # Magenta / Gem
        }
        self.key_colors = key_colors or {11, 6}
        self.door_colors = door_colors or {8, 7}

    def detect_candidate_goals(self, belief: BeliefState, comp_grid: np.ndarray) -> List[GoalHypothesis]:
        """Scans the composite environment grid and belief objects to extract candidate goals."""
        candidates: List[GoalHypothesis] = []
        H, W = comp_grid.shape
        player_pos = belief.player_location

        # 1. Scan for distinct colored objects/landmarks
        for obj_id, obj in belief.observed_objects.items():
            color = obj.color
            if color == belief.player_color or color == 0:
                continue

            # Prior evidence based on color saliency
            prior_ev = self.goal_color_priorities.get(color, 0.4)
            # Small compact objects are much more likely to be goals/keys than massive walls
            size_penalty = min(0.3, max(0.0, (obj.size - 1) * 0.05))
            evidence = max(0.2, prior_ev - size_penalty)

            centroid_coord = (int(round(obj.centroid[0])), int(round(obj.centroid[1])))

            if color in self.key_colors:
                g = GoalHypothesis(
                    goal_id=f"goal_collect_{color}_{centroid_coord[0]}_{centroid_coord[1]}",
                    goal_type=GoalType.COLLECT_OBJECT,
                    target_object_id=obj_id,
                    target_coordinate=centroid_coord,
                    target_color=color,
                    evidence=evidence,
                    consistency=0.8,
                    persistence=0.7,
                    description=f"Collect key/item {obj_id} (color {color}) at {centroid_coord}",
                )
                g.update_progress(player_pos, (H, W))
                candidates.append(g)

            elif color in self.door_colors:
                g = GoalHypothesis(
                    goal_id=f"goal_unlock_{color}_{centroid_coord[0]}_{centroid_coord[1]}",
                    goal_type=GoalType.UNLOCK_BARRIER,
                    target_object_id=obj_id,
                    target_coordinate=centroid_coord,
                    target_color=color,
                    evidence=evidence * 0.9,
                    consistency=0.75,
                    persistence=0.8,
                    description=f"Unlock barrier {obj_id} (color {color}) at {centroid_coord}",
                )
                g.update_progress(player_pos, (H, W))
                candidates.append(g)

            elif color in self.goal_color_priorities:
                g = GoalHypothesis(
                    goal_id=f"goal_reach_{color}_{centroid_coord[0]}_{centroid_coord[1]}",
                    goal_type=GoalType.REACH_LOCATION,
                    target_object_id=obj_id,
                    target_coordinate=centroid_coord,
                    target_color=color,
                    evidence=evidence,
                    consistency=0.9,
                    persistence=0.9,
                    description=f"Reach target goal landmark (color {color}) at {centroid_coord}",
                )
                g.update_progress(player_pos, (H, W))
                candidates.append(g)

        # 2. If no explicit landmarks found, add boundary / exploration targets
        if not candidates:
            corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
            for cr, cc in corners:
                g = GoalHypothesis(
                    goal_id=f"goal_explore_corner_{cr}_{cc}",
                    goal_type=GoalType.REACH_LOCATION,
                    target_coordinate=(cr, cc),
                    evidence=0.3,
                    consistency=0.5,
                    persistence=0.4,
                    description=f"Explore corner at ({cr}, {cc})",
                )
                g.update_progress(player_pos, (H, W))
                candidates.append(g)

        # Sort candidates by composite score descending
        candidates.sort(key=lambda g: g.score, reverse=True)
        return candidates
