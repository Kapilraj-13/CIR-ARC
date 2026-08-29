"""Goal Manager orchestrating goal ranking, hierarchical subgoals, and status transitions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.goals.detector import GoalDetector
from cir_arc.goals.hypothesis import GoalHypothesis, GoalType
from cir_arc.goals.scorer import GoalScorer


class GoalManager:
    """Orchestrates candidate goal hypotheses, hierarchical subgoal decomposition, and active goal selection."""

    def __init__(
        self,
        detector: Optional[GoalDetector] = None,
        scorer: Optional[GoalScorer] = None,
    ) -> None:
        self.detector = detector or GoalDetector()
        self.scorer = scorer or GoalScorer()
        self.candidate_goals: List[GoalHypothesis] = []
        self.active_goal: Optional[GoalHypothesis] = None
        self.active_subgoal: Optional[GoalHypothesis] = None
        self.satisfied_goals: List[GoalHypothesis] = []
        self.pruned_goals: List[GoalHypothesis] = []

    def update_from_belief(self, belief: BeliefState, comp_grid: np.ndarray) -> Optional[GoalHypothesis]:
        """Update goal hypotheses from latest belief state, generate subgoals, and return active goal."""
        # 1. If no candidate goals exist, detect them
        if not self.candidate_goals:
            self.candidate_goals = self.detector.detect_candidate_goals(belief, comp_grid)

        # 2. Update progress on all active candidate goals
        player_pos = belief.player_location
        grid_shape = belief.grid_shape

        for g in self.candidate_goals:
            g.update_progress(player_pos, grid_shape)
            # Check if an entity target color has disappeared / been collected
            if g.target_coordinate is not None and g.target_color is not None:
                tr, tc = g.target_coordinate
                if 0 <= tr < comp_grid.shape[0] and 0 <= tc < comp_grid.shape[1]:
                    if comp_grid[tr, tc] != g.target_color and comp_grid[tr, tc] != belief.player_color:
                        g.is_satisfied = True

        # 3. Check for satisfied goals
        remaining_goals = []
        for g in self.candidate_goals:
            if g.is_satisfied:
                self.satisfied_goals.append(g)
            elif g.score > 0.05:
                remaining_goals.append(g)
            else:
                self.pruned_goals.append(g)
        self.candidate_goals = remaining_goals

        # 4. Check if we need to decompose goals into subgoals (e.g. Key -> Door -> Goal)
        self._hierarchical_subgoal_decomposition(belief)

        # 5. Rank remaining goals
        self.candidate_goals.sort(key=lambda g: g.score, reverse=True)

        # 6. Select active target
        if self.active_subgoal is not None and not self.active_subgoal.is_satisfied:
            self.active_goal = self.active_subgoal
        elif self.candidate_goals:
            self.active_goal = self.candidate_goals[0]
        else:
            self.active_goal = None

        return self.active_goal

    def _hierarchical_subgoal_decomposition(self, belief: BeliefState) -> None:
        """Decomposes primary goal into prerequisite subgoals if barriers/keys are detected."""
        keys = [g for g in self.candidate_goals if g.goal_type == GoalType.COLLECT_OBJECT and not g.is_satisfied]
        doors = [g for g in self.candidate_goals if g.goal_type == GoalType.UNLOCK_BARRIER and not g.is_satisfied]
        primary_goals = [g for g in self.candidate_goals if g.goal_type == GoalType.REACH_LOCATION and not g.is_satisfied]

        if keys:
            self.active_subgoal = keys[0]
            if doors and primary_goals:
                primary_goals[0].subgoals = [keys[0], doors[0]]
        elif doors:
            self.active_subgoal = doors[0]
            if primary_goals:
                primary_goals[0].subgoals = [doors[0]]
        else:
            self.active_subgoal = None

    def mark_goal_contradicted(self, goal_id: str) -> None:
        """Penalize a goal that resulted in a contradiction or unreachable state."""
        for g in self.candidate_goals:
            if g.goal_id == goal_id:
                self.scorer.penalize_contradiction(g, 0.4)
                if g.score < 0.1:
                    self.candidate_goals.remove(g)
                    self.pruned_goals.append(g)
                break

    def get_active_target_coordinate(self) -> Optional[Tuple[int, int]]:
        if self.active_subgoal is not None and not self.active_subgoal.is_satisfied:
            return self.active_subgoal.target_coordinate
        if self.active_goal is not None:
            return self.active_goal.target_coordinate
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "active_goal": self.active_goal.to_dict() if self.active_goal else None,
            "active_subgoal": self.active_subgoal.to_dict() if self.active_subgoal else None,
            "candidates_count": len(self.candidate_goals),
            "satisfied_count": len(self.satisfied_goals),
        }
