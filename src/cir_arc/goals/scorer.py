"""Goal scoring and re-ranking engine based on evidence, progress, and consistency."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from cir_arc.goals.hypothesis import GoalHypothesis


class GoalScorer:
    """Evaluates and adjusts GoalHypothesis scores based on dynamic interaction telemetry."""

    def __init__(
        self,
        weight_evidence: float = 0.35,
        weight_progress: float = 0.25,
        weight_consistency: float = 0.20,
        weight_persistence: float = 0.20,
        weight_contradiction: float = 0.40,
        weight_cost: float = 0.10,
    ) -> None:
        self.w_ev = weight_evidence
        self.w_prog = weight_progress
        self.w_cons = weight_consistency
        self.w_pers = weight_persistence
        self.w_cont = weight_contradiction
        self.w_cost = weight_cost

    def score_goal(self, goal: GoalHypothesis) -> float:
        """Calculate weighted score for a single goal hypothesis."""
        base = (
            self.w_ev * goal.evidence
            + self.w_prog * goal.progress
            + self.w_cons * goal.consistency
            + self.w_pers * goal.persistence
        )
        penalty = (
            self.w_cont * goal.contradiction
            + self.w_cost * min(1.0, goal.action_cost / 50.0)
        )
        return max(0.0, min(1.0, base - penalty))

    def penalize_contradiction(self, goal: GoalHypothesis, penalty_amount: float = 0.3) -> None:
        """Apply contradiction penalty when an attempt towards this goal fails or encounters a counterexample."""
        goal.contradiction = min(1.0, goal.contradiction + penalty_amount)

    def reinforce_success(self, goal: GoalHypothesis, boost_amount: float = 0.2) -> None:
        """Reinforce persistence and progress upon positive partial progress."""
        goal.evidence = min(1.0, goal.evidence + boost_amount)
        goal.persistence = min(1.0, goal.persistence + 0.1)
