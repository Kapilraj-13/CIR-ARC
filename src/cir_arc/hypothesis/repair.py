"""Hypothesis repair engine converting empirical counterexamples into corrected causal models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from cir_arc.belief.facts import FactType, Provenance
from cir_arc.belief.state import BeliefState
from cir_arc.hypothesis.counterexample import Counterexample
from cir_arc.hypothesis.induction_engine import HypothesisInductionEngine
from cir_arc.hypothesis.transition_grammar import TransitionRule

logger = logging.getLogger(__name__)


class HypothesisRepair:
    """Repairs hypothesis rules and belief states when counterexamples invalidate assumptions."""

    def __init__(self) -> None:
        self.repair_history: List[Dict[str, Any]] = []

    def repair_from_counterexample(
        self,
        counterexample: Counterexample,
        belief: BeliefState,
        induction_engine: HypothesisInductionEngine,
    ) -> Dict[str, Any]:
        """Analyzes a counterexample and updates affected rules and belief facts."""
        rule_id = counterexample.failed_rule_id
        action = counterexample.action
        repairs_applied = []

        # 1. Check if the failed rule is in the induction engine
        if rule_id and rule_id in induction_engine.rules:
            rule = induction_engine.rules[rule_id]
            rule.refutation_count += 1
            rule.confidence = float(rule.support_count) / float(rule.support_count + rule.refutation_count)

            # If confidence drops significantly, demote rule or strengthen preconditions
            if rule.confidence < 0.5:
                rule.condition_type = "CONDITIONAL_ON_NEIGHBOR"
                repairs_applied.append(f"Strengthened preconditions on {rule_id}")

        # 2. Analyze mismatched cells to deduce obstacle/collision constraints
        for r, c, pred_val, actual_val in counterexample.mismatched_cells:
            # If we predicted player to move into (r, c) with pred_val == player_color, but actual_val is still an obstacle color
            if pred_val == belief.player_color and actual_val != belief.player_color:
                # Discovered an impassable cell / obstacle!
                obstacle_color = actual_val
                belief.facts.add_or_update(
                    FactType.PASSABILITY,
                    subject=f"color_{obstacle_color}",
                    predicate="is_passable",
                    value=False,
                    provenance=Provenance.OBSERVED,
                    confidence=0.95,
                )
                repairs_applied.append(f"Marked color_{obstacle_color} at ({r},{c}) as impassable obstacle")

        repair_record = {
            "counterexample_id": counterexample.counterexample_id,
            "step": counterexample.step,
            "action": action.action_id,
            "repairs_applied": repairs_applied,
        }
        self.repair_history.append(repair_record)
        belief.failed_hypotheses.append(repair_record)
        return repair_record
