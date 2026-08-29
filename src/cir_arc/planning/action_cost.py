"""Action cost and risk estimation models for planning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.environment.actions import Action


class ActionCostModel:
    """Estimates the execution cost and risk profile for taking actions."""

    def __init__(self, step_cost: float = 1.0, turn_cost: float = 0.2) -> None:
        self.step_cost = step_cost
        self.turn_cost = turn_cost

    def estimate_plan_cost(self, plan: List[Action]) -> float:
        """Computes total cost of a primitive action sequence."""
        if not plan:
            return 0.0
        total = float(len(plan)) * self.step_cost
        # Add slight penalty for frequent direction switches
        for i in range(1, len(plan)):
            if plan[i].action_id != plan[i - 1].action_id:
                total += self.turn_cost
        return total
