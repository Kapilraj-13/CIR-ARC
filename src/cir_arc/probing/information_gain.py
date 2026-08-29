"""Information-Gain Explorer choosing high-value exploratory actions under uncertainty."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.belief.state import BeliefState
from cir_arc.environment.actions import DIRECTION_VECTORS, Action, ActionType


class InformationGainExplorer:
    """Evaluates and ranks candidate actions by expected epistemic information gain per unit cost."""

    def __init__(
        self,
        weight_uncertainty: float = 1.0,
        weight_novelty: float = 0.5,
        weight_cost: float = 0.1,
        risk_penalty: float = 0.5,
    ) -> None:
        self.w_unc = weight_uncertainty
        self.w_nov = weight_novelty
        self.w_cost = weight_cost
        self.w_risk = risk_penalty

    def score_action(
        self,
        action: Action,
        belief: BeliefState,
        comp_grid: np.ndarray,
    ) -> float:
        """Scores a single candidate action based on information gain and risk."""
        aid = action.action_id
        if aid not in DIRECTION_VECTORS or belief.player_location is None:
            # Interaction / no-op action
            return 0.1

        pr, pc = belief.player_location
        dr, dc = DIRECTION_VECTORS[aid]
        nr, nc = pr + dr, pc + dc
        H, W = comp_grid.shape

        # Out of bounds -> zero value, high penalty
        if not (0 <= nr < H and 0 <= nc < W):
            return -1.0

        target_color = int(comp_grid[nr, nc])

        # 1. Spatial uncertainty reduction: is the target cell unvisited?
        spatial_uncertainty = belief.uncertainty.get_spatial_uncertainty_map()[nr, nc]

        # 2. Color / affordance uncertainty reduction
        color_uncertainty = belief.uncertainty.get_color_uncertainty(target_color)

        # 3. Novelty bonus: haven't stepped on (nr, nc)
        is_novel = 1.0 if (nr, nc) not in belief.visited_coordinates else 0.0

        # 4. Collision / dead-end risk: if confirmed blocked wall, slight penalty
        is_blocked = target_color in belief.facts.get_known_blocked_colors() or target_color == 5
        risk = 0.8 if is_blocked else 0.0

        # Compute information gain per unit cost (action cost = 1)
        expected_gain = (self.w_unc * (0.6 * spatial_uncertainty + 0.4 * color_uncertainty) + self.w_nov * is_novel)
        score = expected_gain - (self.w_risk * risk)
        return float(score)

    def select_best_exploratory_action(
        self,
        belief: BeliefState,
        comp_grid: np.ndarray,
        available_actions: List[int],
    ) -> Action:
        """Selects the action yielding maximum information gain."""
        best_score = -float("inf")
        best_action = Action(ActionType.ACTION1)

        for act_code in available_actions:
            act = Action.from_int(act_code)
            score = self.score_action(act, belief, comp_grid)
            if score > best_score:
                best_score = score
                best_action = act

        return best_action
