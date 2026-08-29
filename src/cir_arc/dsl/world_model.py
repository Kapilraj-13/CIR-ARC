from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from cir_arc.environment.actions import Action, DIRECTION_VECTORS
from cir_arc.environment.frame import MultiLayerGrid
from cir_arc.hypothesis.transition_grammar import TransitionRule


class DSLWorldModel:
    """Predictive forward world model simulating grid transitions using learned and DSL rules."""

    def __init__(self, rules: Optional[List[TransitionRule]] = None) -> None:
        self.rules: List[TransitionRule] = rules or []

    def set_rules(self, rules: List[TransitionRule]) -> None:
        self.rules = list(rules)

    def predict_step(
        self,
        grid: Union[MultiLayerGrid, np.ndarray],
        action: Action,
        player_color: int = 9,
        barrier_color: int = 5,
    ) -> np.ndarray:
        """Predict the next state grid given the current grid and action."""
        if isinstance(grid, MultiLayerGrid):
            current = grid.composite()
        else:
            current = grid.copy()

        aid = action.action_id

        # 1. Check if any induced rule applies
        matching_rules = [r for r in self.rules if r.matches(action)]
        if matching_rules:
            best_rule = max(matching_rules, key=lambda r: r.confidence)
            return best_rule.apply_to_grid(current, player_color=player_color)

        # 2. Fallback to default directional physics
        if aid in DIRECTION_VECTORS:
            dr, dc = DIRECTION_VECTORS[aid]
            from cir_arc.dsl.interactive_primitives import step_translate
            return step_translate(current, player_color, dr, dc, barrier_colors={barrier_color})

        return current

    def predict_trajectory(
        self,
        initial_grid: Union[MultiLayerGrid, np.ndarray],
        actions: List[Action],
        player_color: int = 9,
    ) -> List[np.ndarray]:
        """Simulate an entire sequence of actions."""
        states = []
        if isinstance(initial_grid, MultiLayerGrid):
            curr = initial_grid.composite()
        else:
            curr = initial_grid.copy()
        states.append(curr)

        for action in actions:
            curr = self.predict_step(curr, action, player_color=player_color)
            states.append(curr)

        return states

    @staticmethod
    def evaluate_prediction_loss(predicted_grid: np.ndarray, target_grid: np.ndarray) -> float:
        """Compute normalized pixel error in [0, 1]."""
        if predicted_grid.shape != target_grid.shape:
            return 1.0
        diff_count = int(np.count_nonzero(predicted_grid != target_grid))
        return float(diff_count) / max(predicted_grid.size, 1)
