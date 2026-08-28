from __future__ import annotations
from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask
from cir_arc.dsl.primitives import RULE_REGISTRY, apply_rule
from typing import List, Optional, Dict, Tuple
import numpy as np


class HeuristicAgent:
    """
    Rule-matching heuristic agent.

    Strategy: for each known DSL rule, check if applying it (with various
    parameter settings) to all train inputs produces the train outputs.
    If a matching rule is found, apply it to the test input.
    Falls back to copy_input if no rule matches.
    """
    name = "heuristic"

    # Parameter-free rules (no params needed to try)
    PARAMETERLESS_RULES = [
        "reflect_horizontal", "reflect_vertical",
        "reflect_diagonal", "reflect_antidiagonal",
        "rotate_90", "rotate_180", "rotate_270",
    ]

    def predict(self, task: ArcTask) -> List[Grid]:
        predictions = []
        for test_pair in task.test_pairs:
            result = self._solve_one(task, test_pair.input)
            if result is None:
                result = test_pair.input.copy()
            predictions.append(result)
        return predictions

    def _solve_one(self, task: ArcTask, test_input: Grid) -> Optional[Grid]:
        # Try parameter-free rules first
        for rule_name in self.PARAMETERLESS_RULES:
            if self._rule_matches_all_demos(task, rule_name, {}):
                return apply_rule(rule_name, test_input, {})

        # Try color_swap_all with inferred colors
        swap_params = self._infer_color_swap(task)
        if swap_params is not None:
            return apply_rule("color_swap_all", test_input, swap_params)

        # Try gravity in each direction
        for direction in ["down", "up", "left", "right"]:
            params = {"direction": direction, "background": 0}
            if self._rule_matches_all_demos(task, "gravity", params):
                return apply_rule("gravity", test_input, params)

        # Try scale_up with factor 2 and 3
        for factor in [2, 3]:
            params = {"factor": factor}
            if self._rule_matches_all_demos(task, "scale_up", params):
                return apply_rule("scale_up", test_input, params)

        # Try draw_border with each color
        for color in range(1, 10):
            params = {"color": color}
            if self._rule_matches_all_demos(task, "draw_border", params):
                return apply_rule("draw_border", test_input, params)

        # Try tile_pattern with small factors
        for nr in range(2, 5):
            for nc in range(2, 5):
                params = {"n_rows": nr, "n_cols": nc}
                if self._rule_matches_all_demos(task, "tile_pattern", params):
                    return apply_rule("tile_pattern", test_input, params)

        # Try fill_enclosed with each color
        for fill_color in range(1, 10):
            params = {"fill_color": fill_color, "background": 0}
            if self._rule_matches_all_demos(task, "fill_enclosed", params):
                return apply_rule("fill_enclosed", test_input, params)

        return None

    def _rule_matches_all_demos(
        self, task: ArcTask, rule_name: str, params: Dict
    ) -> bool:
        """Check if a rule with given params transforms all train inputs to outputs."""
        for pair in task.train_pairs:
            if pair.output is None:
                continue
            try:
                predicted = apply_rule(rule_name, pair.input, params)
                if predicted != pair.output:
                    return False
            except Exception:
                return False
        return True

    def _infer_color_swap(self, task: ArcTask) -> Optional[Dict]:
        """Try to infer a color swap from the first train pair."""
        if not task.train_pairs or task.train_pairs[0].output is None:
            return None

        inp_data = task.train_pairs[0].input.data
        out_data = task.train_pairs[0].output.data

        if inp_data.shape != out_data.shape:
            return None

        diff_mask = inp_data != out_data
        if not diff_mask.any():
            return None

        inp_diff_vals = set(inp_data[diff_mask].tolist())
        out_diff_vals = set(out_data[diff_mask].tolist())

        if len(inp_diff_vals) == 2 and inp_diff_vals == out_diff_vals:
            colors = sorted(inp_diff_vals)
            params = {"color_a": colors[0], "color_b": colors[1]}
            if self._rule_matches_all_demos(task, "color_swap_all", params):
                return params

        return None
