"""Uncertainty quantification and epistemic entropy tracking across environment dimensions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class UncertaintyModel:
    """Quantifies agent uncertainty over grid locations, affordances, and transitions."""

    def __init__(self, height: int = 30, width: int = 30) -> None:
        self.height = height
        self.width = width
        # Spatial visit count matrix: counts how many times player or perception has focused on (r, c)
        self.visit_counts = np.zeros((height, width), dtype=np.int32)
        # Affordance uncertainty per color: map color_id -> float [0.0 (certain) to 1.0 (completely unknown)]
        self.color_passability_uncertainty: Dict[int, float] = {c: 1.0 for c in range(16)}
        # Background is initially known to be passable (0: background)
        self.color_passability_uncertainty[0] = 0.0
        # Interaction rule uncertainty: map (color_a, action_type) -> float
        self.interaction_uncertainty: Dict[Tuple[int, int], float] = {}

    def resize(self, height: int, width: int) -> None:
        if height != self.height or width != self.width:
            new_counts = np.zeros((height, width), dtype=np.int32)
            min_h = min(height, self.height)
            min_w = min(width, self.width)
            new_counts[:min_h, :min_w] = self.visit_counts[:min_h, :min_w]
            self.visit_counts = new_counts
            self.height = height
            self.width = width

    def record_visit(self, r: int, c: int) -> None:
        if 0 <= r < self.height and 0 <= c < self.width:
            self.visit_counts[r, c] += 1

    def update_color_certainty(self, color: int, confidence: float) -> None:
        self.color_passability_uncertainty[color] = max(0.0, min(1.0, 1.0 - confidence))

    def update_interaction_certainty(self, color: int, action: int, confidence: float) -> None:
        self.interaction_uncertainty[(color, action)] = max(0.0, min(1.0, 1.0 - confidence))

    def get_spatial_uncertainty_map(self) -> np.ndarray:
        """Computes normalized spatial uncertainty map in [0, 1] where 1.0 means unvisited."""
        # Decays exponentially with visit counts: u(r, c) = exp(-0.5 * visits)
        return np.exp(-0.5 * self.visit_counts).astype(np.float32)

    def get_color_uncertainty(self, color: int) -> float:
        return self.color_passability_uncertainty.get(color, 1.0)

    def get_interaction_uncertainty(self, color: int, action: int) -> float:
        return self.interaction_uncertainty.get((color, action), 1.0)

    def compute_total_epistemic_entropy(self) -> float:
        """Aggregate epistemic uncertainty metric across space and affordances."""
        spatial_ent = float(np.mean(self.get_spatial_uncertainty_map()))
        color_ent = float(np.mean(list(self.color_passability_uncertainty.values()))) if self.color_passability_uncertainty else 0.0
        return 0.6 * spatial_ent + 0.4 * color_ent
