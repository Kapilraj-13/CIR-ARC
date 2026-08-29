"""Statistical validation and accuracy tracking for executable world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class WorldModelValidationMetrics:
    total_transitions_evaluated: int = 0
    exact_matches: int = 0
    counterexamples_detected: int = 0
    accuracy: float = 1.0
    consistency_score: float = 1.0


class WorldModelValidator:
    """Monitors live consistency and accuracy of the executable world model digital twin."""

    def __init__(self) -> None:
        self.metrics = WorldModelValidationMetrics()

    def record_evaluation(self, is_exact_match: bool) -> None:
        self.metrics.total_transitions_evaluated += 1
        if is_exact_match:
            self.metrics.exact_matches += 1
        else:
            self.metrics.counterexamples_detected += 1

        tot = self.metrics.total_transitions_evaluated
        self.metrics.accuracy = float(self.metrics.exact_matches) / max(1, tot)
        self.metrics.consistency_score = 1.0 - (float(self.metrics.counterexamples_detected) / max(1, tot))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self.metrics.total_transitions_evaluated,
            "accuracy": round(self.metrics.accuracy, 4),
            "consistency_score": round(self.metrics.consistency_score, 4),
            "counterexamples": self.metrics.counterexamples_detected,
        }
