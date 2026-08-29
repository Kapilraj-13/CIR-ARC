"""Counterexample detection for model-based digital twin verification and hypothesis falsification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData


@dataclass
class Counterexample:
    """Represents an observed discrepancy between a model prediction and reality."""
    counterexample_id: str
    action: Action
    predicted_state_hash: str
    actual_state_hash: str
    mismatched_cells: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (r, c, pred_val, actual_val)
    failed_rule_id: Optional[str] = None
    failed_assumption: str = ""
    step: int = 0
    severity: float = 1.0  # [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            "action": self.action.to_dict(),
            "mismatched_count": len(self.mismatched_cells),
            "failed_rule_id": self.failed_rule_id,
            "failed_assumption": self.failed_assumption,
            "step": self.step,
        }


class CounterexampleDetector:
    """Compares predicted next frame against actual observed frame to identify counterexamples."""

    def __init__(self, mismatch_threshold: int = 0) -> None:
        self.mismatch_threshold = mismatch_threshold

    def detect_counterexample(
        self,
        predicted_grid: np.ndarray,
        actual_frame: FrameData,
        action: Action,
        failed_rule_id: Optional[str] = None,
        assumption: str = "",
    ) -> Optional[Counterexample]:
        """Compares predicted composite grid with actual composite grid."""
        actual_comp = actual_frame.grid.composite()

        if predicted_grid.shape != actual_comp.shape:
            ce = Counterexample(
                counterexample_id=f"ce_shape_{actual_frame.step_count}",
                action=action,
                predicted_state_hash="shape_mismatch",
                actual_state_hash=actual_frame.hash(),
                failed_rule_id=failed_rule_id,
                failed_assumption=f"Shape mismatch: pred {predicted_grid.shape} vs actual {actual_comp.shape}",
                step=actual_frame.step_count,
                severity=1.0,
            )
            return ce

        diff_indices = np.argwhere(predicted_grid != actual_comp)
        if len(diff_indices) > self.mismatch_threshold:
            mismatches = []
            for r, c in diff_indices:
                r_idx, c_idx = int(r), int(c)
                pred_val = int(predicted_grid[r_idx, c_idx])
                actual_val = int(actual_comp[r_idx, c_idx])
                mismatches.append((r_idx, c_idx, pred_val, actual_val))

            failed_assump = assumption or f"Expected {len(mismatches)} cells to match prediction under action {action.action_id}"
            ce = Counterexample(
                counterexample_id=f"ce_step_{actual_frame.step_count}_{action.action_id}",
                action=action,
                predicted_state_hash=str(hash(predicted_grid.tobytes())),
                actual_state_hash=actual_frame.hash(),
                mismatched_cells=mismatches,
                failed_rule_id=failed_rule_id,
                failed_assumption=failed_assump,
                step=actual_frame.step_count,
                severity=min(1.0, float(len(mismatches)) / 10.0),
            )
            return ce

        return None
