"""Evaluation and metrics package for neural perception in CIR-ARC."""

from cir_arc.neural.evaluation.perception_metrics import (
    reconstruction_accuracy,
    object_detection_f1,
    color_accuracy,
    position_mae,
    size_mae,
    compute_perception_metrics,
)
from cir_arc.neural.evaluation.reasoner_metrics import (
    ReasonerMetricsTracker,
    compute_f1_from_confusion,
)

__all__ = [
    "reconstruction_accuracy",
    "object_detection_f1",
    "color_accuracy",
    "position_mae",
    "size_mae",
    "compute_perception_metrics",
    "ReasonerMetricsTracker",
    "compute_f1_from_confusion",
]
