"""Training infrastructure, dataset utilities, and perception model for CIR-ARC Phase 2."""

from cir_arc.neural.training.trainer import PerceptionModel, Trainer
from cir_arc.neural.training.dataset import (
    SyntheticArcDataset,
    SyntheticDataset,
    collate_variable_grids,
)
from cir_arc.neural.training.reasoning_dataset import (
    ReasoningArcDataset,
    collate_reasoning_batch,
    ensure_synthetic_data,
)

__all__ = [
    "PerceptionModel",
    "Trainer",
    "SyntheticArcDataset",
    "SyntheticDataset",
    "collate_variable_grids",
    "ReasoningArcDataset",
    "collate_reasoning_batch",
    "ensure_synthetic_data",
]

