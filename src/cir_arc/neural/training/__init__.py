"""Training infrastructure, dataset utilities, and perception model for CIR-ARC Phase 2."""

from cir_arc.neural.training.trainer import PerceptionModel, Trainer
from cir_arc.neural.training.dataset import (
    SyntheticArcDataset,
    SyntheticDataset,
    collate_variable_grids,
)

__all__ = [
    "PerceptionModel",
    "Trainer",
    "SyntheticArcDataset",
    "SyntheticDataset",
    "collate_variable_grids",
]
