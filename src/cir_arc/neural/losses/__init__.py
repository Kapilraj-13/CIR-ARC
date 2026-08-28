"""CIR-ARC Phase 2 Neural Loss Functions and Hungarian Matching."""

from cir_arc.neural.losses.matching import hungarian_matching
from cir_arc.neural.losses.reconstruction import (
    ReconstructionLoss,
    reconstruction_loss,
)
from cir_arc.neural.losses.property import (
    PropertyLoss,
    color_loss,
    position_loss,
    size_loss,
    objectness_loss,
    compute_property_losses,
)
from cir_arc.neural.losses.diversity import (
    SlotDiversityLoss,
    ObjectnessSparsityLoss,
    diversity_loss,
    objectness_sparsity_loss,
)

__all__ = [
    "hungarian_matching",
    "ReconstructionLoss",
    "reconstruction_loss",
    "PropertyLoss",
    "color_loss",
    "position_loss",
    "size_loss",
    "objectness_loss",
    "compute_property_losses",
    "SlotDiversityLoss",
    "ObjectnessSparsityLoss",
    "diversity_loss",
    "objectness_sparsity_loss",
]
