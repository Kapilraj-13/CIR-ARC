"""Losses package for CIR-ARC neural perception training."""

from cir_arc.neural.losses.matching import hungarian_matching, compute_cost_matrix
from cir_arc.neural.losses.reconstruction import reconstruction_loss
from cir_arc.neural.losses.property import (
    color_loss,
    position_loss,
    size_loss,
    shape_loss,
    orientation_loss,
    symmetry_loss,
    objectness_loss,
    bbox_loss,
    dimensions_loss,
    compute_property_losses,
    PropertyLoss,
)
from cir_arc.neural.losses.diversity import (
    slot_diversity_loss,
    diversity_loss,
    objectness_sparsity_loss,
)
from cir_arc.neural.losses.boundary import (
    boundary_loss,
    cell_objectness_loss,
)
from cir_arc.neural.losses.mask import (
    slot_mask_loss,
    mask_exclusivity_loss,
)
from cir_arc.neural.losses.relation import relation_loss
from cir_arc.neural.losses.identity import object_identity_contrastive_loss

__all__ = [
    "hungarian_matching",
    "compute_cost_matrix",
    "reconstruction_loss",
    "color_loss",
    "position_loss",
    "size_loss",
    "shape_loss",
    "orientation_loss",
    "symmetry_loss",
    "objectness_loss",
    "bbox_loss",
    "dimensions_loss",
    "compute_property_losses",
    "PropertyLoss",
    "slot_diversity_loss",
    "diversity_loss",
    "objectness_sparsity_loss",
    "boundary_loss",
    "cell_objectness_loss",
    "slot_mask_loss",
    "mask_exclusivity_loss",
    "relation_loss",
    "object_identity_contrastive_loss",
]
