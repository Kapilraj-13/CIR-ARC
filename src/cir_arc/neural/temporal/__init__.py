"""Temporal dynamics, tracking, event encoding, and mechanics inference for CIR-ARC."""

from cir_arc.neural.temporal.kinematics import classify_motion_direction, compute_velocity
from cir_arc.neural.temporal.tracker import TemporalSlotTracker, TrackedSlot, TemporalWorldState
from cir_arc.neural.temporal.event_encoder import CategoricalEventEncoder, TemporalEventMemory
from cir_arc.neural.temporal.transition import ActionConditionedTransitionModel, OnlineMechanicsTracker

__all__ = [
    "compute_velocity",
    "classify_motion_direction",
    "TemporalSlotTracker",
    "TrackedSlot",
    "TemporalWorldState",
    "CategoricalEventEncoder",
    "TemporalEventMemory",
    "ActionConditionedTransitionModel",
    "OnlineMechanicsTracker",
]
