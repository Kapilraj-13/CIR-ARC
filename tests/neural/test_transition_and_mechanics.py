"""Unit tests for ActionConditionedTransitionModel and OnlineMechanicsTracker."""

from __future__ import annotations

import pytest
import torch

from cir_arc.neural.world_state import StructuredObject
from cir_arc.neural.temporal.transition import ActionConditionedTransitionModel, OnlineMechanicsTracker


def test_transition_model_prediction():
    model = ActionConditionedTransitionModel(slot_dim=128)
    slots = torch.randn(2, 24, 128)
    action = torch.tensor([1, 4])

    next_slots, delta_pos = model(slots, action)
    assert next_slots.shape == (2, 24, 128)
    assert delta_pos.shape == (2, 24, 2)
    assert not torch.isnan(next_slots).any()


def test_mechanics_tracker_gravity_learning():
    tracker = OnlineMechanicsTracker()

    obj_t0 = StructuredObject(
        slot_id=0, color=1, confidence=1.0, centroid=(0.1, 0.5),
        bbox=(0.0, 0.4, 0.2, 0.6), width=0.2, height=0.2, area=0.04,
        perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
        symmetries=(True, True, True, True), has_holes=False,
    )
    obj_t1 = StructuredObject(
        slot_id=0, color=1, confidence=1.0, centroid=(0.3, 0.5),  # Fell downwards (dr > 0)
        bbox=(0.2, 0.4, 0.4, 0.6), width=0.2, height=0.2, area=0.04,
        perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
        symmetries=(True, True, True, True), has_holes=False,
    )

    for _ in range(4):
        belief = tracker.update_from_transition(action_id=0, prev_objects=[obj_t0], curr_objects=[obj_t1])

    assert belief.gravity[0] == 1.0  # Downwards vertical drift
    assert belief.gravity_confidence > 0.5

    effects = tracker.compute_action_effects()
    assert len(effects) == 8
    assert effects[1].moves_player is True
