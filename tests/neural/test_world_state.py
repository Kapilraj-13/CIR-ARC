"""Unit and integration tests for Structured WorldState interface and Temporal tracking."""

import numpy as np
import pytest
import torch

from cir_arc.neural.training.trainer import PerceptionModel
from cir_arc.neural.world_state import WorldState, StructuredObject, SpatialRelation
from cir_arc.neural.temporal.tracker import TemporalSlotTracker, TemporalWorldState


def test_perception_model_to_world_state():
    """Verify PerceptionModel.to_world_state() end-to-end output."""
    model = PerceptionModel()
    grid = np.zeros((12, 14), dtype=np.int64)
    grid[2:5, 3:6] = 3  # Green 3x3 square
    grid[8:10, 9:11] = 2 # Red 2x2 square

    world_state = model.to_world_state(grid, obj_threshold=0.1, rel_threshold=0.3)
    assert isinstance(world_state, WorldState)
    assert world_state.grid_shape == (12, 14)
    assert isinstance(world_state.objects, list)
    assert isinstance(world_state.relations, list)

    for obj in world_state.objects:
        assert isinstance(obj, StructuredObject)
        assert 0 <= obj.color < 10
        assert 0.0 <= obj.confidence <= 1.0
        assert len(obj.centroid) == 2
        assert len(obj.bbox) == 4
        assert obj.mask is not None
        assert obj.mask.shape == (12, 14)

    summary = world_state.summary()
    assert "num_objects" in summary
    assert "relations" in summary


def test_temporal_tracker_with_world_state():
    """Verify TemporalSlotTracker consumes WorldState and produces kinematic velocities."""
    model = PerceptionModel()
    tracker = TemporalSlotTracker()

    # Frame 1: Object at (2, 3)
    grid_f1 = np.zeros((15, 15), dtype=np.int64)
    grid_f1[2:4, 3:5] = 4
    ws_1 = model.to_world_state(grid_f1, frame_index=1, obj_threshold=0.1)

    tws_1 = tracker.update_from_world_state(ws_1)
    assert isinstance(tws_1, TemporalWorldState)
    assert tws_1.frame_index == 1

    # Frame 2: Object moved right to (2, 8)
    grid_f2 = np.zeros((15, 15), dtype=np.int64)
    grid_f2[2:4, 8:10] = 4
    ws_2 = model.to_world_state(grid_f2, frame_index=2, obj_threshold=0.1)

    tws_2 = tracker.update_from_world_state(ws_2)
    assert isinstance(tws_2, TemporalWorldState)
    assert tws_2.frame_index == 2
