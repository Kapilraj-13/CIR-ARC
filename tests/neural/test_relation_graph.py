"""Unit and integration tests for Relational Graph Head and Ground-Truth Relation Extractor."""

import numpy as np
import pytest
import torch

from cir_arc.core.objects import ArcObject
from cir_arc.neural.perception.relation_graph import RelationalGraphHead, extract_ground_truth_relations
from cir_arc.neural.world_state import RELATION_TYPES, NUM_RELATIONS, RelationGraph


def test_relation_graph_head_forward():
    """Verify RelationalGraphHead output shape and property ranges."""
    head = RelationalGraphHead(slot_dim=128, hidden_dim=64, num_relations=14)
    slots = torch.randn(2, 24, 128)
    objectness = torch.sigmoid(torch.randn(2, 24))

    logits = head(slots, objectness=objectness)
    assert logits.shape == (2, 24, 24, 14)

    # Test predict_graph
    graphs = head.predict_graph(slots, objectness=objectness, threshold=0.5, obj_threshold=0.3)
    assert len(graphs) == 2
    assert isinstance(graphs[0], RelationGraph)
    assert graphs[0].adj_matrix.shape == (24, 24, 14)


def test_extract_ground_truth_relations():
    """Verify ground-truth relation extractor on geometric test layout."""
    # Object A: red square at top-left (rows 0-1, cols 0-1)
    pixels_a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    obj_a = ArcObject(color=2, pixels=pixels_a, connectivity=4)

    # Object B: blue square to the right of A (rows 0-1, cols 10-11)
    pixels_b = np.array([[0, 10], [0, 11], [1, 10], [1, 11]])
    obj_b = ArcObject(color=1, pixels=pixels_b, connectivity=4)

    # Object C: red square touching A below (rows 2-3, cols 0-1)
    pixels_c = np.array([[2, 0], [2, 1], [3, 0], [3, 1]])
    obj_c = ArcObject(color=2, pixels=pixels_c, connectivity=4)

    gt_mat = extract_ground_truth_relations([obj_a, obj_b, obj_c], H=15, W=15)
    assert gt_mat.shape == (3, 3, NUM_RELATIONS)

    # Test A -> B relations
    # 0: LEFT_OF
    assert gt_mat[0, 1, 0] == 1.0
    # 1: RIGHT_OF
    assert gt_mat[1, 0, 1] == 1.0
    # 12: ALIGNED_X (same row)
    assert gt_mat[0, 1, 12] == 1.0

    # Test A -> C relations
    # 2: ABOVE
    assert gt_mat[0, 2, 2] == 1.0
    # 3: BELOW
    assert gt_mat[2, 0, 3] == 1.0
    # 4: TOUCHING (row 1 and row 2 share adjacent pixels)
    assert gt_mat[0, 2, 4] == 1.0
    # 10: SAME_COLOR (both color=2)
    assert gt_mat[0, 2, 10] == 1.0
    # 13: ALIGNED_Y (same column)
    assert gt_mat[0, 2, 13] == 1.0
