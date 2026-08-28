"""Unit tests for perception evaluation metrics (Phase 2)."""

import pytest
import torch
import numpy as np

from cir_arc.core.objects import ArcObject

# Safe import for progressive testability during milestone builds
metrics_mod = pytest.importorskip("cir_arc.neural.evaluation.perception_metrics")
reconstruction_accuracy = metrics_mod.reconstruction_accuracy
object_detection_f1 = metrics_mod.object_detection_f1
color_accuracy = metrics_mod.color_accuracy
position_mae = metrics_mod.position_mae
size_mae = metrics_mod.size_mae
compute_perception_metrics = metrics_mod.compute_perception_metrics


def _create_dummy_arc_object(color: int, pixels_list) -> ArcObject:
    pixels = np.array(pixels_list, dtype=np.int64)
    return ArcObject(color=color, pixels=pixels)


def test_reconstruction_accuracy_perfect_and_zero():
    """Verify reconstruction accuracy gives 1.0 on perfect predictions and 0.0 on completely wrong ones."""
    B, H, W = 2, 4, 4
    target = torch.randint(0, 10, (B, H, W), dtype=torch.long)

    # Perfect prediction: argmax matches target everywhere
    perfect_logits = torch.zeros(B, H, W, 10)
    for b in range(B):
        for r in range(H):
            for c in range(W):
                perfect_logits[b, r, c, target[b, r, c]] = 10.0

    acc_perfect = reconstruction_accuracy(perfect_logits, target)
    assert acc_perfect == pytest.approx(1.0, abs=1e-5)

    # Completely wrong prediction: argmax is (target + 1) % 10
    wrong_logits = torch.zeros(B, H, W, 10)
    for b in range(B):
        for r in range(H):
            for c in range(W):
                wrong_logits[b, r, c, (target[b, r, c] + 1) % 10] = 10.0

    acc_zero = reconstruction_accuracy(wrong_logits, target)
    assert acc_zero == pytest.approx(0.0, abs=1e-5)


def test_reconstruction_accuracy_with_mask():
    """Verify masked cells do not affect reconstruction accuracy calculation."""
    B, H, W = 1, 4, 4
    target = torch.zeros((B, H, W), dtype=torch.long)
    logits = torch.zeros((B, H, W, 10))

    # Make top row (0, :4) correct (color 0)
    logits[0, 0, :, 0] = 10.0
    # Make rest wrong (color 1)
    logits[0, 1:, :, 1] = 10.0

    # Mask only top row
    mask = torch.zeros((B, H, W), dtype=torch.float32)
    mask[0, 0, :] = 1.0

    acc_masked = reconstruction_accuracy(logits, target, mask=mask)
    assert acc_masked == pytest.approx(1.0, abs=1e-5)


def test_object_detection_f1_scores():
    """Verify object detection F1 on exact, partial, and zero detections."""
    # Case 1: Perfect match (2 active slots, 2 GT objects)
    obj_perfect = torch.tensor([[0.9, 0.8, 0.1, 0.2]])  # 2 active
    f1_perfect = object_detection_f1(obj_perfect, gt_counts=[2], threshold=0.5)
    assert f1_perfect == pytest.approx(1.0, abs=1e-5)

    # Case 2: Zero active slots with 2 GT objects
    obj_zero = torch.tensor([[0.1, 0.2, 0.1, 0.2]])  # 0 active
    f1_zero = object_detection_f1(obj_zero, gt_counts=[2], threshold=0.5)
    assert f1_zero == pytest.approx(0.0, abs=1e-5)

    # Case 3: Over-segmentation (4 active slots, 2 GT objects)
    obj_over = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    # Precision = 2/4 = 0.5, Recall = 2/2 = 1.0 -> F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 1.0 / 1.5 = 0.6667
    f1_over = object_detection_f1(obj_over, gt_counts=[2], threshold=0.5)
    assert f1_over == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_color_accuracy_matched_pairs():
    """Verify color prediction accuracy on Hungarian-matched slot pairs."""
    gt_objs = [
        _create_dummy_arc_object(color=2, pixels_list=[(0, 0)]),
        _create_dummy_arc_object(color=7, pixels_list=[(1, 1)]),
    ]
    matches = [(0, 0), (1, 1)]

    # Perfect color logits
    pred_colors = torch.zeros(2, 10)
    pred_colors[0, 2] = 10.0  # Slot 0 predicts color 2
    pred_colors[1, 7] = 10.0  # Slot 1 predicts color 7

    acc = color_accuracy(pred_colors, gt_objs, matches)
    assert acc == pytest.approx(1.0, abs=1e-5)

    # One wrong
    pred_colors[1, 7] = 0.0
    pred_colors[1, 3] = 10.0  # Slot 1 predicts color 3 instead of 7
    acc_partial = color_accuracy(pred_colors, gt_objs, matches)
    assert acc_partial == pytest.approx(0.5, abs=1e-5)


def test_position_and_size_mae_perfect_vs_deviated():
    """Verify position MAE and size MAE calculate zero on exact predictions and positive on error."""
    H, W = 10, 10
    # Object: size 2 pixels, centroid (row 1.0, col 1.5) -> normalized (1.5/10, 2.0/10) = (0.15, 0.2)
    # Normalized size = 2 / 100 = 0.02
    gt_objs = [_create_dummy_arc_object(color=1, pixels_list=[(1, 1), (1, 2)])]
    matches = [(0, 0)]

    pred_pos_perfect = torch.tensor([[0.15, 0.2]])
    pred_size_perfect = torch.tensor([[0.02]])

    p_mae = position_mae(pred_pos_perfect, gt_objs, matches, H=H, W=W)
    s_mae = size_mae(pred_size_perfect, gt_objs, matches, H=H, W=W)

    assert p_mae == pytest.approx(0.0, abs=1e-4)
    assert s_mae == pytest.approx(0.0, abs=1e-4)


def test_compute_perception_metrics_summary():
    """Verify compute_perception_metrics aggregates all individual metrics into a valid dictionary."""
    B, H, W = 1, 8, 8
    pred_logits = torch.randn(B, H, W, 10)
    target_grid = torch.randint(0, 10, (B, H, W), dtype=torch.long)
    objectness = torch.rand(B, 24)
    gt_objs = [[_create_dummy_arc_object(color=1, pixels_list=[(2, 2)])]]

    pred_props = {
        "color": torch.randn(24, 10),
        "position": torch.rand(24, 2),
        "size": torch.rand(24, 1),
    }

    metrics = compute_perception_metrics(
        pred_logits=pred_logits,
        target_grid=target_grid,
        objectness=objectness,
        pred_props=pred_props,
        gt_objects_batch=gt_objs,
        heights=[H],
        widths=[W],
    )

    assert isinstance(metrics, dict)
    for key in ["recon_acc", "object_f1", "color_acc", "pos_mae", "size_mae"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
