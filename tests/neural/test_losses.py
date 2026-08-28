"""Unit tests for neural loss functions and Hungarian matching (Phase 2)."""

import pytest
import torch
import numpy as np

from cir_arc.core.objects import ArcObject

# Safe import for progressive testability during milestone builds
matching = pytest.importorskip("cir_arc.neural.losses.matching")
reconstruction = pytest.importorskip("cir_arc.neural.losses.reconstruction")
property_loss_mod = pytest.importorskip("cir_arc.neural.losses.property")
diversity = pytest.importorskip("cir_arc.neural.losses.diversity")

hungarian_matching = matching.hungarian_matching
reconstruction_loss = reconstruction.reconstruction_loss
color_loss = property_loss_mod.color_loss
position_loss = property_loss_mod.position_loss
size_loss = property_loss_mod.size_loss
objectness_loss = property_loss_mod.objectness_loss
compute_property_losses = property_loss_mod.compute_property_losses
diversity_loss = diversity.diversity_loss
objectness_sparsity_loss = diversity.objectness_sparsity_loss


def _create_dummy_arc_object(color: int, pixels_list, H: int = 10, W: int = 10) -> ArcObject:
    """Helper to construct a valid ArcObject instance."""
    pixels = np.array(pixels_list, dtype=np.int64)
    return ArcObject(color=color, pixels=pixels)


def test_hungarian_matching_bijective():
    """Verify Hungarian matching produces a valid 1-to-1 bijection between slots and GT objects."""
    K = 24
    H, W = 10, 10
    gt_objects = [
        _create_dummy_arc_object(1, [(0, 0), (0, 1)], H, W),
        _create_dummy_arc_object(2, [(5, 5), (5, 6), (6, 5)], H, W),
        _create_dummy_arc_object(3, [(8, 8)], H, W),
    ]

    pred_props = {
        "color": torch.randn(K, 10),
        "position": torch.rand(K, 2),
        "size": torch.rand(K, 1),
    }

    matches = hungarian_matching(pred_props, gt_objects, H=H, W=W)

    assert isinstance(matches, list)
    assert len(matches) == len(gt_objects)

    matched_slots = [p[0] for p in matches]
    matched_gts = [p[1] for p in matches]

    # Check bijection: no duplicate slot indices and no duplicate GT indices
    assert len(matched_slots) == len(set(matched_slots))
    assert len(matched_gts) == len(set(matched_gts))

    for slot_idx, gt_idx in matches:
        assert 0 <= slot_idx < K
        assert 0 <= gt_idx < len(gt_objects)


def test_hungarian_matching_empty_objects():
    """Verify Hungarian matching gracefully handles empty ground truth (M=0)."""
    K = 24
    pred_props = {
        "color": torch.randn(K, 10),
        "position": torch.rand(K, 2),
    }

    matches = hungarian_matching(pred_props, [], H=10, W=10)
    assert matches == []


def test_hungarian_matching_exact_preference():
    """Verify Hungarian matching correctly pairs a slot whose predictions match a GT object."""
    K = 24
    H, W = 10, 10
    gt_obj = _create_dummy_arc_object(color=3, pixels_list=[(4, 4), (4, 5)], H=H, W=W)

    # Initialize random props
    color_logits = torch.zeros(K, 10)
    positions = torch.zeros(K, 2)

    # Slot 7 is crafted to strongly match gt_obj
    color_logits[7, 3] = 10.0  # High logit for color 3
    # Centroid for [(4,4), (4,5)] is row 4.0, col 4.5 -> normalized: (4.5/10, 5.0/10) = (0.45, 0.5)
    positions[7] = torch.tensor([0.45, 0.5])

    pred_props = {
        "color": color_logits,
        "position": positions,
    }

    matches = hungarian_matching(pred_props, [gt_obj], H=H, W=W)
    assert len(matches) == 1
    assert matches[0] == (7, 0)


def test_reconstruction_loss_masked_behavior():
    """Verify reconstruction loss masks out padded pixels."""
    B, H, W = 2, 8, 8
    pred_logits = torch.randn(B, H, W, 10, requires_grad=True)
    target_grid = torch.randint(0, 10, (B, H, W), dtype=torch.long)

    # Mask where only top-left 4x4 is valid
    mask = torch.zeros(B, H, W, dtype=torch.float32)
    mask[:, :4, :4] = 1.0

    loss = reconstruction_loss(pred_logits, target_grid, mask=mask)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert pred_logits.grad is not None
    # Gradients for masked-out region should be zero
    assert (pred_logits.grad[:, 4:, 4:, :] == 0).all()


def test_reconstruction_loss_perfect_accuracy():
    """Verify reconstruction loss is very small when logits strongly predict the target."""
    B, H, W = 1, 4, 4
    target = torch.randint(0, 10, (B, H, W), dtype=torch.long)
    pred_logits = torch.zeros(B, H, W, 10)

    # Put high confidence on correct color
    for r in range(H):
        for c in range(W):
            pred_logits[0, r, c, target[0, r, c]] = 20.0

    loss = reconstruction_loss(pred_logits, target)
    assert loss.item() < 1e-4


def test_property_losses_finite_and_non_negative():
    """Verify all individual and combined property losses are finite non-negative scalars."""
    K = 24
    H, W = 10, 10
    gt_objects = [
        _create_dummy_arc_object(2, [(1, 1), (1, 2)], H, W),
        _create_dummy_arc_object(5, [(7, 7)], H, W),
    ]
    matches = [(3, 0), (12, 1)]

    pred_props = {
        "color": torch.randn(K, 10, requires_grad=True),
        "position": torch.rand(K, 2, requires_grad=True),
        "size": torch.rand(K, 1, requires_grad=True),
    }
    objectness = torch.rand(K, requires_grad=True)

    c_loss = color_loss(pred_props["color"], gt_objects, matches)
    p_loss = position_loss(pred_props["position"], gt_objects, matches, H, W)
    s_loss = size_loss(pred_props["size"], gt_objects, matches, H, W)
    o_loss = objectness_loss(objectness, matches, total_slots=K)

    for l_val in [c_loss, p_loss, s_loss, o_loss]:
        assert isinstance(l_val, torch.Tensor)
        assert l_val.dim() == 0
        assert torch.isfinite(l_val)
        assert l_val.item() >= 0.0

    prop_dict = compute_property_losses(pred_props, objectness, gt_objects, matches, H, W)
    assert "color_loss" in prop_dict
    assert "pos_loss" in prop_dict
    assert "size_loss" in prop_dict
    assert "obj_loss" in prop_dict


def test_diversity_loss_orthogonal_vs_identical():
    """Verify diversity loss is strictly higher for identical slots than orthogonal slots."""
    B, K, D = 1, 4, 16

    # Orthogonal slots (canonical basis)
    ortho_slots = torch.zeros(B, K, D)
    for k in range(K):
        ortho_slots[0, k, k] = 1.0

    # Identical slots
    identical_slots = torch.ones(B, K, D)

    ortho_div_loss = diversity_loss(ortho_slots)
    ident_div_loss = diversity_loss(identical_slots)

    assert ortho_div_loss.item() < 1e-4
    assert ident_div_loss.item() > ortho_div_loss.item()


def test_sparsity_loss_l1_monotonic():
    """Verify sparsity loss increases monotonically with objectness activation magnitude."""
    low_obj = torch.full((2, 24), 0.1)
    high_obj = torch.full((2, 24), 0.9)

    loss_low = objectness_sparsity_loss(low_obj)
    loss_high = objectness_sparsity_loss(high_obj)

    assert loss_low.item() >= 0.0
    assert loss_high.item() > loss_low.item()


def test_reconstruction_loss_with_ignore_index_padding():
    """Verify reconstruction loss handles pad token 10 without out-of-bounds error."""
    B, H, W = 2, 6, 6
    pred_logits = torch.randn(B, H, W, 10, requires_grad=True)
    # Target grid contains 10 at padded locations
    target_grid = torch.full((B, H, W), 10, dtype=torch.long)
    target_grid[:, :3, :3] = torch.randint(0, 10, (B, 3, 3))
    mask = torch.zeros(B, H, W, dtype=torch.float32)
    mask[:, :3, :3] = 1.0

    loss = reconstruction_loss(pred_logits, target_grid, mask=mask, ignore_index=10)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_logits.grad is not None
    assert (pred_logits.grad[:, 3:, 3:, :] == 0).all()


def test_objectness_loss_batched_empty_and_valid_matches():
    """Verify objectness_loss handles batched (B, K) tensors with empty and valid matches."""
    B, K = 3, 24
    obj_2d = torch.rand(B, K, requires_grad=True)

    # Case 1: Empty matches list
    loss_empty = objectness_loss(obj_2d, matches=[])
    assert isinstance(loss_empty, torch.Tensor)
    assert loss_empty.dim() == 0
    assert torch.isfinite(loss_empty)
    loss_empty.backward()
    assert obj_2d.grad is not None

    # Case 2: Batched list of lists
    obj_2d.grad.zero_()
    matches_batched = [[(0, 0), (1, 1)], [], [(5, 0)]]
    loss_batched = objectness_loss(obj_2d, matches=matches_batched)
    assert isinstance(loss_batched, torch.Tensor)
    assert loss_batched.dim() == 0
    assert torch.isfinite(loss_batched)
    loss_batched.backward()
    assert obj_2d.grad is not None

