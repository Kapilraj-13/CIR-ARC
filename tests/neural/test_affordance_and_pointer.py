"""Unit tests for ObjectAffordanceHead and TwoStagePointerHead."""

from __future__ import annotations

import pytest
import torch

from cir_arc.neural.world_state import AFFORDANCE_NAMES
from cir_arc.neural.perception.affordance_head import ObjectAffordanceHead
from cir_arc.neural.perception.pointer_head import TwoStagePointerHead


def test_object_affordance_head_forward():
    head = ObjectAffordanceHead(slot_dim=128)
    slots = torch.randn(4, 24, 128)

    logits = head(slots, return_probs=False)
    assert logits.shape == (4, 24, 9)

    probs = head(slots, return_probs=True)
    assert probs.shape == (4, 24, 9)
    assert float(probs.detach().min()) >= 0.0
    assert float(probs.detach().max()) <= 1.0


def test_object_affordance_dict():
    head = ObjectAffordanceHead(slot_dim=128)
    slot_vec = torch.randn(128)

    aff_dict = head.predict_affordance_dict(slot_vec)
    assert len(aff_dict) == 9
    for name in AFFORDANCE_NAMES:
        assert name in aff_dict
        assert 0.0 <= aff_dict[name] <= 1.0


def test_two_stage_pointer_head_resolution():
    head = TwoStagePointerHead(slot_dim=128, feat_dim=128)
    slots = torch.randn(2, 24, 128)
    spatial_tokens = torch.randn(2, 100, 128)  # 10x10

    out = head(slots, spatial_tokens, H=10, W=10)

    assert "slot_logits" in out
    assert "selected_slot" in out
    assert "pixel_heatmap" in out
    assert "coords_pixel" in out
    assert "coords_xy" in out

    assert out["slot_logits"].shape == (2, 24)
    assert out["selected_slot"].shape == (2,)
    assert out["pixel_heatmap"].shape == (2, 10, 10)
    assert out["coords_pixel"].shape == (2, 2)
    assert out["coords_xy"].shape == (2, 2)


def test_two_stage_pointer_explicit_slot():
    head = TwoStagePointerHead(slot_dim=128, feat_dim=128)
    slots = torch.randn(2, 24, 128)
    spatial_tokens = torch.randn(2, 100, 128)

    explicit_slot = torch.tensor([5, 12])
    out = head(slots, spatial_tokens, H=10, W=10, target_slot_idx=explicit_slot)
    assert (out["selected_slot"] == explicit_slot).all()
