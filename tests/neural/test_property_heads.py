"""Unit tests for PropertyHeads module (Phase 2.5)."""

import pytest
import torch
from cir_arc.neural.perception.property_heads import PropertyHeads


def test_property_heads_instantiation_and_parameter_count():
    """Verify PropertyHeads initializes with all property heads."""
    heads = PropertyHeads(slot_dim=128)
    assert isinstance(heads, torch.nn.Module)
    total_params = sum(p.numel() for p in heads.parameters())
    assert 90_000 <= total_params <= 120_000


@pytest.mark.parametrize("batch_size,num_slots", [
    (1, 1),
    (2, 10),
    (4, 24),
    (1, 32),
])
def test_property_heads_output_keys_and_shapes(batch_size, num_slots):
    """Verify PropertyHeads returns dict with all required keys and correct tensor shapes."""
    heads = PropertyHeads(slot_dim=128)
    slots = torch.randn(batch_size, num_slots, 128)
    props = heads(slots)

    assert isinstance(props, dict)
    expected_keys = {
        "color", "shape", "size", "position", "bbox", "dimensions",
        "aspect_ratio", "orientation", "symmetry", "holes", "identity", "presence"
    }
    assert expected_keys.issubset(props.keys())

    assert props["color"].shape == (batch_size, num_slots, 10)
    assert props["shape"].shape == (batch_size, num_slots, 8)
    assert props["position"].shape == (batch_size, num_slots, 2)
    assert props["bbox"].shape == (batch_size, num_slots, 4)
    assert props["dimensions"].shape == (batch_size, num_slots, 4)
    assert props["orientation"].shape == (batch_size, num_slots, 4)
    assert props["symmetry"].shape == (batch_size, num_slots, 4)
    assert props["identity"].shape == (batch_size, num_slots, 64)


def test_property_heads_sigmoid_ranges():
    """Verify bounded heads output values strictly in [0.0, 1.0]."""
    heads = PropertyHeads(slot_dim=128)
    slots = torch.randn(4, 24, 128)
    props = heads(slots)

    for k in ["position", "size", "bbox", "dimensions", "aspect_ratio", "holes", "presence"]:
        assert (props[k] >= 0.0).all(), f"Negative values in {k}"
        assert (props[k] <= 1.0).all(), f"Values > 1.0 in {k}"


def test_property_heads_logits_finite():
    """Verify all discrete logits (color, shape, orientation, symmetry) are finite."""
    heads = PropertyHeads(slot_dim=128)
    slots = torch.randn(2, 24, 128)
    props = heads(slots)

    for key in ["color", "shape", "orientation", "symmetry", "identity"]:
        assert torch.isfinite(props[key]).all(), f"Non-finite values in {key}"


def test_property_heads_gradient_flow():
    """Verify gradients propagate to input slots and through all MLP heads."""
    heads = PropertyHeads(slot_dim=128)
    slots = torch.randn(2, 24, 128, requires_grad=True)
    props = heads(slots)

    loss = (
        props["color"].sum()
        + props["shape"].sum()
        + props["size"].sum()
        + props["position"].sum()
        + props["bbox"].sum()
        + props["dimensions"].sum()
        + props["aspect_ratio"].sum()
        + props["orientation"].sum()
        + props["symmetry"].sum()
        + props["holes"].sum()
        + props["identity"].sum()
        + props["presence"].sum()
    )
    loss.backward()

    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()

    for name, param in heads.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter {name}"


def test_property_heads_batch_independence():
    """Verify predictions for item 0 are independent of item 1 in a batch."""
    heads = PropertyHeads(slot_dim=128)
    slots1 = torch.randn(1, 24, 128)
    slots2 = torch.randn(1, 24, 128)
    batch_slots = torch.cat([slots1, slots2], dim=0)

    props_individual1 = heads(slots1)
    props_batch = heads(batch_slots)

    for k in ["color", "position", "bbox", "dimensions", "identity"]:
        assert torch.allclose(props_individual1[k][0], props_batch[k][0], atol=1e-5)
