"""Unit tests for SlotAttention module (Phase 2)."""

import pytest
import torch
from cir_arc.neural.perception.slot_attention import SlotAttention


def test_slot_attention_instantiation_and_parameter_count():
    """Verify SlotAttention initializes with expected structure and parameter budget (~223.5K)."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)
    assert isinstance(sa, torch.nn.Module)
    total_params = sum(p.numel() for p in sa.parameters())
    assert total_params == 223489


@pytest.mark.parametrize("batch_size,num_tokens", [
    (1, 1),
    (2, 25),    # 5x5
    (1, 96),    # 8x12
    (4, 100),   # 10x10
    (2, 225),   # 15x15
    (1, 900),   # 30x30
])
def test_slot_attention_output_shapes(batch_size, num_tokens):
    """Verify SlotAttention outputs slots (B, 24, 128), objectness (B, 24), and attn_maps (B, 24, N)."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)
    inputs = torch.randn(batch_size, num_tokens, 128)
    slots, objectness, attn_maps = sa(inputs)

    assert slots.shape == (batch_size, 24, 128)
    assert objectness.shape == (batch_size, 24)
    assert attn_maps.shape == (batch_size, 24, num_tokens)

    assert torch.isfinite(slots).all()
    assert torch.isfinite(objectness).all()
    assert torch.isfinite(attn_maps).all()


def test_slot_attention_competitive_binding_invariant():
    """CRITICAL ACCEPTANCE INVARIANT: Attention maps must sum to 1.0 (+/- 1e-5) over slot dimension."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)

    for num_tokens in [25, 100, 225, 900]:
        inputs = torch.randn(2, num_tokens, 128)
        _, _, attn_maps = sa(inputs)

        # Sum over slot dimension (dim=1)
        slot_sum = attn_maps.sum(dim=1)  # (B, N)
        expected = torch.ones_like(slot_sum)

        max_diff = torch.max(torch.abs(slot_sum - expected)).item()
        assert max_diff < 1e-5, f"Competitive binding invariant violated! Max diff: {max_diff}"


def test_slot_attention_objectness_range():
    """Verify slot objectness values are strictly bounded in [0.0, 1.0]."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)
    inputs = torch.randn(4, 100, 128)
    _, objectness, _ = sa(inputs)

    assert (objectness >= 0.0).all()
    assert (objectness <= 1.0).all()


def test_slot_attention_masking_behavior():
    """Verify SlotAttention handles spatial masks for variable-length/padded sequences."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)
    batch_size = 2
    num_tokens = 100
    inputs = torch.randn(batch_size, num_tokens, 128)

    # Mask where first 50 tokens are valid (1.0) and last 50 are padding (0.0)
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.float32)
    mask[:, :50] = 1.0

    slots, objectness, attn_maps = sa(inputs, mask=mask)

    assert slots.shape == (batch_size, 24, 128)
    assert objectness.shape == (batch_size, 24)
    assert attn_maps.shape == (batch_size, 24, num_tokens)

    # Valid tokens should sum to 1.0 across slots
    valid_sum = attn_maps[:, :, :50].sum(dim=1)
    expected_valid = torch.ones_like(valid_sum)
    assert torch.allclose(valid_sum, expected_valid, atol=1e-5)


def test_slot_attention_gradient_flow():
    """Verify gradients propagate through iterative competitive binding and GRU update."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)
    inputs = torch.randn(2, 64, 128, requires_grad=True)
    slots, objectness, attn_maps = sa(inputs)

    loss = slots.sum() + objectness.sum() + attn_maps.sum()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()

    # Check key parameter gradients
    assert sa.slots_mu.grad is not None
    assert sa.slots_log_sigma.grad is not None
    assert sa.project_q.weight.grad is not None
    assert sa.gru.weight_ih.grad is not None


def test_slot_attention_slot_diversity():
    """Verify slots do not instantly collapse to identical representations across the 24 slots."""
    sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)
    inputs = torch.randn(1, 100, 128)
    slots, _, _ = sa(inputs)

    # Check that at least two slots differ
    slot0 = slots[0, 0]
    slot1 = slots[0, 1]
    assert not torch.allclose(slot0, slot1, atol=1e-3)
