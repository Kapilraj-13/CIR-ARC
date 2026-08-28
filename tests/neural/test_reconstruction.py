"""Unit tests for ReconstructionDecoder module (Phase 2)."""

import pytest
import torch
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder


def test_reconstruction_decoder_instantiation_and_parameter_count():
    """Verify ReconstructionDecoder initializes with expected parameter budget (~70.8K)."""
    decoder = ReconstructionDecoder(slot_dim=128, max_h=30, max_w=30, num_colors=10)
    assert isinstance(decoder, torch.nn.Module)
    total_params = sum(p.numel() for p in decoder.parameters())
    assert total_params == 70794


@pytest.mark.parametrize("batch_size,height,width", [
    (1, 1, 1),
    (2, 5, 5),
    (1, 8, 12),
    (4, 10, 10),
    (2, 15, 20),
    (1, 30, 30),
])
def test_reconstruction_decoder_variable_grid_sizes(batch_size, height, width):
    """Verify ReconstructionDecoder reconstructs (B, H, W, 10) color logits for various H, W."""
    decoder = ReconstructionDecoder(slot_dim=128, max_h=30, max_w=30, num_colors=10)
    slots = torch.randn(batch_size, 24, 128)
    objectness = torch.rand(batch_size, 24)

    logits = decoder(slots, objectness=objectness, H=height, W=width)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, height, width, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_reconstruction_decoder_gradient_flow():
    """Verify backpropagation produces valid gradients to slots and decoder parameters."""
    decoder = ReconstructionDecoder()
    slots = torch.randn(2, 24, 128, requires_grad=True)
    objectness = torch.rand(2, 24)

    logits = decoder(slots, objectness=objectness, H=8, W=8)
    loss = logits.sum()
    loss.backward()

    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()

    assert decoder.row_embed.weight.grad is not None
    assert decoder.col_embed.weight.grad is not None
    assert torch.isfinite(decoder.row_embed.weight.grad).all()
    assert torch.isfinite(decoder.col_embed.weight.grad).all()


def test_reconstruction_decoder_position_embedding_max_bounds():
    """Verify position embedding indexes up to max 30x30 without out-of-bounds exception."""
    decoder = ReconstructionDecoder(slot_dim=128, max_h=30, max_w=30)
    slots = torch.randn(1, 24, 128)
    objectness = torch.rand(1, 24)

    # Max allowed dimensions
    logits = decoder(slots, objectness=objectness, H=30, W=30)
    assert logits.shape == (1, 30, 30, 10)
    assert torch.isfinite(logits).all()


def test_reconstruction_decoder_deterministic_eval_mode():
    """Verify decoder produces deterministic output in eval mode."""
    decoder = ReconstructionDecoder()
    decoder.eval()
    slots = torch.randn(2, 24, 128)
    objectness = torch.rand(2, 24)

    with torch.no_grad():
        out1 = decoder(slots, objectness=objectness, H=6, W=6)
        out2 = decoder(slots, objectness=objectness, H=6, W=6)

    assert torch.allclose(out1, out2)
