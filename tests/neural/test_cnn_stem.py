"""Unit tests for CNNStem module (Phase 2)."""

import pytest
import torch
from cir_arc.neural.perception.cnn_stem import CNNStem


def test_cnn_stem_instantiation_and_parameter_count():
    """Verify CNNStem initializes with expected structure and parameter budget (~54.8K)."""
    stem = CNNStem(in_channels=48, hidden_channels=64, out_channels=128)
    assert isinstance(stem, torch.nn.Module)
    total_params = sum(p.numel() for p in stem.parameters())
    # 54,848 parameters as specified in architecture design
    assert total_params == 54848


@pytest.mark.parametrize("batch_size,height,width", [
    (1, 1, 1),
    (2, 5, 5),
    (1, 8, 12),
    (4, 10, 10),
    (2, 15, 15),
    (1, 30, 30),
])
def test_cnn_stem_output_shapes(batch_size, height, width):
    """Verify CNNStem outputs (B, H*W, 128) for various spatial dimensions."""
    stem = CNNStem()
    x = torch.randn(batch_size, height, width, 48, dtype=torch.float32)
    tokens = stem(x)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (batch_size, height * width, 128)
    assert tokens.dtype == torch.float32
    assert torch.isfinite(tokens).all()


def test_cnn_stem_resolution_preservation():
    """Verify CNNStem strictly preserves spatial resolution without pooling or downsampling."""
    stem = CNNStem()
    h, w = 7, 11
    x = torch.randn(2, h, w, 48)
    tokens = stem(x)

    assert tokens.shape[1] == h * w


def test_cnn_stem_single_pixel_grid():
    """Boundary test: 1x1 grid should be processed successfully."""
    stem = CNNStem()
    x = torch.randn(1, 1, 1, 48)
    tokens = stem(x)

    assert tokens.shape == (1, 1, 128)
    assert torch.isfinite(tokens).all()


def test_cnn_stem_gradient_flow():
    """Verify gradient flow through all convolutional layers and normalizations."""
    stem = CNNStem()
    x = torch.randn(2, 6, 6, 48, requires_grad=True)
    tokens = stem(x)

    loss = tokens.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Check all parameters have gradients
    for name, param in stem.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter {name}"


def test_cnn_stem_deterministic_eval_mode():
    """Verify consistent outputs in eval mode."""
    stem = CNNStem()
    stem.eval()
    x = torch.randn(2, 8, 8, 48)

    with torch.no_grad():
        out1 = stem(x)
        out2 = stem(x)

    assert torch.allclose(out1, out2)
