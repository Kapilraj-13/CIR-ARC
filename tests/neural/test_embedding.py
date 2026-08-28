"""Unit tests for ColorEmbedding module (Phase 2)."""

import pytest
import torch
from cir_arc.neural.perception.embedding import ColorEmbedding


def test_color_embedding_instantiation_and_parameter_count():
    """Verify ColorEmbedding initializes with 11 colors, 48 dims, and 528 parameters."""
    emb = ColorEmbedding(num_colors=11, embed_dim=48)
    assert isinstance(emb, torch.nn.Module)
    total_params = sum(p.numel() for p in emb.parameters())
    assert total_params == 11 * 48
    assert total_params == 528


@pytest.mark.parametrize("batch_size,height,width", [
    (1, 1, 1),
    (1, 5, 5),
    (2, 8, 12),
    (4, 10, 10),
    (2, 15, 15),
    (1, 30, 30),
])
def test_color_embedding_output_shapes(batch_size, height, width):
    """Verify ColorEmbedding outputs (B, H, W, 48) for various grid dimensions."""
    emb = ColorEmbedding()
    grid = torch.randint(0, 10, (batch_size, height, width), dtype=torch.long)
    output = emb(grid)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, height, width, 48)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_color_embedding_mask_token_handling():
    """Verify mask token ID=10 is handled correctly without index errors."""
    emb = ColorEmbedding()
    # Create a grid where some positions are regular colors (0-9) and some are mask token (10)
    grid = torch.tensor([
        [[0, 1, 2], [3, 4, 5], [10, 10, 10]],
        [[10, 0, 10], [9, 8, 7], [10, 10, 10]],
    ], dtype=torch.long)

    output = emb(grid)
    assert output.shape == (2, 3, 3, 48)
    assert torch.isfinite(output).all()

    # The embeddings for token 10 should be identical wherever token 10 appears
    mask_emb_1 = output[0, 2, 0]
    mask_emb_2 = output[1, 0, 0]
    assert torch.allclose(mask_emb_1, mask_emb_2)


def test_color_embedding_all_colors_valid():
    """Verify all ARC colors 0-9 plus mask token 10 produce valid continuous vectors."""
    emb = ColorEmbedding()
    all_colors = torch.arange(11, dtype=torch.long).unsqueeze(0).unsqueeze(0)  # (1, 1, 11)
    output = emb(all_colors)

    assert output.shape == (1, 1, 11, 48)
    assert torch.isfinite(output).all()

    # Ensure different color indices have different initializations
    for i in range(11):
        for j in range(i + 1, 11):
            assert not torch.allclose(output[0, 0, i], output[0, 0, j], atol=1e-4)


def test_color_embedding_gradient_flow():
    """Verify backpropagation produces valid non-zero gradients through embedding."""
    emb = ColorEmbedding()
    grid = torch.randint(0, 11, (2, 6, 6), dtype=torch.long)
    output = emb(grid)

    loss = output.sum()
    loss.backward()

    assert emb.embedding.weight.grad is not None
    assert torch.isfinite(emb.embedding.weight.grad).all()
    assert (emb.embedding.weight.grad != 0).any()


def test_color_embedding_deterministic_lookup():
    """Verify identical color integers at different spatial locations produce identical vectors."""
    emb = ColorEmbedding()
    grid = torch.zeros((1, 4, 4), dtype=torch.long)  # All background color 0
    output = emb(grid)

    first_elem = output[0, 0, 0]
    for r in range(4):
        for c in range(4):
            assert torch.allclose(output[0, r, c], first_elem)
