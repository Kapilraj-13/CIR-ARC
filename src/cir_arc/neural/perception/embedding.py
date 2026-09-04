"""Color embedding module for CIR-ARC discrete grid representation.

Maps discrete ARC color values (0-9) and padding/mask token (10) to 48-dimensional dense vectors.
"""

from typing import Optional
import torch
import torch.nn as nn


class ColorEmbedding(nn.Module):
    """Embeds discrete ARC color indices into continuous vector representations.

    ARC grids contain colors 0-9, with color 0 typically representing background.
    An extra token (index 10) is reserved for spatial masking and padding.

    Args:
        num_colors: Total number of discrete color tokens (0-9 plus mask token 10 = 11).
        embed_dim: Dimensionality of each color embedding vector (default: 48).
    """

    def __init__(self, num_colors: int = 11, embed_dim: int = 48) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=num_colors, embedding_dim=embed_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Embed discrete grid of colors into dense feature vectors.

        Args:
            grid: Tensor of shape (B, H, W) containing integer color indices in [0, num_colors - 1].

        Returns:
            FloatTensor of shape (B, H, W, embed_dim) with embedded color features.
        """
        if not torch.is_floating_point(grid):
            grid = grid.long()
        else:
            grid = grid.to(torch.long)
        grid = torch.clamp(grid, 0, self.num_colors - 1)
        return self.embedding(grid)


if __name__ == "__main__":
    print("Running ColorEmbedding smoke test...")
    model = ColorEmbedding()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"ColorEmbedding parameter count: {param_count}")
    assert param_count == 528, f"Expected 528 parameters, got {param_count}"

    test_shapes = [(4, 10, 10), (1, 5, 5), (2, 30, 30)]
    for shape in test_shapes:
        dummy_input = torch.randint(0, 11, shape, dtype=torch.long)
        out = model(dummy_input)
        expected_shape = shape + (48,)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
        assert out.dtype == torch.float32, f"Expected float32 dtype, got {out.dtype}"
        print(f"Passed test for input shape {shape} -> output shape {out.shape}")

    print("All ColorEmbedding smoke tests passed successfully!")
