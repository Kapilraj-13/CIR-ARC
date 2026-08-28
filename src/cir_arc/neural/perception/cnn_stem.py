"""CNN stem module for CIR-ARC spatial feature encoding.

3-layer resolution-preserving encoder:
1. Conv2d(48, 64, 3, padding=1, bias=False) + GroupNorm(8, 64) + GELU
2. DepthwiseSeparableConv(64, 128) + GroupNorm(8, 128) + GELU
3. DepthwiseSeparableConv(128, 128) + GroupNorm(8, 128) + GELU

Transforms channels-last embedded grid (B, H, W, 48) into flattened spatial tokens (B, H*W, 128).
"""

from typing import Optional
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable 2D convolution with GroupNorm and GELU activation.

    Applies depthwise spatial convolution (groups=in_channels, bias=False)
    followed by pointwise 1x1 convolution (bias=True), GroupNorm(8, out_channels),
    and GELU activation.
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for depthwise separable conv block."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class CNNStem(nn.Module):
    """3-layer spatial encoder preserving full grid resolution without downsampling.

    Processes channels-last embeddings (B, H, W, in_channels) through resolution-preserving
    convolutions with GroupNorm and GELU activations, outputting flattened spatial tokens
    (B, H*W, out_channels).

    Args:
        in_channels: Input embedding dimension (default: 48).
        hidden_channels: Intermediate channel dimension (default: 64).
        out_channels: Output feature token dimension (default: 128).
        num_groups: Number of groups for GroupNorm (default: 8).
    """

    def __init__(
        self,
        in_channels: int = 48,
        hidden_channels: int = 64,
        out_channels: int = 128,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Layer 1: Standard 3x3 conv with bias=False + GroupNorm + GELU
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels)
        self.act1 = nn.GELU()

        # Layer 2: Depthwise separable conv 64 -> 128
        self.conv2 = DepthwiseSeparableConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        # Layer 3: Depthwise separable conv 128 -> 128
        self.conv3 = DepthwiseSeparableConv(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of spatial CNN encoder.

        Args:
            x: Input tensor of shape (B, H, W, in_channels) in channels-last format.

        Returns:
            Flattened spatial tokens of shape (B, H*W, out_channels).
        """
        B, H, W, C = x.shape
        # Permute (B, H, W, C) -> (B, C, H, W) for standard PyTorch Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()

        # Layer 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        # Layer 2
        x = self.conv2(x)

        # Layer 3
        x = self.conv3(x)

        # Permute back (B, out_channels, H, W) -> (B, H, W, out_channels)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Flatten spatial dimensions: (B, H*W, out_channels)
        x = x.reshape(B, H * W, self.out_channels)
        return x


if __name__ == "__main__":
    print("Running CNNStem smoke test...")
    model = CNNStem()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"CNNStem parameter count: {param_count}")
    assert param_count == 54848, f"Expected 54,848 parameters, got {param_count}"

    test_cases = [
        ((4, 10, 10, 48), (4, 100, 128)),
        ((1, 5, 5, 48), (1, 25, 128)),
        ((2, 8, 12, 48), (2, 96, 128)),
        ((2, 30, 30, 48), (2, 900, 128)),
    ]

    for in_shape, expected_out_shape in test_cases:
        dummy_input = torch.randn(in_shape)
        out = model(dummy_input)
        assert out.shape == expected_out_shape, f"Expected {expected_out_shape}, got {out.shape}"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
        print(f"Passed test for {in_shape} -> {out.shape}")

    print("All CNNStem smoke tests passed successfully!")
