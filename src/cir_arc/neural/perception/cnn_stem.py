"""CNN stem and Multi-Scale CNN stem modules for CIR-ARC spatial feature encoding.

Contains:
1. CNNStem: Canonical 3-layer resolution-preserving encoder (54,848 parameters):
   - Conv2d(48, 64, 3, padding=1, bias=False) + GroupNorm(8, 64) + GELU
   - DepthwiseSeparableConv(64, 128) + GroupNorm(8, 128) + GELU
   - DepthwiseSeparableConv(128, 128) + GroupNorm(8, 128) + GELU
2. MultiScaleCNNStem: 4-stage hierarchical multi-scale encoder with auxiliary boundary & objectness heads.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable 2D convolution with GroupNorm and GELU activation."""

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
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class CNNStem(nn.Module):
    """3-layer spatial encoder preserving full grid resolution without downsampling (54,848 params).

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
        """Forward pass transforming channels-last embedding (B, H, W, 48) to (B, H*W, 128)."""
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)

        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return tokens


class ResidualDepthwiseSeparableConv(nn.Module):
    """Residual Depthwise separable 2D convolution with GroupNorm and GELU."""

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

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.act(out + res)
        return out


class MultiScaleCNNStem(nn.Module):
    """Hierarchical multi-scale spatial encoder with auxiliary boundary & objectness heads."""

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

        # Stage 1: Local features (48 -> 64)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels)
        self.act1 = nn.GELU()

        # Stage 2: Medium-range features (64 -> 96)
        self.stage2 = ResidualDepthwiseSeparableConv(
            in_channels=hidden_channels,
            out_channels=96,
            num_groups=num_groups,
        )

        # Stage 3: Broad context features (96 -> 128)
        self.stage3 = ResidualDepthwiseSeparableConv(
            in_channels=96,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        # Stage 4: Global context features (128 -> 128)
        self.stage4 = ResidualDepthwiseSeparableConv(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        # Multi-scale feature fusion: (64 + 96 + 128 + 128 = 416 -> out_channels)
        fused_in_dim = hidden_channels + 96 + out_channels + out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_in_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.GELU(),
        )

        # Auxiliary Head 1: Object boundary prediction map (B, 1, H, W)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Auxiliary Head 2: Cell-level objectness / foreground map (B, 1, H, W)
        self.cell_objectness_head = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = x.permute(0, 3, 1, 2).contiguous()

        f1 = self.act1(self.norm1(self.conv1(x)))
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        f_cat = torch.cat([f1, f2, f3, f4], dim=1)
        f_fused = self.fusion(f_cat)

        boundary_map = self.boundary_head(f_fused)
        cell_objectness = self.cell_objectness_head(f_fused)

        B, C, H, W = f_fused.shape
        tokens = f_fused.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if return_maps:
            return tokens, boundary_map, cell_objectness
        return tokens
