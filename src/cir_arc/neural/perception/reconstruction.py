"""Reconstruction decoder module for ARC grids.

Decodes object slots back into full 2D grid discrete color distributions:
- Learned 1D row and column positional embeddings (30, 64) concatenated into 128-dim coordinate queries
- Cross-attention from spatial cell coordinate queries to object slots
- Lightweight MLP (128 -> 128 -> 10) predicting 10-class discrete color logits
- Dynamic support for arbitrary grid dimensions up to 30x30
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionDecoder(nn.Module):
    """Decodes object slots into full spatial grid color logits via cross-attention.

    Args:
        slot_dim: Dimensionality of input slot vectors (default: 128).
        max_h: Maximum grid height supported (default: 30).
        max_w: Maximum grid width supported (default: 30).
        num_colors: Number of target color classes (default: 10).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        max_h: int = 30,
        max_w: int = 30,
        num_colors: int = 10,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.max_h = max_h
        self.max_w = max_w
        self.num_colors = num_colors

        pos_dim = slot_dim // 2  # 64
        self.row_embed = nn.Embedding(max_h, pos_dim)
        self.col_embed = nn.Embedding(max_w, pos_dim)

        # Cross-attention projections (bias=False)
        self.proj_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.proj_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.proj_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # Reconstruction MLP to color logits
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, num_colors),
        )

    def forward(
        self,
        slots: torch.Tensor,
        objectness: Optional[torch.Tensor] = None,
        H: int = 10,
        W: int = 10,
    ) -> torch.Tensor:
        """Decode slots into grid color logits.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).
            objectness: Optional objectness scores of shape (B, K) in [0, 1].
            H: Target grid height (1 <= H <= max_h).
            W: Target grid width (1 <= W <= max_w).

        Returns:
            Color logits tensor of shape (B, H, W, num_colors).
        """
        if H > self.max_h or W > self.max_w:
            raise ValueError(
                f"Grid dimension ({H}, {W}) exceeds maximum allowed ({self.max_h}, {self.max_w})"
            )

        B, K, D = slots.shape
        device = slots.device

        # Generate 2D coordinate queries from row and column embeddings
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        r_emb = self.row_embed(rows).unsqueeze(1).expand(H, W, -1)  # (H, W, 64)
        c_emb = self.col_embed(cols).unsqueeze(0).expand(H, W, -1)  # (H, W, 64)
        pos = torch.cat([r_emb, c_emb], dim=-1)  # (H, W, 128)

        # Reshape to batch queries: (B, H*W, slot_dim)
        cell_queries = pos.reshape(1, H * W, self.slot_dim).expand(B, -1, -1)

        # Cross-attention: Cell queries (Q) attend over slots (K, V)
        q = self.proj_q(cell_queries)  # (B, N, slot_dim)
        k = self.proj_k(slots)  # (B, K, slot_dim)
        v = self.proj_v(slots)  # (B, K, slot_dim)

        scale = self.slot_dim ** -0.5
        attn_logits = torch.einsum("b n d, b k d -> b n k", q, k) * scale

        if objectness is not None:
            # Optionally modulate cross-attention logits with objectness prior
            # Uses log objectness as additive bias with numerical clamp
            obj_bias = torch.log(objectness.unsqueeze(1).clamp(min=1e-6))
            attn_weights = F.softmax(attn_logits + obj_bias, dim=-1)
        else:
            attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate slot values: (B, N, slot_dim)
        cell_feats = torch.einsum("b n k, b k d -> b n d", attn_weights, v)

        # Decode into color logits: (B, H, W, num_colors)
        logits = self.mlp(cell_feats)
        return logits.reshape(B, H, W, self.num_colors)


if __name__ == "__main__":
    print("Running ReconstructionDecoder smoke test...")
    model = ReconstructionDecoder()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"ReconstructionDecoder parameter count: {param_count}")
    assert param_count == 70794, f"Expected 70,794 parameters, got {param_count}"

    test_cases = [
        ((4, 24, 128), 10, 10),
        ((1, 24, 128), 5, 5),
        ((2, 24, 128), 8, 12),
        ((2, 24, 128), 15, 15),
        ((2, 24, 128), 30, 30),
    ]

    for (B, K, D), H, W in test_cases:
        dummy_slots = torch.randn(B, K, D)
        dummy_obj = torch.rand(B, K)

        # Test with and without objectness scores
        out_with_obj = model(dummy_slots, objectness=dummy_obj, H=H, W=W)
        out_without_obj = model(dummy_slots, objectness=None, H=H, W=W)

        expected_shape = (B, H, W, 10)
        assert out_with_obj.shape == expected_shape, f"Expected {expected_shape}, got {out_with_obj.shape}"
        assert out_without_obj.shape == expected_shape, f"Expected {expected_shape}, got {out_without_obj.shape}"
        print(f"Passed test for slots (B={B}, K={K}, D={D}) -> grid ({H}x{W}) -> logits {out_with_obj.shape}")

    print("All ReconstructionDecoder smoke tests passed successfully!")
