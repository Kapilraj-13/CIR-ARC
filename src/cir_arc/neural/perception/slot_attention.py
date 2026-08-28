"""Slot Attention module for object-centric representation in CIR-ARC.

Implements competitive binding slot attention (Locatello et al., 2020):
- Learned Gaussian slot initialization (slots_mu, slots_log_sigma)
- Q, K, V projections (bias=False)
- Competitive softmax over slot dimension (dim=1)
- Spatial normalization over pixel dimension (dim=-1)
- GRUCell recurrent state update + LayerNorm + residual MLP
- Objectness scoring head (Linear(128, 64) -> GELU -> Linear(64, 1) -> Sigmoid)

Outputs slots (B, 24, 128), objectness (B, 24), and attention maps (B, 24, N).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Slot Attention module for unsupervised object discovery.

    Decomposes spatial visual tokens into K discrete object slots via iterative
    competitive binding.

    Args:
        n_slots: Number of object slots K (default: 24).
        slot_dim: Dimensionality of each slot vector (default: 128).
        feat_dim: Dimensionality of input visual feature tokens (default: 128).
        n_iter: Number of iterative attention refinement steps (default: 3).
        eps: Small epsilon for numerical stability during division (default: 1e-8).
        hidden_dim: Hidden dimension of the residual MLP (default: 256).
    """

    def __init__(
        self,
        n_slots: int = 24,
        slot_dim: int = 128,
        feat_dim: int = 128,
        n_iter: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.feat_dim = feat_dim
        self.n_iter = n_iter
        self.eps = eps
        self.hidden_dim = hidden_dim

        # Learned Gaussian initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.zeros_(self.slots_log_sigma)

        # Layer Normalizations
        self.norm_inputs = nn.LayerNorm(feat_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Q, K, V linear projections without bias
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feat_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feat_dim, slot_dim, bias=False)

        # Recurrent state update (GRUCell)
        self.gru = nn.GRUCell(input_size=slot_dim, hidden_size=slot_dim)

        # Residual MLP block
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        # Objectness scoring head (outputs existence probability in [0, 1] per slot)
        self.objectness_head = nn.Sequential(
            nn.Linear(slot_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of Slot Attention.

        Args:
            inputs: Spatial visual feature tokens of shape (B, N, feat_dim).
            mask: Optional spatial mask of shape (B, N), where True/1 indicates valid pixels
                and False/0 indicates padded pixels.

        Returns:
            Tuple of:
                - slots: Tensor of shape (B, n_slots, slot_dim).
                - objectness: Tensor of shape (B, n_slots) in [0, 1].
                - attn_maps: Attention distribution tensor of shape (B, n_slots, N),
                    summing to 1.0 along dim=1 (slot dimension).
        """
        B, N, D = inputs.shape

        # Normalize and project spatial inputs
        normed_inputs = self.norm_inputs(inputs)
        k = self.project_k(normed_inputs)  # (B, N, slot_dim)
        v = self.project_v(normed_inputs)  # (B, N, slot_dim)

        # Initialize slot embeddings from learned Gaussian distribution
        mu = self.slots_mu.expand(B, self.n_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.n_slots, -1)
        if self.training:
            noise = torch.randn(
                B, self.n_slots, self.slot_dim, device=inputs.device, dtype=inputs.dtype
            )
        else:
            gen = torch.Generator(device=inputs.device if inputs.device.type == "cpu" else "cpu").manual_seed(42)
            noise = torch.randn(
                B, self.n_slots, self.slot_dim, generator=gen, dtype=inputs.dtype
            ).to(inputs.device)
        slots = mu + sigma * noise

        scale = self.slot_dim ** -0.5
        attn_maps = torch.empty(0, device=inputs.device)

        # Iterative competitive binding
        for _ in range(self.n_iter):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # Project queries
            q = self.project_q(slots_norm)  # (B, K, slot_dim)

            # Dot-product attention matrix: (B, K, N)
            dots = torch.einsum("b k d, b n d -> b k n", q, k) * scale

            # Competitive binding: Softmax over slot dimension (dim=1)
            # This enforces that all slots compete for each spatial token
            attn = F.softmax(dots, dim=1)  # (B, K, N), sum(dim=1) == 1.0
            attn_maps = attn

            # Spatial normalization across pixel dimension N
            if mask is not None:
                mask_unsq = mask.unsqueeze(1).float()  # (B, 1, N)
                attn_spatial = attn * mask_unsq
                attn_norm = attn_spatial / (
                    attn_spatial.sum(dim=-1, keepdim=True) + self.eps
                )
            else:
                attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            # Aggregate values: (B, K, slot_dim)
            updates = torch.einsum("b k n, b n d -> b k d", attn_norm, v)

            # Recurrent GRU state update
            slots_flat = slots_prev.reshape(-1, self.slot_dim)
            updates_flat = updates.reshape(-1, self.slot_dim)
            slots_updated = self.gru(updates_flat, slots_flat)
            slots = slots_updated.reshape(B, self.n_slots, self.slot_dim)

            # Residual MLP update
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Compute objectness scores: (B, K)
        objectness = self.objectness_head(slots).squeeze(-1)

        return slots, objectness, attn_maps


if __name__ == "__main__":
    print("Running SlotAttention smoke test...")
    model = SlotAttention()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"SlotAttention parameter count: {param_count}")
    assert param_count == 223489, f"Expected 223,489 parameters, got {param_count}"

    test_configs = [
        ((4, 100, 128), None),
        ((1, 25, 128), None),
        ((2, 96, 128), torch.ones(2, 96, dtype=torch.bool)),
        ((2, 900, 128), None),
    ]

    for in_shape, mask in test_configs:
        dummy_inputs = torch.randn(in_shape)
        slots, objectness, attn_maps = model(dummy_inputs, mask=mask)

        B, N, _ = in_shape
        assert slots.shape == (B, 24, 128), f"Unexpected slots shape {slots.shape}"
        assert objectness.shape == (B, 24), f"Unexpected objectness shape {objectness.shape}"
        assert attn_maps.shape == (B, 24, N), f"Unexpected attn_maps shape {attn_maps.shape}"

        # Invariant check 1: Objectness scores in [0, 1]
        assert (objectness >= 0.0).all() and (objectness <= 1.0).all(), "Objectness scores out of [0, 1] range"

        # Invariant check 2: Competitive binding invariant - attention maps sum to 1.0 along slot dimension
        attn_sum = attn_maps.sum(dim=1)
        expected_sum = torch.ones_like(attn_sum)
        max_diff = torch.max(torch.abs(attn_sum - expected_sum)).item()
        assert max_diff < 1e-5, f"Competitive binding invariant violated! Max diff: {max_diff}"

        print(f"Passed test for {in_shape} -> slots: {slots.shape}, objectness: {objectness.shape}, max attn sum diff: {max_diff:.2e}")

    print("All SlotAttention smoke tests passed successfully!")
