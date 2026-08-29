"""Slot Attention module with proposal-guided initialization for CIR-ARC.

Implements competitive binding slot attention (Locatello et al., 2020) enhanced with:
- Learned Gaussian slot prior (slots_mu, slots_log_sigma)
- Optional data-dependent proposal initialization from high-objectness spatial features
- Q, K, V linear projections (bias=False)
- Competitive softmax over slot dimension (dim=1)
- Spatial normalization over pixel dimension (dim=-1) with objectness weighting
- GRUCell recurrent state update + LayerNorm + residual MLP
- Objectness scoring head (Linear(128, 64) -> GELU -> Linear(64, 1) -> Sigmoid)

Outputs slots (B, 24, 128), objectness (B, 24), and attention maps (B, 24, N).
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Slot Attention module for object discovery with competitive binding.

    Args:
        n_slots: Number of object slots K (default: 24).
        slot_dim: Dimensionality of each slot vector (default: 128).
        feat_dim: Dimensionality of input visual feature tokens (default: 128).
        n_iter: Number of iterative attention refinement steps (default: 3).
        eps: Small epsilon for numerical stability during division (default: 1e-8).
        hidden_dim: Hidden dimension of the residual MLP (default: 256).
        proposal_init: Whether to instantiate data-dependent proposal projection (default: False).
    """

    def __init__(
        self,
        n_slots: int = 24,
        slot_dim: int = 128,
        feat_dim: int = 128,
        n_iter: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 256,
        proposal_init: bool = False,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.feat_dim = feat_dim
        self.n_iter = n_iter
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.proposal_init = proposal_init

        # Learned Gaussian initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.zeros_(self.slots_log_sigma)

        # Optional proposal feature projection
        if proposal_init:
            self.proposal_proj = nn.Linear(feat_dim, slot_dim)
        else:
            self.proposal_proj = None

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

    def _initialize_slots(
        self,
        inputs: torch.Tensor,
        cell_objectness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initializes slot embeddings from Gaussian prior or proposal features."""
        B, N, D = inputs.shape

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
        gaussian_slots = mu + sigma * noise

        if cell_objectness is None or self.proposal_proj is None:
            return gaussian_slots

        # Extract top-K objectness feature vectors
        obj_flat = cell_objectness.reshape(B, N) if cell_objectness.dim() > 2 else cell_objectness
        obj_flat = torch.nan_to_num(obj_flat, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        K = min(self.n_slots, N)
        _, topk_indices = torch.topk(obj_flat, k=K, dim=-1)

        batch_indices = torch.arange(B, device=inputs.device).unsqueeze(-1).expand(-1, K)
        selected_tokens = inputs[batch_indices, topk_indices]
        proposal_slots = self.proposal_proj(selected_tokens)

        if K < self.n_slots:
            pad_slots = gaussian_slots[:, K:, :]
            proposal_slots = torch.cat([proposal_slots, pad_slots], dim=1)

        return 0.5 * proposal_slots + 0.5 * gaussian_slots

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cell_objectness: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of Slot Attention.

        Args:
            inputs: Spatial visual feature tokens of shape (B, N, feat_dim).
            mask: Optional spatial mask of shape (B, N).
            cell_objectness: Optional cell objectness map of shape (B, 1, H, W) or (B, N).

        Returns:
            Tuple of slots (B, 24, 128), objectness (B, 24), attn_maps (B, 24, N).
        """
        B, N, D = inputs.shape

        normed_inputs = self.norm_inputs(inputs)
        k = self.project_k(normed_inputs)
        v = self.project_v(normed_inputs)

        slots = self._initialize_slots(normed_inputs, cell_objectness=cell_objectness)

        scale = self.slot_dim ** -0.5
        attn_maps = torch.empty(0, device=inputs.device)

        for _ in range(self.n_iter):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            q = self.project_q(slots_norm)
            dots = torch.einsum("b k d, b n d -> b k n", q, k) * scale

            # Competitive binding: Softmax over slot dimension (dim=1)
            attn = F.softmax(dots, dim=1)  # (B, K, N), sum(dim=1) == 1.0
            attn_maps = attn

            spatial_weights = attn
            if mask is not None:
                mask_unsq = mask.unsqueeze(1).float()
                spatial_weights = spatial_weights * mask_unsq

            if cell_objectness is not None:
                obj_unsq = cell_objectness.reshape(B, 1, N).float()
                spatial_weights = spatial_weights * (0.2 + 0.8 * obj_unsq)

            attn_norm = spatial_weights / (spatial_weights.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum("b k n, b n d -> b k d", attn_norm, v)

            slots_flat = slots_prev.reshape(-1, self.slot_dim)
            updates_flat = updates.reshape(-1, self.slot_dim)
            slots_updated = self.gru(updates_flat, slots_flat)
            slots = slots_updated.reshape(B, self.n_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        objectness = self.objectness_head(slots).squeeze(-1)
        return slots, objectness, attn_maps


if __name__ == "__main__":
    print("Running SlotAttention smoke test...")
    model = SlotAttention()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"SlotAttention parameter count: {param_count}")
    assert param_count == 223489, f"Expected 223,489 parameters, got {param_count}"
    print("All SlotAttention smoke tests passed successfully!")
