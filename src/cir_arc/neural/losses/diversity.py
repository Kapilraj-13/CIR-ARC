"""Slot diversity and objectness sparsity regularization losses."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def diversity_loss(slots: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute slot diversity loss penalizing squared off-diagonal cosine similarity.

    Args:
        slots: Tensor of shape (B, K, D) or (K, D).
        eps: Small epsilon for numerical stability in L2 normalization.

    Returns:
        Scalar loss tensor.
    """
    if slots.dim() == 2:
        slots = slots.unsqueeze(0)  # (1, K, D)

    if slots.dim() != 3:
        raise ValueError(f"Expected slots to have shape (B, K, D) or (K, D), got {slots.shape}")

    B, K, D = slots.shape
    if K <= 1:
        return slots.sum() * 0.0

    # L2 normalize each slot representation along dimension D
    norm_slots = slots / (torch.norm(slots, p=2, dim=-1, keepdim=True) + eps)  # (B, K, D)

    # Pairwise cosine similarity matrix: (B, K, K)
    sim_matrix = torch.bmm(norm_slots, norm_slots.transpose(1, 2))

    # Squared cosine similarity
    sim_matrix_sq = sim_matrix ** 2

    # Off-diagonal mask (excluding self-similarity where i == j)
    eye_mask = torch.eye(K, dtype=torch.bool, device=slots.device).unsqueeze(0)  # (1, K, K)
    off_diag_mask = (~eye_mask).to(dtype=sim_matrix_sq.dtype)

    off_diag_sq = sim_matrix_sq * off_diag_mask
    num_pairs = float(B * K * (K - 1))

    return off_diag_sq.sum() / (num_pairs + eps)


def objectness_sparsity_loss(objectness: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 sparsity loss on objectness probabilities.

    Args:
        objectness: Tensor of shape (B, K), (K,), or (B, K, 1).

    Returns:
        Scalar loss tensor (mean of absolute objectness activations).
    """
    return torch.mean(torch.abs(objectness))


slot_diversity_loss = diversity_loss


class SlotDiversityLoss(nn.Module):
    """
    Module wrapper for Slot Diversity regularization.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return diversity_loss(slots, eps=self.eps)


class ObjectnessSparsityLoss(nn.Module):
    """
    Module wrapper for Objectness L1 Sparsity regularization.
    """

    def __init__(self):
        super().__init__()

    def forward(self, objectness: torch.Tensor) -> torch.Tensor:
        return objectness_sparsity_loss(objectness)


if __name__ == "__main__":
    B, K, D = 2, 24, 128
    slots = torch.randn(B, K, D, requires_grad=True)
    obj = torch.rand(B, K, requires_grad=True)
    div_mod = SlotDiversityLoss()
    sparse_mod = ObjectnessSparsityLoss()
    l_div = div_mod(slots)
    l_sparse = sparse_mod(obj)
    total = l_div + l_sparse
    total.backward()
    print(f"Diversity smoke test passed: div_loss={l_div.item():.4f}, sparse_loss={l_sparse.item():.4f}")
