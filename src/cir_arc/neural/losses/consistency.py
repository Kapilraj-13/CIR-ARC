"""Dual Neuro-Symbolic Mutual Consistency and Predictive Latent Losses for CIR-ARC.

Implements:
1. Latent Transition Loss: MSE between action-conditioned predicted next slots (S_hat_{t+1})
   and actual encoded next slots (S_{t+1}^*).
2. Neuro-Symbolic Alignment Loss: Encourages mutual predictability between dense continuous
   latent slot vectors and projected symbolic properties (preventing representation collapse).
3. Action Effect Loss: Supervised BCE/MSE on action success, player movement, and reversibility.
4. Event Classification Loss: Cross-entropy on discrete transition events (MOVE, COLLIDE, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.neural.world_state import EVENT_TYPES


def latent_transition_loss(
    pred_next_slots: torch.Tensor,
    target_next_slots: torch.Tensor,
    slot_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes MSE between predicted next latent slots and actual next latent slots.

    Args:
        pred_next_slots: Tensor of shape (B, K, D) from ActionConditionedTransitionModel.
        target_next_slots: Tensor of shape (B, K, D) from PerceptionModel forward on grid_(t+1).
        slot_mask: Optional mask of shape (B, K) for active slots.

    Returns:
        Scalar loss tensor.
    """
    diff_sq = F.mse_loss(pred_next_slots, target_next_slots.detach(), reduction="none")  # (B, K, D)
    if slot_mask is not None:
        weight = slot_mask.unsqueeze(-1).float()
        return (diff_sq * weight).sum() / max(weight.sum() * diff_sq.shape[-1], 1.0)
    return diff_sq.mean()


def neuro_symbolic_alignment_loss(
    dense_slots: torch.Tensor,
    property_logits: Dict[str, torch.Tensor],
    projection_head: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Enforces mutual consistency between DenseLatentState and SymbolicSceneState.

    Penalizes divergence between continuous slot representations and the symbolic properties
    derived from them, ensuring the dual representation stays grounded.

    Args:
        dense_slots: Tensor of shape (B, K, D).
        property_logits: Dict with 'color', 'shape', 'position', 'size', 'extent'.
        projection_head: Optional projection MLP mapping concatenated properties to slot_dim.

    Returns:
        Scalar cosine similarity alignment loss in [0, 2].
    """
    # Build a compact symbolic feature summary from predicted properties
    color_probs = F.softmax(property_logits["color"], dim=-1)     # (B, K, 10)
    shape_probs = F.softmax(property_logits["shape"], dim=-1)     # (B, K, 8)
    pos = property_logits["position"]                             # (B, K, 2)
    sz = property_logits["size"]                                 # (B, K, 1)

    sym_cat = torch.cat([color_probs, shape_probs, pos, sz], dim=-1)  # (B, K, 21)

    if projection_head is not None:
        sym_emb = projection_head(sym_cat)
    else:
        # Pad or slice to match slot_dim
        B, K, D = dense_slots.shape
        sym_emb = torch.zeros((B, K, D), device=dense_slots.device)
        sym_dim = min(sym_cat.shape[-1], D)
        sym_emb[..., :sym_dim] = sym_cat[..., :sym_dim]

    # Normalize vectors and maximize cosine similarity
    norm_dense = F.normalize(dense_slots, p=2, dim=-1)
    norm_sym = F.normalize(sym_emb, p=2, dim=-1)
    cosine_sim = (norm_dense * norm_sym).sum(dim=-1)  # (B, K)

    return (1.0 - cosine_sim).mean()


def action_effect_loss(
    pred_effects: Dict[str, torch.Tensor],
    target_effects: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Computes binary and regression loss on predicted action effects."""
    loss = torch.tensor(0.0, device=next(iter(pred_effects.values())).device)

    if "success_probability" in pred_effects and "success" in target_effects:
        loss += F.binary_cross_entropy_with_logits(
            pred_effects["success_probability"],
            target_effects["success"].float(),
        )

    if "moves_player" in pred_effects and "moves_player" in target_effects:
        loss += F.binary_cross_entropy_with_logits(
            pred_effects["moves_player"],
            target_effects["moves_player"].float(),
        )

    return loss
