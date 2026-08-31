"""Contrastive Object Identity Loss module for slot-based metric learning in Phase 2.5."""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def object_identity_contrastive_loss(
    pred_identities: torch.Tensor,
    matches_batch: Optional[List[List[Tuple[int, int]]]] = None,
    temperature: float = 0.1,
    margin: float = 0.2,
) -> torch.Tensor:
    """Computes contrastive metric loss pushing distinct object slot embeddings apart.

    Ensures that active object slots develop unique, discriminative identity embeddings
    that facilitate object tracking and temporal identity persistence across frames.

    Args:
        pred_identities: L2-normalized slot embeddings of shape (B, K, D) or (K, D).
        matches_batch: Optional Hungarian matches indicating which slots are active objects.
        temperature: Softmax scaling temperature for contrastive logits (default: 0.1).
        margin: Cosine separation margin between distinct object slots (default: 0.2).

    Returns:
        Scalar contrastive identity loss tensor.
    """
    if pred_identities.dim() == 2:
        pred_identities = pred_identities.unsqueeze(0)

    B, K, D = pred_identities.shape
    if K <= 1:
        return torch.tensor(0.0, device=pred_identities.device, requires_grad=True)

    # Normalize vectors
    norm_ident = F.normalize(pred_identities, p=2, dim=-1)  # (B, K, D)

    # Compute pairwise cosine similarity matrix between all slots within each batch sample
    sim_matrix = torch.bmm(norm_ident, norm_ident.transpose(1, 2))  # (B, K, K)

    # Mask out diagonal self-similarities
    eye = torch.eye(K, device=pred_identities.device).unsqueeze(0).expand(B, K, K)
    off_diag_sim = sim_matrix * (1.0 - eye)

    # Penalize distinct active slots having high cosine similarity (anti-collapse / discrimination)
    if matches_batch is not None:
        active_mask = torch.zeros((B, K, K), device=pred_identities.device)
        for b in range(min(B, len(matches_batch))):
            active_slots = [slot_idx for slot_idx, _ in matches_batch[b] if slot_idx < K]
            for i in active_slots:
                for j in active_slots:
                    if i != j:
                        active_mask[b, i, j] = 1.0
        active_sim = off_diag_sim * active_mask
        n_active = active_mask.sum().clamp(min=1.0)
        # Margin hinge loss: penalize cosine similarity > margin
        hinge_loss = F.relu(active_sim - margin).pow(2).sum() / n_active
    else:
        n_elements = float(B * K * (K - 1))
        hinge_loss = F.relu(off_diag_sim - margin).pow(2).sum() / max(n_elements, 1.0)

    return hinge_loss
