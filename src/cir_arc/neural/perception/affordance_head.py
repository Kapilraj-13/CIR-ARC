"""Object Affordance Prediction Head for CIR-ARC Phase 2.5/World-State v2.

Predicts 9 interactive affordances per detected object slot:
1. can_move: Autonomous movement or controllable piece
2. can_push: Can be pushed laterally or longitudinally
3. can_collect: Item or pickup (key, coin, diamond)
4. can_interact: Action-5 interactive triggerable entity
5. can_toggle: State switch or pressure plate
6. can_destroy: Breakable wall or destructible hazard
7. can_block: Rigid obstacle/barrier blocking agent
8. can_support: Floor or platform providing structural support
9. can_be_clicked: Selectable/clickable via cursor pointer (ACTION6)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from cir_arc.neural.world_state import AFFORDANCE_NAMES, NUM_AFFORDANCES


class ObjectAffordanceHead(nn.Module):
    """Parallel MLP affordance prediction head operating on slot representations."""

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 128,
        num_affordances: int = NUM_AFFORDANCES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.num_affordances = num_affordances

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_affordances),
        )

    def forward(
        self,
        slots: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Forward pass predicting affordance logits for each slot.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).
            return_probs: If True, applies Sigmoid to return probabilities in [0, 1].

        Returns:
            Affordance tensor of shape (B, K, num_affordances).
        """
        logits = self.mlp(slots)
        if return_probs:
            return torch.sigmoid(logits)
        return logits

    def predict_affordance_dict(
        self,
        slot_vector: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Convenience method returning a name-keyed dict of affordance probabilities."""
        if slot_vector.dim() == 1:
            slot_vector = slot_vector.unsqueeze(0).unsqueeze(0)
        elif slot_vector.dim() == 2:
            slot_vector = slot_vector.unsqueeze(0)

        with torch.no_grad():
            probs = torch.sigmoid(self.mlp(slot_vector))[0, 0].cpu().numpy()

        return {
            name: float(probs[i])
            for i, name in enumerate(AFFORDANCE_NAMES)
            if i < len(probs)
        }


if __name__ == "__main__":
    head = ObjectAffordanceHead(slot_dim=128)
    x = torch.randn(2, 24, 128)
    out = head(x)
    assert out.shape == (2, 24, 9)
    probs = head(x, return_probs=True)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    print("ObjectAffordanceHead verified successfully!")
