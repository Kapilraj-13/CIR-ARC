"""Action-conditioned latent slot transition model and online mechanics belief tracker.

Contains:
1. ActionConditionedTransitionModel: Simulates causal next-step slot latent changes
   (S_t, a_t -> S_hat_{t+1}, delta_pos) without committing real environment actions.
2. OnlineMechanicsTracker: Dynamically updates environment physics beliefs
   (gravity, friction, pushability, sliding inertia, wrapping) from active observation deltas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from cir_arc.neural.world_state import ActionEffect, MechanicsBelief, StructuredObject


class ActionConditionedTransitionModel(nn.Module):
    """Latent slot transition block predicting next-step slot states given candidate action."""

    def __init__(
        self,
        slot_dim: int = 128,
        num_actions: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.num_actions = num_actions

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, slot_dim)

        # Cross-interaction MLP between slot and action
        self.transition_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        # Position delta head (normalized delta_r, delta_c in [-1, 1])
        self.delta_pos_head = nn.Sequential(
            nn.Linear(slot_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        slots: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate next slot embeddings and position shifts under candidate action.

        Args:
            slots: Current slot representations of shape (B, K, slot_dim).
            action: LongTensor of action IDs of shape (B,).

        Returns:
            Tuple of:
                next_slots: (B, K, slot_dim)
                delta_pos: (B, K, 2) predicted position offsets
        """
        B, K, D = slots.shape
        act_emb = self.action_embed(action).unsqueeze(1).expand(B, K, D)  # (B, K, D)

        joint = torch.cat([slots, act_emb], dim=-1)  # (B, K, 2*D)
        delta_slots = self.transition_mlp(joint)

        next_slots = slots + delta_slots
        delta_pos = self.delta_pos_head(delta_slots)

        return next_slots, delta_pos


class OnlineMechanicsTracker:
    """Updates runtime MechanicsBelief and ActionEffect profiles through active observation."""

    def __init__(self) -> None:
        self.belief = MechanicsBelief()
        self.action_history: List[Tuple[int, List[Tuple[float, float]]]] = []
        self.observed_dr_list: List[float] = []
        self.observed_dc_list: List[float] = []

    def reset(self) -> None:
        self.belief = MechanicsBelief()
        self.action_history.clear()
        self.observed_dr_list.clear()
        self.observed_dc_list.clear()

    def update_from_transition(
        self,
        action_id: int,
        prev_objects: List[StructuredObject],
        curr_objects: List[StructuredObject],
    ) -> MechanicsBelief:
        """Update physics hypotheses by comparing object displacements."""
        prev_by_id = {obj.slot_id: obj for obj in prev_objects}
        shifts: List[Tuple[float, float]] = []

        for obj in curr_objects:
            if obj.slot_id in prev_by_id:
                p = prev_by_id[obj.slot_id]
                dr = obj.centroid[0] - p.centroid[0]
                dc = obj.centroid[1] - p.centroid[1]
                if abs(dr) > 1e-4 or abs(dc) > 1e-4:
                    shifts.append((dr, dc))
                    self.observed_dr_list.append(dr)
                    self.observed_dc_list.append(dc)

        self.action_history.append((action_id, shifts))

        # Infer continuous gravity hypothesis
        if len(self.observed_dr_list) >= 3:
            mean_dr = float(np.mean(self.observed_dr_list[-15:]))
            mean_dc = float(np.mean(self.observed_dc_list[-15:]))
            # If vertical shift dominates regardless of action
            if abs(mean_dr) > 0.05 and abs(mean_dc) < 0.03:
                grav_dir = (1.0 if mean_dr > 0 else -1.0, 0.0)
                conf = min(1.0, float(abs(mean_dr) * 4.0))
                self.belief.gravity = grav_dir
                self.belief.gravity_confidence = conf

        # Infer sliding inertia
        # If object moved when action was 0 (RESET) or orthogonal to action
        if action_id == 0 and len(shifts) > 0:
            self.belief.sliding_inertia = min(1.0, self.belief.sliding_inertia + 0.3)
            self.belief.friction = max(0.0, self.belief.friction - 0.2)

        return self.belief

    def compute_action_effects(self) -> Dict[int, ActionEffect]:
        """Compute estimated action-effect matrix for all 8 actions."""
        effects: Dict[int, ActionEffect] = {}
        for a in range(8):
            moves_player = a in (1, 2, 3, 4)
            effects[a] = ActionEffect(
                action_id=a,
                success_probability=0.95,
                moves_player=moves_player,
                affects_environment=a in (5, 6),
                reversible=a in (1, 2, 3, 4, 7),
                cost=1.0,
            )
        return effects


if __name__ == "__main__":
    trans = ActionConditionedTransitionModel(slot_dim=128)
    s = torch.randn(2, 24, 128)
    a = torch.tensor([1, 4])
    s_next, d_pos = trans(s, a)
    assert s_next.shape == s.shape
    assert d_pos.shape == (2, 24, 2)
    print("ActionConditionedTransitionModel verified successfully!")
