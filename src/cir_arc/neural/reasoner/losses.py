r"""Multi-objective training loss for the CIR-ARC ~120.18M Reasoner (Phase 4 Kaggle).

Computes the unified objective:
    L = \lambda_1 * L_state
      + \lambda_2 * L_goal
      + \lambda_3 * L_dynamics
      + \lambda_4 * L_action
      + \lambda_5 * L_counterfactual
      + \lambda_6 * L_value
      + \lambda_7 * L_plan
      + \lambda_8 * L_verify
      + \lambda_9 * L_efficiency
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReasonerLossWeights:
    """Loss balance coefficients for Phase 4 multi-stage curriculum."""
    lambda_state: float = 1.0
    lambda_goal: float = 1.0
    lambda_dynamics: float = 1.5
    lambda_action: float = 2.0
    lambda_counterfactual: float = 0.5
    lambda_value: float = 1.0
    lambda_plan: float = 0.5
    lambda_verify: float = 1.0
    lambda_efficiency: float = 0.1


class ReasonerMultiObjectiveLoss(nn.Module):
    """Calculates multi-objective losses for training CognitiveReasoner120M in Phase 4."""

    def __init__(self, weights: Optional[ReasonerLossWeights] = None) -> None:
        super().__init__()
        self.weights = weights if weights is not None else ReasonerLossWeights()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Calculates all 9 loss components and the composite scalar loss.

        Args:
            outputs: Dictionary emitted by CognitiveReasoner120M forward pass.
            targets: Dictionary of ground-truth supervised labels from trajectories.

        Returns:
            Dictionary containing individual losses and 'total_loss'.
        """
        device = outputs["cognitive_state"].device
        losses: Dict[str, torch.Tensor] = {}

        # 1. State Latent Consistency Loss (L_state)
        if "target_state_latent" in targets:
            l_state = F.mse_loss(outputs["cognitive_state"], targets["target_state_latent"])
        else:
            l_state = torch.tensor(0.0, device=device)
        losses["loss_state"] = l_state

        # 2. Goal Inference Loss (L_goal)
        if "target_goal_latent" in targets and "goals" in outputs:
            # Distance between top predicted goal hypothesis and true terminal state
            top_goal = outputs["goals"][:, 0]  # [B, d_model]
            l_goal = 1.0 - F.cosine_similarity(top_goal, targets["target_goal_latent"]).mean()
        else:
            l_goal = torch.tensor(0.0, device=device)
        losses["loss_goal"] = l_goal

        # 3. Dynamics World Model Loss (L_dynamics)
        l_dyn = torch.tensor(0.0, device=device)
        if "predicted_next_latent" in outputs and "target_next_latent" in targets:
            l_dyn_latent = F.mse_loss(outputs["predicted_next_latent"], targets["target_next_latent"])
            l_dyn = l_dyn + l_dyn_latent
        if "predicted_event_logits" in outputs and "target_event_ids" in targets:
            l_event = F.cross_entropy(outputs["predicted_event_logits"], targets["target_event_ids"])
            l_dyn = l_dyn + l_event
        losses["loss_dynamics"] = l_dyn

        # 4. Action Selection Loss (L_action)
        l_action = torch.tensor(0.0, device=device)
        if "action_logits" in outputs and "target_action_id" in targets:
            l_action_type = F.cross_entropy(outputs["action_logits"], targets["target_action_id"])
            l_action = l_action + l_action_type
        if "pointer_logits" in outputs and outputs["pointer_logits"] is not None and "target_pointer_slot" in targets:
            l_pointer = F.cross_entropy(outputs["pointer_logits"], targets["target_pointer_slot"])
            l_action = l_action + l_pointer
        losses["loss_action"] = l_action

        # 5. Counterfactual Margin Loss (L_counterfactual)
        if "candidate_scores" in outputs and "optimal_action_mask" in targets:
            scores = outputs["candidate_scores"]  # [B, K_cand]
            mask = targets["optimal_action_mask"]  # [B, K_cand] 1 for best action, 0 for others
            l_cf = F.binary_cross_entropy_with_logits(scores, mask.float())
        else:
            l_cf = torch.tensor(0.0, device=device)
        losses["loss_counterfactual"] = l_cf

        # 6. Value Estimation Loss (L_value)
        if "value" in outputs and "target_discounted_return" in targets:
            l_value = F.mse_loss(outputs["value"], targets["target_discounted_return"])
        else:
            l_value = torch.tensor(0.0, device=device)
        losses["loss_value"] = l_value

        # 7. Multi-step Plan Consistency Loss (L_plan)
        if "plan_latent_sequence" in outputs and "target_trajectory_latents" in targets:
            l_plan = F.mse_loss(outputs["plan_latent_sequence"], targets["target_trajectory_latents"])
        else:
            l_plan = torch.tensor(0.0, device=device)
        losses["loss_plan"] = l_plan

        # 8. Verification Loss (L_verify)
        if "prediction_error" in outputs and "target_is_error" in targets:
            pred_err = outputs["prediction_error"]
            l_verify = F.binary_cross_entropy(
                torch.sigmoid(pred_err),
                targets["target_is_error"].float(),
            )
        else:
            l_verify = torch.tensor(0.0, device=device)
        losses["loss_verify"] = l_verify

        # 9. Action Efficiency Penalty (L_efficiency)
        if "action_logits" in outputs:
            # Small entropy regularization or L2 norm on action preferences
            action_probs = F.softmax(outputs["action_logits"], dim=-1)
            l_eff = (action_probs ** 2).sum(dim=-1).mean()
        else:
            l_eff = torch.tensor(0.0, device=device)
        losses["loss_efficiency"] = l_eff

        # Weighted composite loss
        total_loss = (
            self.weights.lambda_state * l_state
            + self.weights.lambda_goal * l_goal
            + self.weights.lambda_dynamics * l_dyn
            + self.weights.lambda_action * l_action
            + self.weights.lambda_counterfactual * l_cf
            + self.weights.lambda_value * l_value
            + self.weights.lambda_plan * l_plan
            + self.weights.lambda_verify * l_verify
            + self.weights.lambda_efficiency * l_eff
        )
        losses["total_loss"] = total_loss

        return losses
