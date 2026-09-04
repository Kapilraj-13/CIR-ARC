"""Cognitive Reasoning & Output Heads for the CIR-ARC ~120.18M Reasoner.

Contains:
1. GoalInferenceHead: Generates K=4 parallel latent goal hypotheses with confidence logits.
2. WorldModelHead: Rich transition predictor P(S_{t+1}, E_{t+1}, A_{success} | S_t, A_t).
       Emits PredictedTransition dataclass.
3. ValueHead: Predicts scalar state value V(S, G) along with action cost and risk estimates.
4. ActionInterface: ActionIntent generator with discrete action logits (MOVE_4, ACTION, UNDO, CLICK)
       and entity pointer scoring.
5. VerificationHead: Compares predicted transition vs observed transition, computing
       prediction error and belief update gating signal.

Total parameters: 6,226,592.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictedTransition:
    """Rich transition prediction output by WorldModelHead."""
    next_latent: torch.Tensor                        # [B, 768] next cognitive state latent
    event_probs: torch.Tensor                        # [B, 14] transition event probabilities
    action_success: torch.Tensor                     # [B, 1] probability action will succeed
    confidence: torch.Tensor                         # [B, 1] model confidence in transition
    prediction_uncertainty: torch.Tensor             # [B, 1] estimated transition uncertainty
    affected_entities: Optional[torch.Tensor] = None # [B, K] probability each entity was affected


@dataclass
class ActionIntent:
    """Structured action proposal emitted by the Reasoner."""
    action_type_id: int                              # 0..6 (MOVE_UP, DOWN, LEFT, RIGHT, ACTION, UNDO, CLICK)
    action_name: str
    target_entity_id: Optional[int] = None           # Pointer to object slot for CLICK actions
    confidence: float = 1.0
    expected_value: float = 0.0
    info_gain: float = 0.0
    action_logits: Optional[torch.Tensor] = None


class GoalInferenceHead(nn.Module):
    """Multi-hypothesis latent goal generator.

    Total parameters: 1,974,276.
    """

    def __init__(self, d_model: int = 768, num_hypotheses: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_hypotheses = num_hypotheses

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, num_hypotheses * d_model),
        )
        self.conf_head = nn.Linear(d_model, num_hypotheses)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cognitive_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infers candidate latent goals and hypothesis confidence.

        Args:
            cognitive_state: Pooled state embedding [B, d_model].

        Returns:
            Tuple of (goals [B, 4, d_model], confidence_probs [B, 4]).
        """
        B = cognitive_state.shape[0]
        goals_flat = self.mlp(cognitive_state)  # [B, 4 * 768]
        goals = goals_flat.view(B, self.num_hypotheses, self.d_model)
        goals = self.norm(goals)

        conf_logits = self.conf_head(cognitive_state)  # [B, 4]
        conf_probs = F.softmax(conf_logits, dim=-1)
        return goals, conf_probs


class WorldModelHead(nn.Module):
    """Latent transition model predicting P(S_{t+1}, E_{t+1}, A_{success} | S_t, A_t).

    Total parameters: 1,292,047.
    """

    def __init__(self, d_model: int = 768, num_actions: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_emb = nn.Embedding(num_actions, 128)
        self.transition_mlp = nn.Sequential(
            nn.Linear(d_model + 128, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.event_pred = nn.Linear(d_model, 14)
        self.success_pred = nn.Linear(d_model, 1)

    def forward(
        self,
        current_state: torch.Tensor,
        action_ids: torch.Tensor,
        entity_tokens: Optional[torch.Tensor] = None,
    ) -> PredictedTransition:
        """Predicts rich transition given current state and candidate action.

        Args:
            current_state: Current pooled state vector [B, d_model].
            action_ids: Action index tensor [B].
            entity_tokens: Optional entity tokens [B, K, d_model] to estimate affected objects.

        Returns:
            PredictedTransition dataclass.
        """
        a_emb = self.action_emb(action_ids.clamp(0, self.action_emb.num_embeddings - 1))
        concat = torch.cat([current_state, a_emb], dim=-1)
        next_latent = self.transition_mlp(concat)

        event_logits = self.event_pred(next_latent)
        event_probs = F.softmax(event_logits, dim=-1)

        success_logit = self.success_pred(next_latent)
        action_success = torch.sigmoid(success_logit)

        # Confidence and uncertainty derived from latent consistency and success probability
        confidence = action_success
        uncertainty = 1.0 - confidence

        # Compute affected entities from latent state change
        affected_entities = None
        if entity_tokens is not None:
            # Cosine distance between next_latent and entity tokens: [B, K]
            norm_next = F.normalize(next_latent.unsqueeze(1), dim=-1)
            norm_ent = F.normalize(entity_tokens, dim=-1)
            sim = (norm_next * norm_ent).sum(dim=-1)
            affected_entities = torch.sigmoid(sim)

        return PredictedTransition(
            next_latent=next_latent,
            event_probs=event_probs,
            action_success=action_success,
            confidence=confidence,
            prediction_uncertainty=uncertainty,
            affected_entities=affected_entities,
        )


class ValueHead(nn.Module):
    """Evaluates state value V(S, G) and estimates action execution risk/cost.

    Total parameters: 591,747.
    """

    def __init__(self, d_model: int = 768, val_hidden_dim: int = 384) -> None:
        super().__init__()
        self.val_mlp = nn.Sequential(
            nn.Linear(d_model, val_hidden_dim),
            nn.GELU(),
            nn.Linear(val_hidden_dim, 1),
        )
        self.risk_mlp = nn.Sequential(
            nn.Linear(d_model, val_hidden_dim),
            nn.GELU(),
            nn.Linear(val_hidden_dim, 2),
        )

    def forward(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates state value and risk.

        Args:
            state: Cognitive state vector [B, d_model].
            goal: Optional goal latent vector [B, d_model].

        Returns:
            Tuple of (value [B, 1], risk_cost [B, 2] where [:, 0] is cost, [:, 1] is risk).
        """
        combined = state if goal is None else state + goal
        value = torch.tanh(self.val_mlp(combined))
        risk_cost = F.softplus(self.risk_mlp(combined))
        return value, risk_cost


class ActionInterface(nn.Module):
    """Emits discrete action type distribution and object pointer scores.

    Actions:
        0: MOVE_UP
        1: MOVE_DOWN
        2: MOVE_LEFT
        3: MOVE_RIGHT
        4: ACTION
        5: UNDO
        6: CLICK (entity pointer)

    Total parameters: 6,921.
    """

    ACTION_NAMES = [
        "MOVE_UP",
        "MOVE_DOWN",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ACTION",
        "UNDO",
        "CLICK",
    ]

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.action_logits = nn.Linear(d_model, 7)
        self.pointer_entity_logits = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(
        self,
        cognitive_state: torch.Tensor,
        entity_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Emits action logits, entity pointer logits, and confidence score.

        Args:
            cognitive_state: Pooled state token [B, d_model].
            entity_tokens: Optional entity tokens [B, K, d_model].

        Returns:
            Tuple of (action_logits [B, 7], pointer_logits [B, K], confidence [B, 1]).
        """
        logits = self.action_logits(cognitive_state)
        conf = torch.sigmoid(self.confidence_head(cognitive_state))

        pointer_logits = None
        if entity_tokens is not None and entity_tokens.shape[1] > 0:
            pointer_logits = self.pointer_entity_logits(entity_tokens).squeeze(-1)  # [B, K]

        return logits, pointer_logits, conf


class VerificationHead(nn.Module):
    """Compares predicted next state vs observed actual state to drive belief revision.

    Total parameters: 2,361,601.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.diff_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.belief_update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        predicted_latent: torch.Tensor,
        observed_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compares prediction vs observation.

        Args:
            predicted_latent: Predicted state [B, d_model].
            observed_latent: Actual observed state [B, d_model].

        Returns:
            Tuple of (prediction_error [B, 1], update_gate [B, d_model]).
        """
        concat = torch.cat([predicted_latent, observed_latent], dim=-1)
        error = F.softplus(self.diff_proj(concat))
        gate = self.belief_update_gate(concat)
        return error, gate
