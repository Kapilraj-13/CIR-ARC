r"""Counterfactual Planner for the CIR-ARC Reasoner.

Executes shallow latent rollout over candidate actions (K=3-5, depth 1-5) using
the WorldModelHead and ValueHead to score actions according to:
    Score(a) = G(a) + \lambda_s * S(a) + \lambda_i * I(a) + \lambda_v * V(a) - \lambda_c * C(a) - \lambda_r * R(a)
where:
    G(a) = Expected goal progress (cosine similarity / latent distance reduction)
    S(a) = Success probability
    I(a) = Information gain (epistemic uncertainty reduction)
    V(a) = Future state value estimate
    C(a) = Action execution cost
    R(a) = Risk penalty
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from cir_arc.neural.reasoner.heads import (
    ActionInterface,
    ActionIntent,
    PredictedTransition,
    ValueHead,
    WorldModelHead,
)


@dataclass
class CandidateActionScore:
    """Detailed score components for a candidate action."""
    action_id: int
    action_name: str
    total_score: float
    goal_progress: float
    success_prob: float
    info_gain: float
    future_value: float
    action_cost: float
    risk_penalty: float
    target_entity: Optional[int] = None
    predicted_transition: Optional[PredictedTransition] = None


class CounterfactualPlanner:
    """Shallow tree search and counterfactual evaluator operating entirely in latent space."""

    def __init__(
        self,
        world_model: WorldModelHead,
        value_head: ValueHead,
        action_interface: ActionInterface,
        lambda_s: float = 0.5,
        lambda_i: float = 0.3,
        lambda_v: float = 1.0,
        lambda_c: float = 0.1,
        lambda_r: float = 0.5,
        top_k: int = 4,
        max_depth: int = 3,
    ) -> None:
        self.world_model = world_model
        self.value_head = value_head
        self.action_interface = action_interface
        self.lambda_s = lambda_s
        self.lambda_i = lambda_i
        self.lambda_v = lambda_v
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.top_k = top_k
        self.max_depth = max_depth

    def score_candidate(
        self,
        current_state: torch.Tensor,
        action_id: int,
        goal_vector: Optional[torch.Tensor] = None,
        entity_tokens: Optional[torch.Tensor] = None,
    ) -> CandidateActionScore:
        """Simulates candidate action in latent space and computes multi-objective score."""
        B = current_state.shape[0]
        a_tensor = torch.tensor([action_id], dtype=torch.long, device=current_state.device)

        # 1. World model latent transition
        pred = self.world_model(current_state, a_tensor, entity_tokens=entity_tokens)
        next_latent = pred.next_latent

        # 2. Goal progress: reduction in distance to goal
        goal_progress = 0.0
        if goal_vector is not None:
            curr_dist = F.cosine_similarity(current_state, goal_vector).item()
            next_dist = F.cosine_similarity(next_latent, goal_vector).item()
            goal_progress = float(next_dist - curr_dist)

        # 3. Success probability and information gain
        success_prob = float(pred.action_success.item())
        info_gain = float(pred.prediction_uncertainty.item())

        # 4. Value and risk estimates
        val, risk_cost = self.value_head(next_latent, goal=goal_vector)
        future_val = float(val.item())
        action_cost = float(risk_cost[0, 0].item())
        risk_penalty = float(risk_cost[0, 1].item())

        # Total multi-objective candidate action score
        total_score = (
            goal_progress
            + self.lambda_s * success_prob
            + self.lambda_i * info_gain
            + self.lambda_v * future_val
            - self.lambda_c * action_cost
            - self.lambda_r * risk_penalty
        )

        action_name = ActionInterface.ACTION_NAMES[action_id] if action_id < len(ActionInterface.ACTION_NAMES) else f"ACTION_{action_id}"

        return CandidateActionScore(
            action_id=action_id,
            action_name=action_name,
            total_score=total_score,
            goal_progress=goal_progress,
            success_prob=success_prob,
            info_gain=info_gain,
            future_value=future_val,
            action_cost=action_cost,
            risk_penalty=risk_penalty,
            predicted_transition=pred,
        )

    def plan_best_action(
        self,
        cognitive_state: torch.Tensor,
        candidate_actions: List[int],
        goal_vector: Optional[torch.Tensor] = None,
        entity_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[ActionIntent, List[CandidateActionScore]]:
        """Evaluates all candidate actions and returns best ActionIntent and all scored candidates."""
        scores: List[CandidateActionScore] = []

        for a in candidate_actions:
            sc = self.score_candidate(
                cognitive_state,
                action_id=a,
                goal_vector=goal_vector,
                entity_tokens=entity_tokens,
            )
            scores.append(sc)

        scores.sort(key=lambda x: x.total_score, reverse=True)
        best = scores[0]

        target_entity = None
        if best.action_id == 6 and entity_tokens is not None and entity_tokens.shape[1] > 0:
            # For CLICK: select entity token with highest activation
            _, pointer_logits, _ = self.action_interface(cognitive_state, entity_tokens)
            if pointer_logits is not None:
                target_entity = int(pointer_logits.argmax(dim=-1).item())

        intent = ActionIntent(
            action_type_id=best.action_id,
            action_name=best.action_name,
            target_entity_id=target_entity,
            confidence=best.success_prob,
            expected_value=best.future_value,
            info_gain=best.info_gain,
        )

        return intent, scores
