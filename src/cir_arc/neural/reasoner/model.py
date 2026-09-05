"""Unified CIR-ARC ~120.18M Direct Cognitive Reasoner.

Full cognitive reasoning model that consumes HybridSceneState (SymbolicSceneState + DenseLatentState),
maintains persistent working memory across time, executes iterative reasoning in an ephemeral
workspace, performs counterfactual action rollouts, emits ActionIntent, and drives belief revision.

Total Parameters: 120,179,360.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from cir_arc.neural.reasoner.config import ReasonerConfig
from cir_arc.neural.reasoner.transformer import CognitiveTransformerTrunk
from cir_arc.neural.reasoner.projections import (
    SymbolicEntityEncoder,
    RelationTokenEncoder,
    EventTokenEncoder,
    MechanicsBeliefEncoder,
    GlobalStateEncoder,
    ActionEffectEncoder,
    UncertaintyEncoder,
    DenseLatentProjections,
)
from cir_arc.neural.reasoner.memory import MemorySystem
from cir_arc.neural.reasoner.heads import (
    GoalInferenceHead,
    WorldModelHead,
    ValueHead,
    ActionInterface,
    ActionIntent,
    PredictedTransition,
    VerificationHead,
)
from cir_arc.neural.reasoner.planner import CounterfactualPlanner, CandidateActionScore
from cir_arc.neural.world_state import HybridSceneState, DenseLatentState, SymbolicSceneState


class CognitiveReasoner120M(nn.Module):
    """The complete CIR-ARC 120.18M Direct Cognitive Reasoner."""

    def __init__(self, config: Optional[ReasonerConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ReasonerConfig()
        self.config = config

        # 1. Input Token Encoders & Dense Projections (5,095,936 parameters)
        self.entity_encoder = SymbolicEntityEncoder(
            d_model=config.d_model,
            num_colors=config.num_colors,
            num_shapes=config.num_shapes,
            num_orientations=config.num_orientations,
        )
        self.relation_encoder = RelationTokenEncoder(
            d_model=config.d_model,
            num_relations=config.num_relations,
        )
        self.event_encoder = EventTokenEncoder(
            d_model=config.d_model,
            num_events=config.num_events,
        )
        self.mechanics_encoder = MechanicsBeliefEncoder(d_model=config.d_model)
        self.global_encoder = GlobalStateEncoder(d_model=config.d_model)
        self.action_effect_encoder = ActionEffectEncoder(d_model=config.d_model)
        self.uncertainty_encoder = UncertaintyEncoder(d_model=config.d_model)
        self.dense_projections = DenseLatentProjections(
            slot_dim=config.slot_dim,
            feat_dim=config.feat_dim,
            d_model=config.d_model,
            num_spatial_tokens=config.num_spatial_tokens,
        )

        # 2. Dual Memory System: Ephemeral Workspace + Persistent Working Memory (3,544,832 parameters)
        self.memory_system = MemorySystem(
            d_model=config.d_model,
            num_reasoning=config.num_reasoning_tokens,
            num_wm=config.num_wm_tokens,
            retrieval_dim=config.retrieval_key_dim,
        )

        # 3. 18-Layer Transformer Trunk with GQA and SwiGLU (105,312,000 parameters)
        self.trunk = CognitiveTransformerTrunk(config)

        # 4. Cognitive Reasoning & Output Heads (6,226,592 parameters)
        self.goal_head = GoalInferenceHead(
            d_model=config.d_model,
            num_hypotheses=config.num_goal_hypotheses,
        )
        self.world_model = WorldModelHead(
            d_model=config.d_model,
            num_actions=config.num_actions,
        )
        self.value_head = ValueHead(
            d_model=config.d_model,
            val_hidden_dim=config.val_hidden_dim,
        )
        self.action_interface = ActionInterface(d_model=config.d_model)
        self.verification_head = VerificationHead(d_model=config.d_model)

        # 5. Counterfactual Planner (External tree search, 0 learnable parameters)
        self.planner = CounterfactualPlanner(
            world_model=self.world_model,
            value_head=self.value_head,
            action_interface=self.action_interface,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Returns exact parameter count broken down by module."""
        trunk_p = sum(p.numel() for p in self.trunk.parameters())
        input_fusion_p = (
            sum(p.numel() for p in self.entity_encoder.parameters())
            + sum(p.numel() for p in self.relation_encoder.parameters())
            + sum(p.numel() for p in self.event_encoder.parameters())
            + sum(p.numel() for p in self.mechanics_encoder.parameters())
            + sum(p.numel() for p in self.global_encoder.parameters())
            + sum(p.numel() for p in self.action_effect_encoder.parameters())
            + sum(p.numel() for p in self.uncertainty_encoder.parameters())
            + sum(p.numel() for p in self.dense_projections.parameters())
        )
        memory_p = sum(p.numel() for p in self.memory_system.parameters())
        heads_p = (
            sum(p.numel() for p in self.goal_head.parameters())
            + sum(p.numel() for p in self.world_model.parameters())
            + sum(p.numel() for p in self.value_head.parameters())
            + sum(p.numel() for p in self.action_interface.parameters())
            + sum(p.numel() for p in self.verification_head.parameters())
        )
        total_p = sum(p.numel() for p in self.parameters())
        return {
            "transformer_trunk": trunk_p,
            "input_fusion": input_fusion_p,
            "memory_system": memory_p,
            "cognitive_heads": heads_p,
            "total": total_p,
        }

    def construct_token_sequence(
        self,
        slot_embeddings: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
        rel_latents: Optional[torch.Tensor] = None,
        temp_latents: Optional[torch.Tensor] = None,
        global_vec: Optional[torch.Tensor] = None,
        working_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Assembles hybrid token stream for the Transformer trunk.

        Tokens:
            [GLOBAL] + [DENSE_SLOTS] + [SPATIAL_TOKENS] + [WORKING_MEMORY] + [REASONING_WORKSPACE]
        """
        B = slot_embeddings.shape[0]
        device = slot_embeddings.device

        # 1. Global state token [B, 1, 768]
        if global_vec is None:
            global_vec = torch.zeros(B, self.config.d_model, device=device, dtype=slot_embeddings.dtype)
        else:
            global_vec = global_vec.to(dtype=slot_embeddings.dtype)
        global_token = self.dense_projections.global_proj(global_vec).unsqueeze(1)

        # 2. Dense Slot tokens [B, K, 768]
        slot_tokens = self.dense_projections.slot_proj(slot_embeddings)

        # 3. Compressed Spatial tokens [B, 128, 768]
        if spatial_features is not None:
            comp_spatial = self.dense_projections.compress_spatial_features(spatial_features.to(dtype=slot_embeddings.dtype))
            spatial_tokens = self.dense_projections.spatial_proj(comp_spatial)
        else:
            spatial_tokens = torch.zeros(B, self.config.num_spatial_tokens, self.config.d_model, device=device, dtype=slot_embeddings.dtype)

        # 4. Working memory tokens [B, 128, 768]
        if working_memory is None:
            working_memory = self.memory_system.get_initial_working_memory(B, device).to(dtype=slot_embeddings.dtype)
        else:
            working_memory = working_memory.to(dtype=slot_embeddings.dtype)

        # 5. Fresh ephemeral reasoning tokens [B, 128, 768]
        reasoning_tokens = self.memory_system.get_initial_reasoning_workspace(B, device).to(dtype=slot_embeddings.dtype)

        # Composite sequence: ~280-500 tokens (well within the 8192 ceiling)
        tokens = torch.cat([global_token, slot_tokens, spatial_tokens, working_memory, reasoning_tokens], dim=1)
        return tokens

    def forward(
        self,
        slot_embeddings: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
        working_memory: Optional[torch.Tensor] = None,
        num_reasoning_passes: int = 1,
    ) -> Dict[str, Any]:
        """Forward cognitive reasoning pass.

        Args:
            slot_embeddings: Continuous slot representations [B, K, slot_dim].
            spatial_features: Dense CNN spatial features [B, H, W, feat_dim] or [B, N, feat_dim].
            working_memory: Persistent working memory from prior timestep [B, 128, d_model].
            num_reasoning_passes: Number of iterative thinking cycles (1 for fast, 2-8 for hard states).

        Returns:
            Dictionary containing:
                - cognitive_state: Pooled state representation [B, d_model].
                - updated_working_memory: Updated persistent memory [B, 128, d_model].
                - goal_hypotheses: Candidate goals [B, 4, d_model] and probabilities.
                - state_value: Predicted future state value [B, 1].
                - action_logits: Discrete action probabilities [B, 7].
                - pointer_logits: Object pointer scores [B, K].
        """
        B = slot_embeddings.shape[0]
        device = slot_embeddings.device

        # Assemble initial composite token sequence
        seq = self.construct_token_sequence(
            slot_embeddings=slot_embeddings,
            spatial_features=spatial_features,
            working_memory=working_memory,
        )

        # Iterative reasoning loop: passes token stream through the 18-layer trunk
        for _ in range(num_reasoning_passes):
            seq, _ = self.trunk(seq)

        # Extract components from transformed sequence
        # Index 0: Global token -> pooled cognitive state
        cognitive_state = seq[:, 0]

        # Extract entity slot tokens for pointer head localization
        K = slot_embeddings.shape[1]
        entity_tokens = seq[:, 1 : 1 + K]

        # Update persistent working memory using cross-attention
        prev_wm = working_memory if working_memory is not None else self.memory_system.get_initial_working_memory(B, device)
        updated_wm = self.memory_system.update_working_memory(seq, prev_wm)

        # Cognitive heads
        goals, goal_probs = self.goal_head(cognitive_state)
        best_goal = goals[:, 0]  # Most confident goal hypothesis

        value, risk_cost = self.value_head(cognitive_state, goal=best_goal)
        action_logits, pointer_logits, confidence = self.action_interface(cognitive_state, entity_tokens)

        return {
            "cognitive_state": cognitive_state,
            "entity_tokens": entity_tokens,
            "updated_working_memory": updated_wm,
            "goals": goals,
            "goal_confidence": goal_probs,
            "value": value,
            "risk_cost": risk_cost,
            "action_logits": action_logits,
            "pointer_logits": pointer_logits,
            "confidence": confidence,
        }

    def plan(
        self,
        slot_embeddings: torch.Tensor,
        candidate_actions: Optional[List[int]] = None,
        spatial_features: Optional[torch.Tensor] = None,
        working_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[ActionIntent, List[CandidateActionScore]]:
        """Executes counterfactual planning and selects optimal ActionIntent."""
        if candidate_actions is None:
            candidate_actions = [0, 1, 2, 3, 4, 6]  # Default active actions

        out = self.forward(
            slot_embeddings=slot_embeddings,
            spatial_features=spatial_features,
            working_memory=working_memory,
        )

        best_goal = out["goals"][:, 0]
        intent, scores = self.planner.plan_best_action(
            cognitive_state=out["cognitive_state"],
            candidate_actions=candidate_actions,
            goal_vector=best_goal,
            entity_tokens=out["entity_tokens"],
        )
        return intent, scores

    def verify_observation(
        self,
        predicted_latent: torch.Tensor,
        observed_latent: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """Compares prediction vs actual observation for belief revision."""
        error, gate = self.verification_head(predicted_latent, observed_latent)
        return float(error.item()), gate

    def forward_scene(
        self,
        scene: HybridSceneState,
        working_memory: Optional[torch.Tensor] = None,
        num_reasoning_passes: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Direct end-to-end cognitive reasoning over a HybridSceneState dataclass.

        Args:
            scene: Dual Neuro-Symbolic HybridSceneState containing SymbolicSceneState and DenseLatentState.
            working_memory: Persistent working memory from prior timestep [B, 128, d_model].
            num_reasoning_passes: Number of iterative thinking cycles (1-8).
            device: Target torch device.

        Returns:
            Dictionary containing cognitive state, updated working memory, goals, values, and action logits.
        """
        if device is None:
            device = next(self.parameters()).device

        # 1. Extract continuous dense slots
        slots = scene.dense.slot_embeddings
        if not isinstance(slots, torch.Tensor):
            slots = torch.tensor(slots, dtype=torch.float32, device=device)
        else:
            slots = slots.to(device=device)

        if slots.dim() == 2:
            slots = slots.unsqueeze(0)  # [1, K, slot_dim]

        # 2. Extract continuous spatial features if present
        spatial = scene.dense.spatial_features
        if spatial is not None:
            if not isinstance(spatial, torch.Tensor):
                spatial = torch.tensor(spatial, dtype=torch.float32, device=device)
            else:
                spatial = spatial.to(device=device)
            if spatial.dim() == 2:
                spatial = spatial.unsqueeze(0)
            elif spatial.dim() == 3 and spatial.shape[0] != slots.shape[0]:
                spatial = spatial.unsqueeze(0)

        # 3. Extract mechanics belief vector
        mb = scene.symbolic.mechanics_beliefs
        mb_vec = torch.tensor([
            float(mb.gravity[0]), float(mb.gravity[1]), float(mb.gravity_confidence),
            float(mb.friction), float(mb.collision_elasticity), float(mb.pushability_rule),
            float(mb.sliding_inertia), float(mb.screen_wrapping), float(mb.teleportation_active),
            float(mb.toggle_mechanics), float(mb.resource_mechanics),
        ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 11]

        # 4. Extract global HUD metrics
        gs = scene.symbolic.global_state
        hud_vec = torch.tensor([
            float(gs.lives), float(gs.energy), float(gs.score), float(gs.level_index),
            float(gs.timer if gs.timer is not None else 0), float(gs.switches_active),
            float(gs.doors_open), float(len(gs.inventory)),
        ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 8]
        inv_token = torch.tensor([gs.inventory[0] if gs.inventory else 0], dtype=torch.long, device=device)

        # Project mechanics and global HUD into 768-D summary vector
        mech_token = self.mechanics_encoder(mb_vec)
        global_token = self.global_encoder(hud_vec, inv_token)
        combined_global = (mech_token + global_token) / 2.0

        return self.forward(
            slot_embeddings=slots,
            spatial_features=spatial,
            working_memory=working_memory,
            num_reasoning_passes=num_reasoning_passes,
        )

