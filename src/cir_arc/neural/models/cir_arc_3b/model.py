"""Unified CIR-ARC-3B Autonomous Scientific Discovery Agent (3,000,000,000 Parameters).

Complete architectural implementation unifying:
1. Perception & Temporal Hungarian Slot Tracker: 140,000,000
2. Game Model & Agent Self-Model: 100,000,000
3. Open-Ended Hypothesis Synthesizer (H_unknown): 140,000,000
4. Counterfactual World Model Ensemble: 320,000,000
5. Causal Program Graph & Attribution: 130,000,000
6. Belief-Space MPC Planner: 180,000,000
7. Unified Action Heads & Safety Gate: 115,000,000
8. Memory Tiers & Falsification Engine: 190,000,000
9. Token Interface & RoPE: 17,641,280
10. 24-Layer GQA-SwiGLU Reasoning Trunk: 1,667,358,720
Grand Total: Exactly 3,000,000,000 parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.neural.models.cir_arc_3b.config import CirArc3BConfig
from cir_arc.neural.models.cir_arc_3b.perception import PerceptionSlotTracker3B
from cir_arc.neural.models.cir_arc_3b.game_model import GameAndSelfModel3B
from cir_arc.neural.models.cir_arc_3b.hypothesis_engine import HypothesisSynthesizer3B
from cir_arc.neural.models.cir_arc_3b.world_model import WorldModelEnsemble3B
from cir_arc.neural.models.cir_arc_3b.causal_graph import CausalProgramGraph3B
from cir_arc.neural.models.cir_arc_3b.mpc_planner import BeliefMpcPlanner3B
from cir_arc.neural.models.cir_arc_3b.action_heads import UnifiedActionHeadsAndSafetyGate3B
from cir_arc.neural.models.cir_arc_3b.memory_falsifier import MemoryAndFalsificationEngine3B
from cir_arc.neural.models.cir_arc_3b.token_interface import TokenInterface3B
from cir_arc.neural.models.cir_arc_3b.reasoning_trunk import GQA_SwiGLU_Trunk3B


class CirArc3B(nn.Module):
    """The Complete 3.000B Parameter Autonomous Scientific Discovery Agent."""

    def __init__(self, config: Optional[CirArc3BConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = CirArc3BConfig()
        self.config = config

        # 1. Perception & Temporal Hungarian Slot Tracker (140M)
        self.perception = PerceptionSlotTracker3B(config)

        # 2. Game Model & Agent Self-Model (100M)
        self.game_model = GameAndSelfModel3B(config)

        # 3. Open-Ended Hypothesis Synthesizer (140M)
        self.hypothesis_engine = HypothesisSynthesizer3B(config)

        # 4. Counterfactual World Model Ensemble (320M)
        self.world_model = WorldModelEnsemble3B(config)

        # 5. Causal Program Graph & Attribution (130M)
        self.causal_graph = CausalProgramGraph3B(config)

        # 6. Belief-Space MPC Planner (180M)
        self.mpc_planner = BeliefMpcPlanner3B(config)

        # 7. Unified Action Heads & Safety Gate (115M)
        self.action_heads = UnifiedActionHeadsAndSafetyGate3B(config)

        # 8. Memory Tiers & Falsification Engine (190M)
        self.memory_falsifier = MemoryAndFalsificationEngine3B(config)

        # 9. Token Interface & RoPE (17.6M)
        self.token_interface = TokenInterface3B(config)

        # 10. 24-Layer GQA-SwiGLU Reasoning Trunk (1,667.4M)
        self.trunk = GQA_SwiGLU_Trunk3B(config)

    def count_parameters(self) -> Dict[str, int]:
        """Audit and return exact parameter counts across all 10 faculties."""
        counts = {
            "perception_slot_tracker": sum(p.numel() for p in self.perception.parameters()),
            "game_model_self_model": sum(p.numel() for p in self.game_model.parameters()),
            "hypothesis_synthesizer": sum(p.numel() for p in self.hypothesis_engine.parameters()),
            "world_model_ensemble": sum(p.numel() for p in self.world_model.parameters()),
            "causal_program_graph": sum(p.numel() for p in self.causal_graph.parameters()),
            "belief_mpc_planner": sum(p.numel() for p in self.mpc_planner.parameters()),
            "action_heads_safety_gate": sum(p.numel() for p in self.action_heads.parameters()),
            "memory_falsification_engine": sum(p.numel() for p in self.memory_falsifier.parameters()),
            "token_interface": sum(p.numel() for p in self.token_interface.parameters()),
            "reasoning_trunk": sum(p.numel() for p in self.trunk.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts

    def forward(
        self,
        grid_t: Optional[torch.Tensor] = None,
        grid_next: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        input_token_ids: Optional[torch.Tensor] = None,
        event_stream: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
    ) -> Dict[str, Any]:
        """Unified forward pass executing multi-stage cognitive reasoning."""
        B = grid_t.shape[0] if grid_t is not None else input_token_ids.shape[0]
        device = grid_t.device if grid_t is not None else input_token_ids.device

        # Stage 1: Perception & Object Slot Extraction
        if grid_t is not None:
            perc_out = self.perception(grid_t, grid_next)
            slot_tokens = perc_out["trunk_tokens"]
            slots = perc_out["slots"]
        else:
            # Fallback mock slots if processing discrete tokens only
            slot_tokens = torch.zeros(B, self.config.num_slots, self.config.d_model, device=device)
            slots = torch.zeros(B, self.config.num_slots, self.config.slot_dim, device=device)
            perc_out = {"slots": slots, "trunk_tokens": slot_tokens}

        # Stage 2: Token Interface & Sequence Construction
        if input_token_ids is not None:
            text_tokens = self.token_interface.embed_tokens(input_token_ids)
            trunk_inputs = torch.cat([slot_tokens, text_tokens], dim=1)
        else:
            trunk_inputs = slot_tokens

        # Stage 3: 24-Layer GQA-SwiGLU Reasoning Trunk
        trunk_hidden, _ = self.trunk(
            trunk_inputs,
            gradient_checkpointing=gradient_checkpointing,
        )
        trunk_hidden = self.token_interface.apply_final_norm(trunk_hidden)
        cognitive_state = trunk_hidden[:, 0, :]  # Global cognitive summary state

        # Stage 4: Agent Self-Model & Game Model
        game_res = self.game_model(slots, cognitive_state, action)
        reversibility = game_res["reversibility"]

        # Stage 5: Causal Program Graph & Attribution
        action_effect = cognitive_state
        causal_res = self.causal_graph(slots, action_effect)

        # Stage 6: Counterfactual World Model Ensemble
        if action is None:
            action = torch.zeros(B, dtype=torch.long, device=device)
        hypothesis_prior = cognitive_state
        wm_res = self.world_model(cognitive_state, action, hypothesis_prior)

        # Stage 7: Open-Ended Hypothesis Synthesis (Prediction Residual)
        target_observed = cognitive_state
        hypo_res = self.hypothesis_engine(target_observed, wm_res["predicted_state"])
        active_hypothesis = hypo_res["hypotheses"][:, 0, :]

        # Stage 8: Three-Tiered Memory & Invariant Falsification
        if event_stream is None:
            event_stream = cognitive_state.unsqueeze(1).expand(-1, self.config.temporal_event_window, -1)
        mem_res = self.memory_falsifier(cognitive_state, event_stream, active_hypothesis)

        # Stage 9: Belief-Space MPC Planner
        mpc_res = self.mpc_planner(cognitive_state, active_hypothesis)

        # Stage 10: Action Policy, Pointer Clicks, and Irreversible Action Safety Gate
        action_res = self.action_heads(cognitive_state, reversibility_score=reversibility)

        return {
            "cognitive_state": cognitive_state,
            "perception": perc_out,
            "game_model": game_res,
            "causal_graph": causal_res,
            "world_model": wm_res,
            "hypothesis": hypo_res,
            "memory": mem_res,
            "mpc_planner": mpc_res,
            "action_policy": action_res,
            "policy_logits": action_res["policy_logits"],
            "pointer_logits": action_res["pointer_logits"],
            "entrapment_risk": action_res["entrapment_risk"],
        }
