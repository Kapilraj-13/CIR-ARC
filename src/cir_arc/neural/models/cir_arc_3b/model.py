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
        p_dev = next(self.perception.parameters()).device
        if grid_t is not None:
            perc_out = self.perception(grid_t.to(p_dev), grid_next.to(p_dev) if grid_next is not None else None)
            slot_tokens = perc_out["trunk_tokens"]
            slots = perc_out["slots"]
        else:
            # Fallback mock slots if processing discrete tokens only
            slot_tokens = torch.zeros(B, self.config.num_slots, self.config.d_model, device=p_dev)
            slots = torch.zeros(B, self.config.num_slots, self.config.slot_dim, device=p_dev)
            perc_out = {"slots": slots, "trunk_tokens": slot_tokens}

        # Stage 2: Token Interface & Sequence Construction
        ti_dev = next(self.token_interface.parameters()).device
        if input_token_ids is not None:
            text_tokens = self.token_interface.embed_tokens(input_token_ids.to(ti_dev))
            trunk_inputs = torch.cat([slot_tokens.to(ti_dev), text_tokens], dim=1)
        else:
            trunk_inputs = slot_tokens.to(ti_dev)

        # Stage 3: 24-Layer GQA-SwiGLU Reasoning Trunk
        trunk_hidden, _ = self.trunk(
            trunk_inputs,
            gradient_checkpointing=gradient_checkpointing,
        )
        fn_dev = self.token_interface.final_norm.weight.device
        trunk_hidden = self.token_interface.apply_final_norm(trunk_hidden.to(fn_dev))
        cognitive_state = trunk_hidden[:, 0, :]  # Global cognitive summary state

        # Stage 4: Agent Self-Model & Game Model
        gm_dev = next(self.game_model.parameters()).device
        act_gm = action.to(gm_dev) if action is not None else None
        game_res = self.game_model(slots.to(gm_dev), cognitive_state.to(gm_dev), act_gm)
        reversibility = game_res["reversibility"]

        # Stage 5: Causal Program Graph & Attribution
        cg_dev = next(self.causal_graph.parameters()).device
        action_effect = cognitive_state.to(cg_dev)
        causal_res = self.causal_graph(slots.to(cg_dev), action_effect)

        # Stage 6: Counterfactual World Model Ensemble
        wm_dev = next(self.world_model.parameters()).device
        act_wm = action.to(wm_dev) if action is not None else torch.zeros(B, dtype=torch.long, device=wm_dev)
        hypothesis_prior = cognitive_state.to(wm_dev)
        wm_res = self.world_model(cognitive_state.to(wm_dev), act_wm, hypothesis_prior)

        # Stage 7: Open-Ended Hypothesis Synthesis (Prediction Residual)
        he_dev = next(self.hypothesis_engine.parameters()).device
        target_observed = cognitive_state.to(he_dev)
        hypo_res = self.hypothesis_engine(target_observed, wm_res["predicted_state"].to(he_dev))
        active_hypothesis = hypo_res["hypotheses"][:, 0, :]

        # Stage 8: Three-Tiered Memory & Invariant Falsification
        mem_dev = next(self.memory_falsifier.parameters()).device
        cog_mem = cognitive_state.to(mem_dev)
        hypo_mem = active_hypothesis.to(mem_dev)
        if event_stream is None:
            event_stream = cog_mem.unsqueeze(1).expand(-1, self.config.temporal_event_window, -1)
        else:
            event_stream = event_stream.to(mem_dev)
        mem_res = self.memory_falsifier(cog_mem, event_stream, hypo_mem)

        # Stage 9: Belief-Space MPC Planner
        mpc_dev = next(self.mpc_planner.parameters()).device
        mpc_res = self.mpc_planner(cognitive_state.to(mpc_dev), active_hypothesis.to(mpc_dev))

        # Stage 10: Action Policy, Pointer Clicks, and Irreversible Action Safety Gate
        ah_dev = next(self.action_heads.parameters()).device
        rev_ah = reversibility.to(ah_dev) if reversibility is not None else None
        action_res = self.action_heads(cognitive_state.to(ah_dev), reversibility_score=rev_ah)

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

    def parallelize(self, dev0: str = "cuda:0", dev1: str = "cuda:1") -> CirArc3B:
        """Pipeline CirArc3B across two GPUs (Tesla T4 x 2)."""
        d0 = torch.device(dev0)
        d1 = torch.device(dev1)

        # GPU 0: Perception + Token Interface + Trunk Layers 0-11 (~3.0 GB parameters)
        self.perception.to(d0)
        self.token_interface.to(d0)
        for i in range(12):
            self.trunk.layers[i].to(d0)

        # GPU 1: Trunk Layers 12-23 + Final Norm + Cognitive Modules (~3.0 GB parameters)
        for i in range(12, 24):
            self.trunk.layers[i].to(d1)
        self.token_interface.final_norm.to(d1)
        self.game_model.to(d1)
        self.causal_graph.to(d1)
        self.world_model.to(d1)
        self.hypothesis_engine.to(d1)
        self.memory_falsifier.to(d1)
        self.mpc_planner.to(d1)
        self.action_heads.to(d1)
        return self

