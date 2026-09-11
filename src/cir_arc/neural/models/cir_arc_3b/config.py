"""Configuration for the CIR-ARC-3B Autonomous Scientific Discovery Agent.

Audited 3,000,000,000 parameter budget:
- Trunk: 24 Layers GQA-SwiGLU (d=2560, Hq=20, Hkv=5, d_ff=6912) -> 1,667,358,720
- Specialized Faculties:
  1. Perception & Temporal Hungarian Slot Tracker: 140,000,000
  2. Game Model & Agent Self-Model: 100,000,000
  3. Open-Ended Hypothesis Synthesizer (H_unknown): 140,000,000
  4. Counterfactual World Model Ensemble: 320,000,000
  5. Causal Program Graph & Attribution: 130,000,000
  6. Belief-Space MPC Planner: 180,000,000
  7. Unified Action Heads & Safety Gate: 115,000,000
  8. Memory Tiers & Falsification Engine: 190,000,000
  9. Token Interface & RoPE: 17,641,280
- Total: Exactly 3,000,000,000 parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CirArc3BConfig:
    """Master configuration for CIR-ARC-3B."""

    # 1. 24-Layer GQA-SwiGLU Reasoning Trunk
    d_model: int = 2560
    n_layers: int = 24
    n_q_heads: int = 20
    n_kv_heads: int = 5
    head_dim: int = 128                      # 20 * 128 = 2560 Q dim, 5 * 128 = 640 KV dim
    d_ff: int = 6912                         # SwiGLU intermediate dimension
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    max_context_len: int = 8192
    dropout: float = 0.0

    # 2. Perception & Hungarian Slot Tracker
    num_slots: int = 64                      # Persistent object slots
    slot_dim: int = 1024                     # Internal slot representation dimension
    cnn_channels: int = 256                  # Base CNN stem channel capacity
    grid_size: int = 32                      # Maximum canonical grid height/width
    num_input_colors: int = 11               # Colors 0-9 + pad/mask

    # 3. Game & Agent Self-Model
    num_affordances: int = 12                # push, pull, slide, unlock, enter, step, toggle, climb, carry, cast, teleport, wait
    reversibility_dim: int = 256

    # 4. Hypothesis Synthesizer
    num_hypotheses: int = 8                  # Active hypothesis pool size
    hypothesis_dim: int = 2560               # Latent hypothesis token dimension
    novelty_threshold: float = 0.45          # Residual norm trigger for H_unknown synthesis

    # 5. Counterfactual World Model Ensemble
    num_world_models: int = 4                # Ensemble members
    wm_hidden_dim: int = 2048                # Dynamics hidden dimension
    planning_horizon: int = 16               # Rollout step horizon

    # 6. Causal Program Graph & Attribution
    max_causal_nodes: int = 64
    max_attribution_depth: int = 4           # a_t -> delta S_1 -> delta S_2 -> delta S_3 -> delta S_4
    num_program_primitives: int = 64

    # 7. Belief-Space MPC Planner
    mpc_candidates: int = 32                 # Number of candidate action sequences sampled per step
    macro_action_vocab: int = 32

    # 8. Unified Action Heads & Safety Gate
    vocab_size: int = 4128                   # Discrete multimodal token vocabulary
    action_space_size: int = 4102            # 8 discrete actions + 1024 click pointers + macro + reasoning tokens
    safety_threshold_irrev: float = 0.85     # Risk gating threshold for irreversible transitions

    # 9. Memory Tiers & Falsification Engine
    episodic_memory_capacity: int = 512
    semantic_invariant_capacity: int = 256
    procedural_skill_capacity: int = 128
    temporal_event_window: int = 16

    # Precision & Execution
    precision: str = "bfloat16"              # "bfloat16" for TPU v3-8 / GPU, "float32" for offline CPU debug

    # Exact Parameter Audits
    BUDGET_TRUNK: int = 1_667_358_720
    BUDGET_PERCEPTION: int = 140_000_000
    BUDGET_GAME_MODEL: int = 100_000_000
    BUDGET_HYPOTHESIS: int = 140_000_000
    BUDGET_WORLD_MODEL: int = 320_000_000
    BUDGET_CAUSAL_GRAPH: int = 130_000_000
    BUDGET_MPC_PLANNER: int = 180_000_000
    BUDGET_ACTION_HEADS: int = 115_000_000
    BUDGET_MEMORY_FALSIFIER: int = 190_000_000
    BUDGET_TOKEN_INTERFACE: int = 17_641_280
    BUDGET_TOTAL: int = 3_000_000_000
