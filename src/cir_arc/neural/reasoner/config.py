"""Configuration for the CIR-ARC ~120.18M Direct Cognitive Reasoner."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReasonerConfig:
    """Configuration for the CIR-ARC 120.18M Direct Cognitive Reasoner."""

    # Transformer Trunk Architecture
    d_model: int = 768
    n_layers: int = 18
    n_q_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 1856                         # SwiGLU hidden dim calibrated for exact 120.18M target
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    max_context_len: int = 8192              # Maximum cognitive context window ceiling
    dropout: float = 0.0

    # Memory System (Dual Workspace: Ephemeral Reasoning vs Persistent Working Memory)
    num_reasoning_tokens: int = 128          # Ephemeral workspace R_t rewritten per reasoning pass
    num_wm_tokens: int = 128                 # Persistent working memory M_t across timesteps
    retrieval_key_dim: int = 256             # Dimension for episodic memory query/key projection

    # Hybrid Input Dimensions (from 3.5M Perception Model)
    slot_dim: int = 224                      # Stage D slot vector dimension
    feat_dim: int = 224                      # CNN feature token dimension
    rel_latent_dim: int = 64                 # Continuous relation latent dimension
    temp_latent_dim: int = 64                # Continuous temporal tracking latent dimension
    num_spatial_tokens: int = 128            # Compressed spatial feature tokens for the Reasoner

    # Symbolic Ontologies (incorporating explicit UNKNOWN / OTHER fallback capacity)
    num_colors: int = 12                     # 0-9 palette, 10 mask, 11 unknown
    num_shapes: int = 10                     # 0-7 canonical shapes, 8 unknown, 9 other
    num_orientations: int = 5                # 0-3 (0, 90, 180, 270 deg), 4 unknown
    num_relations: int = 16                  # 14 canonical predicates + UNKNOWN + OTHER
    num_events: int = 16                     # 14 transition events + UNKNOWN + OTHER
    num_affordances: int = 9                 # 9 canonical affordances
    num_actions: int = 8                     # 8 discrete actions: 0-7
    num_action_types: int = 7                # MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, ACTION, UNDO, CLICK

    # Cognitive Heads
    num_goal_hypotheses: int = 4             # Latent goal candidate hypotheses maintained in parallel
    goal_dim: int = 768                      # Latent goal vector dimension
    val_hidden_dim: int = 384                # Value head intermediate dimension

    # Precision & Execution Mode (Auto-selected for Kaggle / device)
    precision: str = "auto"                  # "auto", "bfloat16", "float16", "float32"
