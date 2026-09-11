"""CIR-ARC-3B Autonomous Scientific Discovery Agent Module Suite.

Fully audited 3,000,000,000 parameter architecture for ARC-AGI-3.
"""

from __future__ import annotations

from cir_arc.neural.models.cir_arc_3b.config import CirArc3BConfig
from cir_arc.neural.models.cir_arc_3b.model import CirArc3B
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

__all__ = [
    "CirArc3BConfig",
    "CirArc3B",
    "PerceptionSlotTracker3B",
    "GameAndSelfModel3B",
    "HypothesisSynthesizer3B",
    "WorldModelEnsemble3B",
    "CausalProgramGraph3B",
    "BeliefMpcPlanner3B",
    "UnifiedActionHeadsAndSafetyGate3B",
    "MemoryAndFalsificationEngine3B",
    "TokenInterface3B",
    "GQA_SwiGLU_Trunk3B",
]
