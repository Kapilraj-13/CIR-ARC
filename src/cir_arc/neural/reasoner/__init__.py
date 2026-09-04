"""CIR-ARC ~120.18M Direct Cognitive Reasoner subpackage."""

from cir_arc.neural.reasoner.config import ReasonerConfig
from cir_arc.neural.reasoner.rope import RotaryEmbedding
from cir_arc.neural.reasoner.gqa import GroupedQueryAttention
from cir_arc.neural.reasoner.swiglu import SwiGLU
from cir_arc.neural.reasoner.block import RMSNorm, CognitiveTransformerBlock
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
    PredictedTransition,
    ActionIntent,
    GoalInferenceHead,
    WorldModelHead,
    ValueHead,
    ActionInterface,
    VerificationHead,
)
from cir_arc.neural.reasoner.planner import CounterfactualPlanner, CandidateActionScore
from cir_arc.neural.reasoner.losses import ReasonerLossWeights, ReasonerMultiObjectiveLoss
from cir_arc.neural.reasoner.model import CognitiveReasoner120M

__all__ = [
    "ReasonerConfig",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "RMSNorm",
    "CognitiveTransformerBlock",
    "CognitiveTransformerTrunk",
    "SymbolicEntityEncoder",
    "RelationTokenEncoder",
    "EventTokenEncoder",
    "MechanicsBeliefEncoder",
    "GlobalStateEncoder",
    "ActionEffectEncoder",
    "UncertaintyEncoder",
    "DenseLatentProjections",
    "MemorySystem",
    "PredictedTransition",
    "ActionIntent",
    "GoalInferenceHead",
    "WorldModelHead",
    "ValueHead",
    "ActionInterface",
    "VerificationHead",
    "CounterfactualPlanner",
    "CandidateActionScore",
    "ReasonerLossWeights",
    "ReasonerMultiObjectiveLoss",
    "CognitiveReasoner120M",
]
