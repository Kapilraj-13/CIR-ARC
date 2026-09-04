"""Neural modules and architectures for CIR-ARC Phase 2."""

from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import CNNStem
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.property_heads import PropertyHeads
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder
from cir_arc.neural.world_state import (
    HybridSceneState,
    DenseLatentState,
    SymbolicSceneState,
    StructuredObject,
    SpatialRelation,
    SemanticEvent,
    MechanicsBelief,
    GlobalStateData,
)
from cir_arc.neural.reasoner import (
    CognitiveReasoner120M,
    ReasonerConfig,
    ActionIntent,
    PredictedTransition,
    ReasonerMultiObjectiveLoss,
)

__all__ = [
    "ColorEmbedding",
    "CNNStem",
    "SlotAttention",
    "PropertyHeads",
    "ReconstructionDecoder",
    "HybridSceneState",
    "DenseLatentState",
    "SymbolicSceneState",
    "StructuredObject",
    "SpatialRelation",
    "SemanticEvent",
    "MechanicsBelief",
    "GlobalStateData",
    "CognitiveReasoner120M",
    "ReasonerConfig",
    "ActionIntent",
    "PredictedTransition",
    "ReasonerMultiObjectiveLoss",
]
