"""Neural perception subpackage for CIR-ARC."""

from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import (
    CNNStem,
    MultiScaleCNNStem,
    ResidualDepthwiseSeparableConv,
    DepthwiseSeparableConv,
)
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.relation_encoder import SlotRelationEncoder, SetTransformerBlock
from cir_arc.neural.perception.property_heads import PropertyHeads
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder, SlotMaskDecoder

__all__ = [
    "ColorEmbedding",
    "CNNStem",
    "MultiScaleCNNStem",
    "ResidualDepthwiseSeparableConv",
    "DepthwiseSeparableConv",
    "SlotAttention",
    "SlotRelationEncoder",
    "SetTransformerBlock",
    "PropertyHeads",
    "ReconstructionDecoder",
    "SlotMaskDecoder",
]
