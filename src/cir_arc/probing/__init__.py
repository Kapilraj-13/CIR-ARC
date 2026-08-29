"""Active probing, state machine extraction, and information-gain exploration."""

from cir_arc.probing.action_matrix import ActionEffectMatrix, ActionEffect
from cir_arc.probing.inspector_agent import EnvironmentInspector, EnvironmentProfile
from cir_arc.probing.object_catalog import DynamicObjectCatalog, ObjectArchetype
from cir_arc.probing.resource_inspector import ResourceInspector
from cir_arc.probing.state_machine import GameStateMachine, GamePhase
from cir_arc.probing.information_gain import InformationGainExplorer

# Aliases for backwards compatibility
ActionDeltaMatrix = ActionEffectMatrix
ActionMatrixEntry = ActionEffect
SystematicInspectorAgent = EnvironmentInspector
ObjectCatalog = DynamicObjectCatalog
ProbedObjectEntry = ObjectArchetype
ProbedStateMachine = GameStateMachine
TransitionEdge = GamePhase

__all__ = [
    "ActionEffectMatrix",
    "ActionEffect",
    "ActionDeltaMatrix",
    "ActionMatrixEntry",
    "EnvironmentInspector",
    "EnvironmentProfile",
    "SystematicInspectorAgent",
    "DynamicObjectCatalog",
    "ObjectArchetype",
    "ObjectCatalog",
    "ProbedObjectEntry",
    "ResourceInspector",
    "GameStateMachine",
    "GamePhase",
    "ProbedStateMachine",
    "TransitionEdge",
    "InformationGainExplorer",
]
