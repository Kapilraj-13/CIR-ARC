"""Solving runtime, model-based cognitive loop, and telemetry package."""

from cir_arc.solving.code_agent import CodeAgentSolver
from cir_arc.solving.cognitive_loop import CognitiveLoop, CognitiveContext, CognitiveStage
from cir_arc.solving.runtime import SolvingRuntime, ScorecardReport
from cir_arc.solving.search_solvers import AStarSolver, BFSSolver
from cir_arc.solving.telemetry import AgentOpsTelemetry

# Backwards compatibility alias
BFSPathSolver = BFSSolver

__all__ = [
    "CodeAgentSolver",
    "CognitiveLoop",
    "CognitiveContext",
    "CognitiveStage",
    "SolvingRuntime",
    "ScorecardReport",
    "AStarSolver",
    "BFSSolver",
    "BFSPathSolver",
    "AgentOpsTelemetry",
]
