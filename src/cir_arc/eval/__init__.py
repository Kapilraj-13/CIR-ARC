"""Evaluation and benchmarking package for CIR-ARC."""

from cir_arc.eval.evaluator import Evaluator, EvalReport, TaskResult, cell_exact_match
from cir_arc.eval.benchmark import run_benchmark
from cir_arc.eval.phase3_benchmark import (
    Phase3BenchmarkSuite,
    Phase3BenchmarkReport,
    GameBenchmarkResult,
)

__all__ = [
    "Evaluator",
    "EvalReport",
    "TaskResult",
    "cell_exact_match",
    "run_benchmark",
    "Phase3BenchmarkSuite",
    "Phase3BenchmarkReport",
    "GameBenchmarkResult",
]
