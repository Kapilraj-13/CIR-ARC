"""
Phase 1 benchmark runner.
Usage: python scripts/run_benchmark.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from cir_arc.eval.benchmark import run_benchmark
from cir_arc.baselines.random_agent import RandomColorAgent, CopyInputAgent
from cir_arc.baselines.heuristic_agent import HeuristicAgent


def main():
    agents = [
        RandomColorAgent(),
        CopyInputAgent(),
        HeuristicAgent(),
    ]

    data_dirs = {}
    synthetic_held = Path("data/synthetic/held_out")
    if synthetic_held.exists():
        data_dirs["synthetic_held_out"] = str(synthetic_held)

    if not data_dirs:
        print("No data directories found. Run 'python scripts/generate_data.py' first.")
        return

    run_benchmark(
        agents=agents,
        data_dirs=data_dirs,
        output_path=Path("experiments/phase1/benchmark_FROZEN.json")
    )


if __name__ == "__main__":
    main()
