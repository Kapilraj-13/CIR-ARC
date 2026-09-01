"""
Benchmark Runner for CIR-ARC Phase 3 Interactive POMDP Environments.

Usage:
    python scripts/benchmark_phase3.py --output_dir experiments/phase3_benchmark
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cir_arc.eval.phase3_benchmark import Phase3BenchmarkSuite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Phase3Benchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIR-ARC Phase 3 Interactive Benchmark")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["mock_maze_01", "mock_maze_02", "mock_locksmith_01", "mock_locksmith_02"],
        help="List of game IDs to benchmark",
    )
    parser.add_argument("--max_actions", type=int, default=80, help="Max action budget per game")
    parser.add_argument("--record", action="store_true", default=True, help="Record gameplay traces")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/phase3_benchmark",
        help="Directory to save benchmark reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Phase 3 Benchmark Suite on %d games...", len(args.games))

    suite = Phase3BenchmarkSuite(
        game_ids=args.games,
        max_actions_per_game=args.max_actions,
        record=args.record,
        output_dir=args.output_dir,
    )

    report = suite.run()
    print(report.summary_table())
    logger.info("Benchmark complete. Results saved to %s/benchmark_results.json", args.output_dir)


if __name__ == "__main__":
    main()
