"""
CLI runner for the CIR-ARC Phase 3 Interactive Cognitive Agent.

Usage:
    python scripts/run_interactive_agent.py --game_id mock_maze_01 --max_actions 50 --record
    python scripts/run_interactive_agent.py --game_id mock_locksmith_01 --max_actions 80
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cir_arc.environment.mock_engine import MockEngine
from cir_arc.solving.runtime import SolvingRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CIR-ARC-Interactive")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIR-ARC Interactive Solving Runtime")
    parser.add_argument("--game_id", type=str, default="mock_maze_01", help="Game ID to solve")
    parser.add_argument("--max_actions", type=int, default=80, help="Maximum action budget")
    parser.add_argument("--record", action="store_true", default=True, help="Record session to .recording.jsonl")
    parser.add_argument("--output_dir", type=str, default="recordings", help="Output directory for recordings")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing game environment: %s", args.game_id)
    env = MockEngine(game_id=args.game_id)

    runtime = SolvingRuntime(max_actions=args.max_actions, record=args.record)
    logger.info("Starting Cognitive Loop Solving Runtime (Max Actions: %d)...", args.max_actions)

    scorecard = runtime.run_game(env)

    print("\n" + "=" * 60)
    print("  CIR-ARC INTERACTIVE SOLVER SCORECARD")
    print("=" * 60)
    print(f"  Game ID:          {scorecard.game_id}")
    print(f"  Outcome:          {'WIN' if scorecard.is_win else scorecard.state.value}")
    print(f"  Levels Completed: {scorecard.levels_completed} / {scorecard.win_levels}")
    print(f"  Actions Taken:    {scorecard.actions_taken}")
    print(f"  Elapsed Time:     {scorecard.elapsed_seconds:.3f} s")
    if scorecard.recording_path:
        print(f"  Session Log:      {scorecard.recording_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
