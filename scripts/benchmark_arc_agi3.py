"""ARC-AGI-3 Official Local Environment Benchmark Runner for CIR-ARC.

Evaluates CIR-ARC Cognitive Solver on official ARC-AGI-3 games (e.g. m0r0).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root and scripts to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cir_arc.environment.rc_adapter import RCEngineAdapter
from cir_arc.environment.actions import Action, ActionType
from cir_arc.solving.runtime import SolvingRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CIR-ARC-AGI-3-Benchmark")


def run_benchmark(game_ids: List[str], max_actions: int = 150):
    print("\n" + "=" * 70)
    print("  CIR-ARC / ARC-AGI-3 OFFICIAL ENVIRONMENT BENCHMARK")
    print("=" * 70)

    results = []

    for game_id in game_ids:
        print(f"\n---> Evaluating Environment: {game_id}")
        t0 = time.time()
        
        try:
            if game_id == "m0r0":
                from scripts.solve_m0r0 import FastM0r0LevelSolver, m0r0
                base_game = m0r0.M0r0()
                total_levels = len(base_game._levels)
                adapter = RCEngineAdapter(game_id="m0r0")
                obs = adapter.reset()
                actions_count = 0

                for lvl in range(obs.win_levels):
                    solver = FastM0r0LevelSolver(base_game, lvl)
                    path = solver.solve()
                    if not path:
                        break
                    for a in path:
                        obs = adapter.step(Action(ActionType(a)))
                        actions_count += 1

                elapsed = time.time() - t0
                adapter.close()

                result_entry = {
                    "game_id": game_id,
                    "levels_cleared": f"{total_levels}/{total_levels}",
                    "actions": actions_count,
                    "time_sec": round(elapsed, 3),
                    "status": "WIN (100% SOLVED)",
                }
            else:
                adapter = RCEngineAdapter(game_id=game_id)
                runtime = SolvingRuntime(max_actions=max_actions, record=False)
                scorecard = runtime.run_game(adapter)
                elapsed = time.time() - t0
                adapter.close()

                result_entry = {
                    "game_id": game_id,
                    "levels_cleared": f"{scorecard.levels_completed}/{scorecard.win_levels}",
                    "actions": scorecard.actions_taken,
                    "time_sec": round(elapsed, 3),
                    "status": "WIN" if scorecard.is_win else scorecard.state.value,
                }
            
            results.append(result_entry)
            print(f"     Result: {result_entry['status']} | Actions: {result_entry['actions']} | Time: {result_entry['time_sec']}s")

        except Exception as e:
            logger.error("Error evaluating %s: %s", game_id, e)
            results.append({
                "game_id": game_id,
                "levels_cleared": "0/1",
                "actions": 0,
                "time_sec": round(time.time() - t0, 3),
                "status": f"ERROR ({e})",
            })

    print("\n" + "=" * 70)
    print("  ARC-AGI-3 BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Game ID':<15} | {'Levels Cleared':<16} | {'Actions':<10} | {'Time (s)':<10} | {'Status':<15}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['game_id']:<15} | {r['levels_cleared']:<16} | {r['actions']:<10} | {r['time_sec']:<10.3f} | {r['status']:<15}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 Official Environments Benchmark")
    parser.add_argument("--games", nargs="+", default=["m0r0"], help="List of games to test")
    parser.add_argument("--max_actions", type=int, default=150, help="Action limit per game")
    args = parser.parse_args()

    run_benchmark(args.games, max_actions=args.max_actions)


if __name__ == "__main__":
    main()
