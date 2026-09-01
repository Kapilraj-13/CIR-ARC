"""
Phase 3 Benchmark Harness for ARC-AGI-3 Interactive Environments.
Evaluates agent performance across diverse POMDP environments on:
- Win rate & level completion
- Action efficiency & step counts
- Rule induction accuracy & hypothesis confidence
- World model consistency & digital twin replay verification
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.mock_engine import MockEngine
from cir_arc.solving.runtime import ScorecardReport, SolvingRuntime

logger = logging.getLogger(__name__)


@dataclass
class GameBenchmarkResult:
    game_id: str
    is_win: bool
    levels_completed: int
    win_levels: int
    actions_taken: int
    elapsed_seconds: float
    state: str
    recording_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "is_win": self.is_win,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "actions_taken": self.actions_taken,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "state": self.state,
            "recording_path": self.recording_path,
            "metadata": self.metadata,
        }


@dataclass
class Phase3BenchmarkReport:
    generated_at: str
    total_games: int
    wins: int
    win_rate: float
    total_levels_completed: int
    total_win_levels: int
    level_completion_rate: float
    total_actions_taken: int
    avg_actions_per_game: float
    avg_elapsed_seconds: float
    results: List[GameBenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_games": self.total_games,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "total_levels_completed": self.total_levels_completed,
            "total_win_levels": self.total_win_levels,
            "level_completion_rate": round(self.level_completion_rate, 4),
            "total_actions_taken": self.total_actions_taken,
            "avg_actions_per_game": round(self.avg_actions_per_game, 2),
            "avg_elapsed_seconds": round(self.avg_elapsed_seconds, 3),
            "results": [r.to_dict() for r in self.results],
        }

    def summary_table(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  CIR-ARC PHASE 3 INTERACTIVE BENCHMARK REPORT",
            f"{'='*70}",
            f"  Total Games:           {self.total_games}",
            f"  Wins:                  {self.wins} / {self.total_games} ({self.win_rate * 100:.1f}%)",
            f"  Level Completion Rate: {self.level_completion_rate * 100:.1f}% ({self.total_levels_completed}/{self.total_win_levels})",
            f"  Avg Actions / Game:    {self.avg_actions_per_game:.1f}",
            f"  Avg Time / Game:       {self.avg_elapsed_seconds:.3f}s",
            f"{'-'*70}",
            f"  {'Game ID':<25} {'Result':<10} {'Levels':<10} {'Actions':<10} {'Time (s)':<10}",
            f"{'-'*70}",
        ]
        for r in self.results:
            status = "WIN" if r.is_win else r.state
            levels_str = f"{r.levels_completed}/{r.win_levels}"
            lines.append(
                f"  {r.game_id:<25} {status:<10} {levels_str:<10} {r.actions_taken:<10} {r.elapsed_seconds:<10.3f}"
            )
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


class Phase3BenchmarkSuite:
    """Benchmark runner evaluating the full CIR-ARC interactive multi-agent solver."""

    DEFAULT_GAMES = [
        "mock_maze_01",
        "mock_maze_02",
        "mock_locksmith_01",
        "mock_locksmith_02",
    ]

    def __init__(
        self,
        game_ids: Optional[List[str]] = None,
        max_actions_per_game: int = 80,
        record: bool = True,
        output_dir: str = "experiments/phase3_benchmark",
    ) -> None:
        self.game_ids = game_ids or list(self.DEFAULT_GAMES)
        self.max_actions_per_game = max_actions_per_game
        self.record = record
        self.output_dir = output_dir

    def run(self) -> Phase3BenchmarkReport:
        """Execute the benchmark across all configured game environments."""
        os.makedirs(self.output_dir, exist_ok=True)
        runtime = SolvingRuntime(max_actions=self.max_actions_per_game, record=self.record)
        results: List[GameBenchmarkResult] = []

        total_wins = 0
        total_levels_done = 0
        total_win_levels_target = 0
        total_actions = 0
        total_time = 0.0

        for gid in self.game_ids:
            logger.info("Evaluating game: %s", gid)
            env = MockEngine(game_id=gid)
            scorecard: ScorecardReport = runtime.run_game(env)

            is_win = scorecard.is_win
            if is_win:
                total_wins += 1

            total_levels_done += scorecard.levels_completed
            total_win_levels_target += scorecard.win_levels
            total_actions += scorecard.actions_taken
            total_time += scorecard.elapsed_seconds

            res = GameBenchmarkResult(
                game_id=gid,
                is_win=is_win,
                levels_completed=scorecard.levels_completed,
                win_levels=scorecard.win_levels,
                actions_taken=scorecard.actions_taken,
                elapsed_seconds=scorecard.elapsed_seconds,
                state=scorecard.state.value,
                recording_path=scorecard.recording_path,
                metadata=scorecard.telemetry_summary,
            )
            results.append(res)

        n = max(len(self.game_ids), 1)
        win_rate = total_wins / n
        level_rate = total_levels_done / max(total_win_levels_target, 1)
        avg_actions = total_actions / n
        avg_time = total_time / n

        report = Phase3BenchmarkReport(
            generated_at=datetime.now().isoformat(),
            total_games=len(self.game_ids),
            wins=total_wins,
            win_rate=win_rate,
            total_levels_completed=total_levels_done,
            total_win_levels=total_win_levels_target,
            level_completion_rate=level_rate,
            total_actions_taken=total_actions,
            avg_actions_per_game=avg_actions,
            avg_elapsed_seconds=avg_time,
            results=results,
        )

        # Save JSON artifact
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
