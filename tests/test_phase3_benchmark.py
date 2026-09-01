"""Tests for Phase 3 Interactive Benchmark Suite and Report Generation."""

from __future__ import annotations

import json
import os
import tempfile
import pytest

from cir_arc.eval.phase3_benchmark import (
    GameBenchmarkResult,
    Phase3BenchmarkReport,
    Phase3BenchmarkSuite,
)
from cir_arc.environment.mock_engine import MockEngine


class TestPhase3Benchmark:
    def test_single_game_benchmark_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Phase3BenchmarkSuite(
                game_ids=["mock_maze_01"],
                max_actions_per_game=40,
                record=True,
                output_dir=tmpdir,
            )
            report = suite.run()

            assert isinstance(report, Phase3BenchmarkReport)
            assert report.total_games == 1
            assert report.wins == 1
            assert report.win_rate == 1.0
            assert len(report.results) == 1

            res = report.results[0]
            assert isinstance(res, GameBenchmarkResult)
            assert res.game_id == "mock_maze_01"
            assert res.is_win is True
            assert res.actions_taken > 0
            assert res.recording_path is not None
            assert os.path.exists(res.recording_path)

            # Verify saved json artifact
            json_file = os.path.join(tmpdir, "benchmark_results.json")
            assert os.path.exists(json_file)
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_games"] == 1
            assert data["win_rate"] == 1.0

    def test_multi_game_benchmark_suite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            games = ["mock_maze_01", "mock_maze_02", "mock_locksmith_01", "mock_locksmith_02"]
            suite = Phase3BenchmarkSuite(
                game_ids=games,
                max_actions_per_game=70,
                record=False,
                output_dir=tmpdir,
            )
            report = suite.run()

            assert report.total_games == 4
            assert report.wins == 4
            assert report.win_rate == 1.0
            assert report.avg_actions_per_game > 0
            assert report.level_completion_rate == 1.0

            table_str = report.summary_table()
            assert "CIR-ARC PHASE 3 INTERACTIVE BENCHMARK REPORT" in table_str
            for g in games:
                assert g in table_str

    def test_report_serialization(self):
        report = Phase3BenchmarkReport(
            generated_at="2026-09-01T00:00:00",
            total_games=2,
            wins=2,
            win_rate=1.0,
            total_levels_completed=2,
            total_win_levels=2,
            level_completion_rate=1.0,
            total_actions_taken=24,
            avg_actions_per_game=12.0,
            avg_elapsed_seconds=0.05,
            results=[
                GameBenchmarkResult(
                    game_id="mock_maze_01",
                    is_win=True,
                    levels_completed=1,
                    win_levels=1,
                    actions_taken=10,
                    elapsed_seconds=0.02,
                    state="WIN",
                ),
                GameBenchmarkResult(
                    game_id="mock_locksmith_01",
                    is_win=True,
                    levels_completed=1,
                    win_levels=1,
                    actions_taken=14,
                    elapsed_seconds=0.03,
                    state="WIN",
                ),
            ],
        )
        d = report.to_dict()
        assert d["total_games"] == 2
        assert d["win_rate"] == 1.0
        assert len(d["results"]) == 2
