"""
Phase 1 benchmark runner.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Protocol
import json
from datetime import datetime

from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask
from cir_arc.eval.evaluator import Evaluator


class Agent(Protocol):
    name: str
    def predict(self, task: ArcTask) -> List[Grid]: ...


def run_benchmark(
    agents: List,
    data_dirs: dict,
    output_path: Path
) -> dict:
    report = {
        "generated_at": datetime.now().isoformat(),
        "results": {}
    }

    for dataset_label, data_dir in data_dirs.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_label}")
        print(f"{'='*50}")

        tasks = []
        data_path = Path(data_dir)
        if data_path.exists():
            for json_path in data_path.rglob("*.json"):
                try:
                    tasks.append(ArcTask.load(json_path))
                except Exception as e:
                    print(f"  Skipping {json_path}: {e}")

        print(f"Loaded {len(tasks)} tasks")

        report["results"][dataset_label] = {}

        for agent in agents:
            print(f"\n  Agent: {agent.name}")
            evaluator = Evaluator()

            for task in tasks:
                try:
                    predictions = agent.predict(task)
                    evaluator.evaluate_task(task, predictions)
                except Exception as e:
                    print(f"    Error on {task.task_id}: {e}")

            eval_report = evaluator.compute_report()
            print(eval_report)

            report["results"][dataset_label][agent.name] = {
                "total": eval_report.total,
                "correct": eval_report.correct,
                "accuracy": eval_report.accuracy,
                "ci_low": eval_report.ci_low,
                "ci_high": eval_report.ci_high,
                "by_rule_type": eval_report.by_rule_type,
            }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFrozen benchmark saved to {output_path}")

    return report
