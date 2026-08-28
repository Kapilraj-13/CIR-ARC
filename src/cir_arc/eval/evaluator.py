from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask


def cell_exact_match(predicted: Grid, ground_truth: Grid) -> bool:
    """
    ARC's real metric: full grid must match exactly.
    Returns True only if shape AND all cell values match.
    """
    if predicted.shape != ground_truth.shape:
        return False
    return np.array_equal(predicted.data, ground_truth.data)


@dataclass
class TaskResult:
    task_id: str
    rule_type: Optional[str]
    source: str
    difficulty: Optional[int]
    correct: bool
    predicted: Optional[Grid]
    ground_truth: Optional[Grid]


@dataclass
class EvalReport:
    total: int
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    by_rule_type: Dict[str, dict]
    by_source: Dict[str, dict]
    by_difficulty: Dict[int, dict]

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  EVALUATION REPORT",
            f"{'='*60}",
            f"  Total tasks:  {self.total}",
            f"  Correct:      {self.correct}",
            f"  Accuracy:     {self.accuracy:.3f}  "
            f"(95% CI: [{self.ci_low:.3f}, {self.ci_high:.3f}])",
            f"\n  By Rule Type:",
        ]
        for rule, stats in sorted(self.by_rule_type.items()):
            lines.append(
                f"    {rule:<35} {stats['accuracy']:.3f}  "
                f"({stats['correct']}/{stats['total']})"
            )
        lines.append(f"\n  By Source:")
        for src, stats in sorted(self.by_source.items()):
            lines.append(
                f"    {src:<20} {stats['accuracy']:.3f}  "
                f"({stats['correct']}/{stats['total']})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def wilson_interval(n: int, k: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


class Evaluator:
    def __init__(self):
        self.results: List[TaskResult] = []

    def evaluate_task(
        self,
        task: ArcTask,
        predicted_outputs: List[Grid]
    ) -> List[TaskResult]:
        results = []
        for i, (test_pair, predicted) in enumerate(
            zip(task.test_pairs, predicted_outputs)
        ):
            if test_pair.output is None:
                continue
            correct = cell_exact_match(predicted, test_pair.output)
            result = TaskResult(
                task_id=f"{task.task_id}_test{i}",
                rule_type=task.rule_type,
                source=task.source,
                difficulty=task.difficulty,
                correct=correct,
                predicted=predicted,
                ground_truth=test_pair.output,
            )
            results.append(result)
            self.results.append(result)
        return results

    def compute_report(self) -> EvalReport:
        total = len(self.results)
        correct = sum(r.correct for r in self.results)
        accuracy = correct / total if total > 0 else 0.0
        ci_low, ci_high = wilson_interval(total, correct)

        def aggregate(results):
            t = len(results)
            c = sum(r.correct for r in results)
            lo, hi = wilson_interval(t, c)
            return {"total": t, "correct": c,
                    "accuracy": c / t if t else 0.0,
                    "ci_low": lo, "ci_high": hi}

        rule_groups: Dict[str, List] = {}
        for r in self.results:
            key = r.rule_type or "unknown"
            rule_groups.setdefault(key, []).append(r)

        src_groups: Dict[str, List] = {}
        for r in self.results:
            src_groups.setdefault(r.source, []).append(r)

        diff_groups: Dict[int, List] = {}
        for r in self.results:
            key = r.difficulty or 0
            diff_groups.setdefault(key, []).append(r)

        return EvalReport(
            total=total,
            correct=correct,
            accuracy=accuracy,
            ci_low=ci_low,
            ci_high=ci_high,
            by_rule_type={k: aggregate(v) for k, v in rule_groups.items()},
            by_source={k: aggregate(v) for k, v in src_groups.items()},
            by_difficulty={k: aggregate(v) for k, v in diff_groups.items()},
        )

    def reset(self):
        self.results = []
