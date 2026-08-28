import pytest
import numpy as np

from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask, GridPair
from cir_arc.eval.evaluator import Evaluator, cell_exact_match, wilson_interval
from cir_arc.baselines.random_agent import RandomColorAgent, CopyInputAgent
from cir_arc.baselines.heuristic_agent import HeuristicAgent


class TestCellExactMatch:
    def test_exact_match(self):
        a = Grid(np.array([[1, 2], [3, 4]]))
        b = Grid(np.array([[1, 2], [3, 4]]))
        assert cell_exact_match(a, b) is True

    def test_mismatch_values(self):
        a = Grid(np.array([[1, 2], [3, 4]]))
        b = Grid(np.array([[1, 2], [3, 5]]))
        assert cell_exact_match(a, b) is False

    def test_mismatch_shape(self):
        a = Grid(np.array([[1, 2]]))
        b = Grid(np.array([[1], [2]]))
        assert cell_exact_match(a, b) is False


class TestWilsonInterval:
    def test_all_correct(self):
        low, high = wilson_interval(100, 100)
        assert low > 0.95
        assert high == pytest.approx(1.0)

    def test_none_correct(self):
        low, high = wilson_interval(100, 0)
        assert low == 0.0
        assert high < 0.05

    def test_empty(self):
        low, high = wilson_interval(0, 0)
        assert low == 0.0
        assert high == 1.0


class TestEvaluator:
    def _make_simple_task(self, correct: bool):
        inp = Grid(np.array([[1, 2], [3, 4]]))
        gt = Grid(np.array([[4, 3], [2, 1]]))
        if correct:
            pred = Grid(np.array([[4, 3], [2, 1]]))
        else:
            pred = Grid(np.array([[1, 1], [1, 1]]))

        task = ArcTask(
            task_id="eval_test",
            source="synthetic",
            rule_type="test_rule",
            difficulty=1,
            train_pairs=[],
            test_pairs=[GridPair(input=inp, output=gt)],
        )
        return task, [pred]

    def test_evaluate_correct(self):
        evaluator = Evaluator()
        task, preds = self._make_simple_task(correct=True)
        results = evaluator.evaluate_task(task, preds)
        assert len(results) == 1
        assert results[0].correct is True

    def test_evaluate_incorrect(self):
        evaluator = Evaluator()
        task, preds = self._make_simple_task(correct=False)
        results = evaluator.evaluate_task(task, preds)
        assert results[0].correct is False

    def test_report_accuracy(self):
        evaluator = Evaluator()
        for _ in range(7):
            task, preds = self._make_simple_task(correct=True)
            evaluator.evaluate_task(task, preds)
        for _ in range(3):
            task, preds = self._make_simple_task(correct=False)
            evaluator.evaluate_task(task, preds)

        report = evaluator.compute_report()
        assert report.total == 10
        assert report.correct == 7
        assert abs(report.accuracy - 0.7) < 0.01

    def test_report_by_rule_type(self):
        evaluator = Evaluator()
        task, preds = self._make_simple_task(correct=True)
        evaluator.evaluate_task(task, preds)
        report = evaluator.compute_report()
        assert "test_rule" in report.by_rule_type

    def test_reset(self):
        evaluator = Evaluator()
        task, preds = self._make_simple_task(correct=True)
        evaluator.evaluate_task(task, preds)
        evaluator.reset()
        assert len(evaluator.results) == 0


class TestBaselines:
    @pytest.fixture
    def simple_task(self):
        return ArcTask(
            task_id="baseline_test",
            source="synthetic",
            rule_type="reflect_horizontal",
            difficulty=1,
            train_pairs=[
                GridPair(
                    input=Grid(np.array([[1, 2], [3, 4]])),
                    output=Grid(np.array([[3, 4], [1, 2]]))
                )
            ],
            test_pairs=[
                GridPair(
                    input=Grid(np.array([[5, 6], [7, 8]])),
                    output=Grid(np.array([[7, 8], [5, 6]]))
                )
            ]
        )

    def test_random_agent_returns_correct_shape(self, simple_task):
        agent = RandomColorAgent(seed=42)
        preds = agent.predict(simple_task)
        assert len(preds) == 1
        assert preds[0].shape == simple_task.test_pairs[0].input.shape

    def test_copy_input_agent(self, simple_task):
        agent = CopyInputAgent()
        preds = agent.predict(simple_task)
        assert preds[0] == simple_task.test_pairs[0].input

    def test_heuristic_agent_solves_reflect(self, simple_task):
        agent = HeuristicAgent()
        preds = agent.predict(simple_task)
        assert preds[0] == simple_task.test_pairs[0].output

    def test_heuristic_agent_solves_color_swap(self):
        # Train pair must NOT accidentally match reflect_horizontal or other rules
        task = ArcTask(
            task_id="swap_test",
            source="synthetic",
            rule_type="color_swap_all",
            rule_params={"color_a": 1, "color_b": 2},
            difficulty=1,
            train_pairs=[
                GridPair(
                    input=Grid(np.array([[1, 2, 0], [1, 0, 2]])),
                    output=Grid(np.array([[2, 1, 0], [2, 0, 1]]))
                )
            ],
            test_pairs=[
                GridPair(
                    input=Grid(np.array([[0, 1, 2]])),
                    output=Grid(np.array([[0, 2, 1]]))
                )
            ]
        )
        agent = HeuristicAgent()
        preds = agent.predict(task)
        assert preds[0] == task.test_pairs[0].output
