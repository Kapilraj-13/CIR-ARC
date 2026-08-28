import pytest
import numpy as np

from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask
from cir_arc.generators.single_rule import GENERATOR_REGISTRY, SingleRuleGenerator
from cir_arc.generators.composition import TwoRuleGenerator
from cir_arc.dsl.primitives import apply_rule


class TestSingleRuleGenerators:
    @pytest.mark.parametrize("rule_name", list(GENERATOR_REGISTRY.keys()))
    def test_generator_produces_valid_task(self, rule_name):
        gen = GENERATOR_REGISTRY[rule_name]()
        tasks = gen.generate_batch(5, seed=42)
        assert len(tasks) == 5
        for task in tasks:
            assert isinstance(task, ArcTask)
            assert task.source == "synthetic"
            assert task.rule_type == rule_name
            assert task.n_train == 3
            assert task.n_test == 1
            for pair in task.train_pairs + task.test_pairs:
                assert pair.input is not None
                assert pair.output is not None

    @pytest.mark.parametrize("rule_name", list(GENERATOR_REGISTRY.keys()))
    def test_generated_tasks_are_consistent(self, rule_name):
        """Verify that the output actually matches applying the rule to input."""
        gen = GENERATOR_REGISTRY[rule_name]()
        tasks = gen.generate_batch(3, seed=123)
        for task in tasks:
            for pair in task.train_pairs + task.test_pairs:
                expected = apply_rule(rule_name, pair.input, task.rule_params)
                assert pair.output == expected, (
                    f"Generator {rule_name}: output mismatch"
                )

    @pytest.mark.parametrize("rule_name", list(GENERATOR_REGISTRY.keys()))
    def test_reproducible_with_same_seed(self, rule_name):
        gen = GENERATOR_REGISTRY[rule_name]()
        batch1 = gen.generate_batch(3, seed=42)
        batch2 = gen.generate_batch(3, seed=42)
        for t1, t2 in zip(batch1, batch2):
            assert t1.content_hash == t2.content_hash

    @pytest.mark.parametrize("rule_name", list(GENERATOR_REGISTRY.keys()))
    def test_different_seeds_give_different_tasks(self, rule_name):
        gen = GENERATOR_REGISTRY[rule_name]()
        batch1 = gen.generate_batch(3, seed=42)
        batch2 = gen.generate_batch(3, seed=999)
        hashes1 = {t.content_hash for t in batch1}
        hashes2 = {t.content_hash for t in batch2}
        # Very unlikely to have all the same
        assert hashes1 != hashes2


class TestTwoRuleGenerator:
    def test_composition_produces_valid_tasks(self):
        gen = TwoRuleGenerator("reflect_horizontal", "color_swap_all")
        tasks = gen.generate_batch(5, seed=42)
        assert len(tasks) == 5
        for task in tasks:
            assert "compose_" in task.rule_type
            assert task.difficulty == 2

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError):
            TwoRuleGenerator("nonexistent", "reflect_horizontal")

    def test_composition_is_reproducible(self):
        gen = TwoRuleGenerator("rotate_90", "color_swap_all")
        b1 = gen.generate_batch(3, seed=42)
        b2 = gen.generate_batch(3, seed=42)
        for t1, t2 in zip(b1, b2):
            assert t1.content_hash == t2.content_hash
