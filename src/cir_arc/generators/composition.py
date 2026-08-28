from __future__ import annotations
import numpy as np
from typing import List, Tuple
from cir_arc.core.task import ArcTask, GridPair
from cir_arc.dsl.compose import compose_rules
from cir_arc.generators.single_rule import SingleRuleGenerator, GENERATOR_REGISTRY


class TwoRuleGenerator:
    """
    Generates tasks that require applying two rules in sequence.
    Input: original grid
    Output: grid after Rule_A then Rule_B
    Intermediate step is NEVER shown.
    """

    def __init__(self, rule_a: str, rule_b: str):
        if rule_a not in GENERATOR_REGISTRY:
            raise ValueError(f"Unknown rule: {rule_a}")
        if rule_b not in GENERATOR_REGISTRY:
            raise ValueError(f"Unknown rule: {rule_b}")

        self.gen_a = GENERATOR_REGISTRY[rule_a]()
        self.gen_b = GENERATOR_REGISTRY[rule_b]()
        self.rule_a = rule_a
        self.rule_b = rule_b

    def generate_one(
        self,
        rng: np.random.Generator,
        task_id: str,
        n_train: int = 3
    ) -> ArcTask:
        params_a = self.gen_a.sample_params(rng)
        params_b = self.gen_b.sample_params(rng)
        rule_sequence = [(self.rule_a, params_a), (self.rule_b, params_b)]

        pairs = []
        for _ in range(n_train + 1):
            inp = self.gen_a.sample_input_grid(rng)
            out = compose_rules(inp, rule_sequence)
            pairs.append(GridPair(input=inp, output=out))

        return ArcTask(
            task_id=task_id,
            source="synthetic",
            rule_type=f"compose_{self.rule_a}__{self.rule_b}",
            rule_params={"rule_a": params_a, "rule_b": params_b},
            difficulty=2,
            train_pairs=pairs[:n_train],
            test_pairs=[pairs[n_train]],
        )

    def generate_batch(self, n: int, seed: int) -> List[ArcTask]:
        rng = np.random.default_rng(seed)
        prefix = f"compose_{self.rule_a}__{self.rule_b}"
        return [
            self.generate_one(rng, task_id=f"{prefix}_{i:06d}")
            for i in range(n)
        ]
