from __future__ import annotations
import numpy as np
from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask
from typing import List


class RandomColorAgent:
    """
    Predicts a grid of the same shape as test input,
    filled with uniform random colors (0-9).
    """
    name = "random_color"

    def __init__(self, seed: int = 300):
        self.rng = np.random.default_rng(seed)

    def predict(self, task: ArcTask) -> List[Grid]:
        predictions = []
        for test_pair in task.test_pairs:
            h, w = test_pair.input.shape
            data = self.rng.integers(0, 10, size=(h, w), dtype=np.int8)
            predictions.append(Grid(data))
        return predictions


class CopyInputAgent:
    """
    Predicts the test output is identical to the test input.
    """
    name = "copy_input"

    def predict(self, task: ArcTask) -> List[Grid]:
        return [test_pair.input.copy() for test_pair in task.test_pairs]
