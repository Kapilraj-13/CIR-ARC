from __future__ import annotations
import numpy as np
from typing import List
from cir_arc.core.grid import Grid
from cir_arc.core.task import ArcTask, GridPair
from cir_arc.dsl.primitives import apply_rule


class SingleRuleGenerator:
    """
    Base class for single-rule task generators.
    Subclass and implement: sample_params() and rule_name.
    """
    rule_name: str = ""

    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty

    def sample_input_grid(
        self,
        rng: np.random.Generator,
        min_h: int = 3,
        max_h: int = 10,
        min_w: int = 3,
        max_w: int = 10,
        n_colors: int = 3,
        background: int = 0,
        min_fill: float = 0.15,
        max_fill: float = 0.6
    ) -> Grid:
        """Generate a random input grid."""
        h = int(rng.integers(min_h, max_h + 1))
        w = int(rng.integers(min_w, max_w + 1))

        available = list(range(1, 10))
        rng.shuffle(available)
        colors = [background] + available[:n_colors - 1]

        probs = np.array([0.6] + [0.4 / (n_colors - 1)] * (n_colors - 1))
        data = rng.choice(colors, size=(h, w), p=probs).astype(np.int8)
        return Grid(data)

    def sample_params(self, rng: np.random.Generator) -> dict:
        """Override to sample rule-specific parameters."""
        return {}

    def generate_one(
        self,
        rng: np.random.Generator,
        task_id: str,
        n_train: int = 3
    ) -> ArcTask:
        """Generate one complete ARC task."""
        params = self.sample_params(rng)
        pairs = []

        for _ in range(n_train + 1):
            inp = self.sample_input_grid(rng)
            out = apply_rule(self.rule_name, inp, params)
            pairs.append(GridPair(input=inp, output=out))

        return ArcTask(
            task_id=task_id,
            source="synthetic",
            rule_type=self.rule_name,
            rule_params=params,
            difficulty=self.difficulty,
            train_pairs=pairs[:n_train],
            test_pairs=[pairs[n_train]],
        )

    def generate_batch(
        self,
        n: int,
        seed: int,
        id_prefix: str = ""
    ) -> List[ArcTask]:
        """Generate a batch of n tasks with reproducible seed."""
        rng = np.random.default_rng(seed)
        prefix = id_prefix or self.rule_name
        return [
            self.generate_one(rng, task_id=f"{prefix}_{i:06d}")
            for i in range(n)
        ]


# â”€â”€ Concrete generators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ReflectHorizontalGenerator(SingleRuleGenerator):
    rule_name = "reflect_horizontal"


class ReflectVerticalGenerator(SingleRuleGenerator):
    rule_name = "reflect_vertical"


class ReflectDiagonalGenerator(SingleRuleGenerator):
    rule_name = "reflect_diagonal"

    def sample_input_grid(self, rng, **kwargs):
        # Square grids for diagonal reflection to keep sizes manageable
        size = int(rng.integers(3, 10))
        return super().sample_input_grid(rng, min_h=size, max_h=size,
                                          min_w=size, max_w=size)


class ReflectAntidiagonalGenerator(SingleRuleGenerator):
    rule_name = "reflect_antidiagonal"

    def sample_input_grid(self, rng, **kwargs):
        size = int(rng.integers(3, 10))
        return super().sample_input_grid(rng, min_h=size, max_h=size,
                                          min_w=size, max_w=size)


class Rotate90Generator(SingleRuleGenerator):
    rule_name = "rotate_90"

    def sample_input_grid(self, rng, **kwargs):
        size = int(rng.integers(3, 10))
        return super().sample_input_grid(rng, min_h=size, max_h=size,
                                          min_w=size, max_w=size)


class Rotate180Generator(SingleRuleGenerator):
    rule_name = "rotate_180"


class Rotate270Generator(SingleRuleGenerator):
    rule_name = "rotate_270"

    def sample_input_grid(self, rng, **kwargs):
        size = int(rng.integers(3, 10))
        return super().sample_input_grid(rng, min_h=size, max_h=size,
                                          min_w=size, max_w=size)


class ColorSwapGenerator(SingleRuleGenerator):
    rule_name = "color_swap_all"

    def sample_params(self, rng):
        colors = list(range(1, 10))
        rng.shuffle(colors)
        return {"color_a": int(colors[0]), "color_b": int(colors[1])}

    def sample_input_grid(self, rng, **kwargs):
        return super().sample_input_grid(rng, n_colors=4)


class GravityGenerator(SingleRuleGenerator):
    rule_name = "gravity"

    def sample_params(self, rng):
        direction = str(rng.choice(["down", "up", "left", "right"]))
        return {"direction": direction, "background": 0}


class ScaleUpGenerator(SingleRuleGenerator):
    rule_name = "scale_up"

    def sample_params(self, rng):
        factor = int(rng.choice([2, 3]))
        return {"factor": factor}

    def sample_input_grid(self, rng, **kwargs):
        return super().sample_input_grid(rng, max_h=8, max_w=8)


class DrawBorderGenerator(SingleRuleGenerator):
    rule_name = "draw_border"

    def sample_params(self, rng):
        return {"color": int(rng.integers(1, 10))}

    def sample_input_grid(self, rng, **kwargs):
        return super().sample_input_grid(rng, min_h=4, min_w=4)


class TilePatternGenerator(SingleRuleGenerator):
    rule_name = "tile_pattern"

    def sample_params(self, rng):
        return {"n_rows": int(rng.integers(2, 4)), "n_cols": int(rng.integers(2, 4))}

    def sample_input_grid(self, rng, **kwargs):
        return super().sample_input_grid(rng, max_h=7, max_w=7)


class FillEnclosedGenerator(SingleRuleGenerator):
    rule_name = "fill_enclosed"

    def sample_params(self, rng):
        return {"fill_color": int(rng.integers(1, 10)), "background": 0}

    def sample_input_grid(self, rng, **kwargs):
        """Generate a grid with an enclosed region."""
        size = int(rng.integers(6, 12))
        data = np.zeros((size, size), dtype=np.int8)
        wall_color = int(rng.integers(1, 10))
        # Draw a rectangle wall
        margin = int(rng.integers(1, 3))
        r0, c0 = margin, margin
        r1, c1 = size - margin - 1, size - margin - 1
        if r1 > r0 + 1 and c1 > c0 + 1:
            data[r0, c0:c1+1] = wall_color
            data[r1, c0:c1+1] = wall_color
            data[r0:r1+1, c0] = wall_color
            data[r0:r1+1, c1] = wall_color
        return Grid(data)


# Registry of all generators
GENERATOR_REGISTRY = {
    "reflect_horizontal":   ReflectHorizontalGenerator,
    "reflect_vertical":     ReflectVerticalGenerator,
    "reflect_diagonal":     ReflectDiagonalGenerator,
    "reflect_antidiagonal": ReflectAntidiagonalGenerator,
    "rotate_90":            Rotate90Generator,
    "rotate_180":           Rotate180Generator,
    "rotate_270":           Rotate270Generator,
    "color_swap_all":       ColorSwapGenerator,
    "gravity":              GravityGenerator,
    "scale_up":             ScaleUpGenerator,
    "draw_border":          DrawBorderGenerator,
    "tile_pattern":         TilePatternGenerator,
    "fill_enclosed":        FillEnclosedGenerator,
}
