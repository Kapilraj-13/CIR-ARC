import pytest
import numpy as np

from cir_arc.core.grid import Grid
from cir_arc.dsl.primitives import apply_rule, RULE_REGISTRY
from cir_arc.dsl.compose import compose_rules


class TestPrimitiveRules:
    @pytest.fixture
    def grid_2x3(self):
        return Grid(np.array([[1, 2, 3], [4, 5, 6]]))

    @pytest.fixture
    def grid_3x3(self):
        return Grid(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # Tier 1 rules
    def test_reflect_horizontal(self, grid_2x3):
        result = apply_rule("reflect_horizontal", grid_2x3)
        expected = Grid(np.array([[4, 5, 6], [1, 2, 3]]))
        assert result == expected

    def test_reflect_vertical(self, grid_2x3):
        result = apply_rule("reflect_vertical", grid_2x3)
        expected = Grid(np.array([[3, 2, 1], [6, 5, 4]]))
        assert result == expected

    def test_reflect_diagonal(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        result = apply_rule("reflect_diagonal", g)
        expected = Grid(np.array([[1, 3], [2, 4]]))
        assert result == expected

    def test_reflect_antidiagonal(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        result = apply_rule("reflect_antidiagonal", g)
        expected = Grid(np.array([[4, 2], [3, 1]]))
        assert result == expected

    def test_rotate_90(self, grid_2x3):
        result = apply_rule("rotate_90", grid_2x3)
        expected = Grid(np.array([[3, 6], [2, 5], [1, 4]]))
        assert result == expected

    def test_rotate_180(self, grid_2x3):
        result = apply_rule("rotate_180", grid_2x3)
        expected = Grid(np.array([[6, 5, 4], [3, 2, 1]]))
        assert result == expected

    def test_rotate_270(self, grid_2x3):
        result = apply_rule("rotate_270", grid_2x3)
        expected = Grid(np.array([[4, 1], [5, 2], [6, 3]]))
        assert result == expected

    def test_color_swap_all(self):
        g = Grid(np.array([[1, 2], [2, 1]]))
        result = apply_rule("color_swap_all", g, {"color_a": 1, "color_b": 2})
        expected = Grid(np.array([[2, 1], [1, 2]]))
        assert result == expected

    def test_move_objects(self):
        g = Grid(np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]))
        result = apply_rule("move_objects", g, {"dr": -1, "dc": 0, "background": 0})
        expected = Grid(np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]))
        assert result == expected

    # Tier 2 rules
    def test_gravity_down(self):
        g = Grid(np.array([
            [1, 0, 2],
            [0, 0, 0],
            [0, 0, 0],
        ]))
        result = apply_rule("gravity", g, {"direction": "down", "background": 0})
        expected = Grid(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 2],
        ]))
        assert result == expected

    def test_gravity_up(self):
        g = Grid(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 2],
        ]))
        result = apply_rule("gravity", g, {"direction": "up", "background": 0})
        expected = Grid(np.array([
            [1, 0, 2],
            [0, 0, 0],
            [0, 0, 0],
        ]))
        assert result == expected

    def test_gravity_left(self):
        g = Grid(np.array([
            [0, 0, 1],
            [0, 2, 0],
        ]))
        result = apply_rule("gravity", g, {"direction": "left", "background": 0})
        expected = Grid(np.array([
            [1, 0, 0],
            [2, 0, 0],
        ]))
        assert result == expected

    def test_gravity_right(self):
        g = Grid(np.array([
            [1, 0, 0],
            [0, 2, 0],
        ]))
        result = apply_rule("gravity", g, {"direction": "right", "background": 0})
        expected = Grid(np.array([
            [0, 0, 1],
            [0, 0, 2],
        ]))
        assert result == expected

    def test_scale_up_2x(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        result = apply_rule("scale_up", g, {"factor": 2})
        assert result.shape == (4, 4)
        assert result.data[0, 0] == 1
        assert result.data[0, 1] == 1
        assert result.data[1, 0] == 1
        assert result.data[1, 1] == 1

    def test_draw_border(self):
        g = Grid(np.zeros((4, 4), dtype=np.int8))
        result = apply_rule("draw_border", g, {"color": 3})
        assert result.data[0, 0] == 3
        assert result.data[0, 3] == 3
        assert result.data[3, 0] == 3
        assert result.data[1, 1] == 0  # interior unchanged

    def test_tile_pattern(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        result = apply_rule("tile_pattern", g, {"n_rows": 2, "n_cols": 2})
        assert result.shape == (4, 4)
        assert result.data[2, 0] == 1  # second row tile
        assert result.data[0, 2] == 1  # second col tile

    def test_fill_enclosed(self):
        g = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]))
        result = apply_rule("fill_enclosed", g, {"fill_color": 2, "background": 0})
        assert result.data[2, 2] == 2  # interior filled
        assert result.data[0, 0] == 0  # exterior unchanged


class TestRuleRegistry:
    def test_all_rules_registered(self):
        expected_rules = [
            "reflect_horizontal", "reflect_vertical",
            "reflect_diagonal", "reflect_antidiagonal",
            "rotate_90", "rotate_180", "rotate_270",
            "color_swap_all", "color_remap", "move_objects",
            "gravity", "scale_up", "draw_border",
            "tile_pattern", "fill_enclosed",
        ]
        for rule in expected_rules:
            assert rule in RULE_REGISTRY, f"{rule} not in registry"

    def test_unknown_rule_raises(self):
        g = Grid(np.array([[1, 2]]))
        with pytest.raises(ValueError):
            apply_rule("nonexistent_rule", g)


class TestCompose:
    def test_two_rule_composition(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        result = compose_rules(g, [
            ("reflect_horizontal", {}),
            ("reflect_vertical", {}),
        ])
        # reflect_horizontal then reflect_vertical = rotate 180
        expected = g.rotate_180()
        assert result == expected

    def test_empty_composition(self):
        g = Grid(np.array([[1, 2]]))
        result = compose_rules(g, [])
        assert result == g

    def test_single_rule_composition(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        result = compose_rules(g, [("rotate_90", {})])
        expected = g.rotate_90()
        assert result == expected
