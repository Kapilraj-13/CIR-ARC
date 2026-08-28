import pytest
import numpy as np
import json
import tempfile

from cir_arc.core.grid import Grid, GridValidationError, MAX_H, MAX_W


class TestGridCreation:
    def test_create_valid_grid(self):
        data = np.array([[0, 1], [2, 3]], dtype=np.int8)
        g = Grid(data)
        assert g.height == 2
        assert g.width == 2
        assert g.shape == (2, 2)

    def test_create_from_python_list(self):
        g = Grid(np.array([[0, 1, 2], [3, 4, 5]]))
        assert g.shape == (2, 3)

    def test_auto_cast_to_int8(self):
        data = np.array([[0, 1], [2, 3]], dtype=np.int64)
        g = Grid(data)
        assert g.data.dtype == np.int8

    def test_reject_1d_array(self):
        with pytest.raises(GridValidationError):
            Grid(np.array([1, 2, 3]))

    def test_reject_3d_array(self):
        with pytest.raises(GridValidationError):
            Grid(np.zeros((2, 2, 2)))

    def test_reject_empty_grid(self):
        with pytest.raises(GridValidationError):
            Grid(np.array([]).reshape(0, 0))

    def test_reject_oversized_grid(self):
        with pytest.raises(GridValidationError):
            Grid(np.zeros((MAX_H + 1, 5), dtype=np.int8))

    def test_reject_negative_values(self):
        with pytest.raises(GridValidationError):
            Grid(np.array([[-1, 0], [0, 0]]))

    def test_reject_values_above_9(self):
        with pytest.raises(GridValidationError):
            Grid(np.array([[0, 10], [0, 0]]))

    def test_min_size_1x1(self):
        g = Grid(np.array([[5]]))
        assert g.shape == (1, 1)

    def test_max_size_30x30(self):
        g = Grid(np.zeros((30, 30), dtype=np.int8))
        assert g.shape == (30, 30)


class TestGridProperties:
    def test_colors_used(self):
        g = Grid(np.array([[0, 1], [2, 0]]))
        assert g.colors_used == [0, 1, 2]

    def test_n_colors(self):
        g = Grid(np.array([[0, 1], [2, 0]]))
        assert g.n_colors == 3

    def test_single_color_grid(self):
        g = Grid(np.zeros((3, 3), dtype=np.int8))
        assert g.colors_used == [0]
        assert g.n_colors == 1

    def test_data_returns_copy(self):
        g = Grid(np.array([[0, 1], [2, 3]]))
        d = g.data
        d[0, 0] = 9
        assert g.data[0, 0] == 0  # original unchanged


class TestGridEquality:
    def test_equal_grids(self):
        a = Grid(np.array([[0, 1], [2, 3]]))
        b = Grid(np.array([[0, 1], [2, 3]]))
        assert a == b

    def test_unequal_grids(self):
        a = Grid(np.array([[0, 1], [2, 3]]))
        b = Grid(np.array([[0, 1], [2, 4]]))
        assert a != b

    def test_different_shapes(self):
        a = Grid(np.array([[0, 1]]))
        b = Grid(np.array([[0], [1]]))
        assert a != b

    def test_not_equal_to_non_grid(self):
        g = Grid(np.array([[0, 1], [2, 3]]))
        assert g != "not a grid"

    def test_hash_equal_for_same_content(self):
        a = Grid(np.array([[0, 1], [2, 3]]))
        b = Grid(np.array([[0, 1], [2, 3]]))
        assert hash(a) == hash(b)


class TestGridSerialization:
    def test_to_list(self):
        g = Grid(np.array([[0, 1], [2, 3]]))
        assert g.to_list() == [[0, 1], [2, 3]]

    def test_from_list(self):
        g = Grid.from_list([[0, 1], [2, 3]])
        assert g.shape == (2, 2)
        assert np.array_equal(g.data, np.array([[0, 1], [2, 3]]))

    def test_roundtrip_list(self):
        original = Grid(np.array([[0, 1, 2], [3, 4, 5]]))
        restored = Grid.from_list(original.to_list())
        assert original == restored

    def test_roundtrip_json_string(self):
        original = Grid(np.array([[0, 1], [8, 9]]))
        s = original.to_json_string()
        restored = Grid.from_json_string(s)
        assert original == restored


class TestGridOperations:
    def test_copy(self):
        g = Grid(np.array([[0, 1], [2, 3]]))
        c = g.copy()
        assert g == c
        assert g is not c

    def test_crop(self):
        g = Grid(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        cropped = g.crop(0, 0, 2, 2)
        expected = Grid(np.array([[0, 1], [3, 4]]))
        assert cropped == expected

    def test_pad(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        padded = g.pad(1, 1, 1, 1, fill=0)
        assert padded.shape == (4, 4)
        assert padded.data[0, 0] == 0
        assert padded.data[1, 1] == 1

    def test_resize_canvas(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        resized = g.resize_canvas(4, 4, fill=0)
        assert resized.shape == (4, 4)
        assert resized.data[0, 0] == 1
        assert resized.data[3, 3] == 0

    def test_recolor(self):
        g = Grid(np.array([[0, 1], [1, 2]]))
        recolored = g.recolor({1: 5, 2: 7})
        expected = Grid(np.array([[0, 5], [5, 7]]))
        assert recolored == expected

    def test_recolor_swap(self):
        """Ensure swapping two colors works correctly (no overwrite bug)."""
        g = Grid(np.array([[1, 2], [2, 1]]))
        swapped = g.recolor({1: 2, 2: 1})
        expected = Grid(np.array([[2, 1], [1, 2]]))
        assert swapped == expected


class TestGridTransformations:
    @pytest.fixture
    def sample_grid(self):
        return Grid(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_rotate_90(self, sample_grid):
        r = sample_grid.rotate_90()
        expected = Grid(np.array([[3, 6], [2, 5], [1, 4]]))
        assert r == expected

    def test_rotate_180(self, sample_grid):
        r = sample_grid.rotate_180()
        expected = Grid(np.array([[6, 5, 4], [3, 2, 1]]))
        assert r == expected

    def test_rotate_270(self, sample_grid):
        r = sample_grid.rotate_270()
        expected = Grid(np.array([[4, 1], [5, 2], [6, 3]]))
        assert r == expected

    def test_rotate_360_is_identity(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        assert g.rotate_90().rotate_90().rotate_90().rotate_90() == g

    def test_reflect_horizontal(self, sample_grid):
        r = sample_grid.reflect_horizontal()
        expected = Grid(np.array([[4, 5, 6], [1, 2, 3]]))
        assert r == expected

    def test_reflect_vertical(self, sample_grid):
        r = sample_grid.reflect_vertical()
        expected = Grid(np.array([[3, 2, 1], [6, 5, 4]]))
        assert r == expected

    def test_reflect_diagonal(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        r = g.reflect_diagonal()
        expected = Grid(np.array([[1, 3], [2, 4]]))
        assert r == expected

    def test_reflect_antidiagonal(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        r = g.reflect_antidiagonal()
        expected = Grid(np.array([[4, 2], [3, 1]]))
        assert r == expected

    def test_double_reflect_is_identity(self):
        g = Grid(np.array([[1, 2], [3, 4]]))
        assert g.reflect_horizontal().reflect_horizontal() == g
        assert g.reflect_vertical().reflect_vertical() == g


class TestGridAscii:
    def test_to_ascii(self):
        g = Grid(np.array([[0, 1], [2, 3]]))
        expected = "0 1\n2 3"
        assert g.to_ascii() == expected

    def test_repr(self):
        g = Grid(np.array([[0, 1], [2, 3]]))
        r = repr(g)
        assert "Grid" in r
        assert "(2, 2)" in r
