import pytest
import numpy as np

from cir_arc.core.grid import Grid
from cir_arc.core.objects import (
    ArcObject, extract_objects, build_adjacency_matrix, get_background_color
)


class TestExtractObjects:
    def test_single_object(self):
        data = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data))
        assert len(objects) == 1
        assert objects[0].color == 1
        assert objects[0].size == 1

    def test_two_separate_objects(self):
        data = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data))
        assert len(objects) == 2

    def test_connected_component_4(self):
        data = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data), connectivity=4)
        assert len(objects) == 1
        assert objects[0].size == 3

    def test_diagonal_split_4_connectivity(self):
        """Diagonally adjacent cells are separate objects in 4-connectivity."""
        data = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data), connectivity=4)
        assert len(objects) == 2

    def test_diagonal_merged_8_connectivity(self):
        """Diagonally adjacent cells form one object in 8-connectivity."""
        data = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data), connectivity=8)
        assert len(objects) == 1

    def test_multiple_colors(self):
        data = np.array([
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 0, 0],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data))
        assert len(objects) == 3
        colors = sorted([o.color for o in objects])
        assert colors == [1, 2, 3]

    def test_background_excluded(self):
        data = np.zeros((5, 5), dtype=np.int8)
        objects = extract_objects(Grid(data))
        assert len(objects) == 0

    def test_custom_background_color(self):
        data = np.full((3, 3), 5, dtype=np.int8)
        data[1, 1] = 0  # 0 is now an "object" since background is 5
        objects = extract_objects(Grid(data), background_color=5)
        assert len(objects) == 1
        assert objects[0].color == 0

    def test_invalid_connectivity(self):
        with pytest.raises(ValueError):
            extract_objects(Grid(np.zeros((3, 3), dtype=np.int8)), connectivity=6)

    def test_sorted_by_size_descending(self):
        data = np.array([
            [1, 0, 2, 2, 2],
            [0, 0, 0, 0, 0],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data))
        assert objects[0].size >= objects[1].size


class TestArcObjectProperties:
    @pytest.fixture
    def rect_object(self):
        pixels = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]])
        return ArcObject(color=1, pixels=pixels, connectivity=4)

    def test_bounding_box(self, rect_object):
        assert rect_object.bounding_box == (1, 1, 2, 3)

    def test_height_width(self, rect_object):
        assert rect_object.height == 2
        assert rect_object.width == 3

    def test_size(self, rect_object):
        assert rect_object.size == 6

    def test_is_rectangle(self, rect_object):
        assert rect_object.is_rectangle is True

    def test_is_square_false(self, rect_object):
        assert rect_object.is_square is False

    def test_is_square_true(self):
        pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        obj = ArcObject(color=1, pixels=pixels, connectivity=4)
        assert obj.is_square is True

    def test_centroid(self):
        pixels = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
        obj = ArcObject(color=1, pixels=pixels, connectivity=8)
        assert obj.centroid == (1.0, 1.0)

    def test_to_mask(self):
        pixels = np.array([[0, 0], [0, 1], [1, 0]])
        obj = ArcObject(color=1, pixels=pixels, connectivity=4)
        mask = obj.to_mask()
        expected = np.array([[True, True], [True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_symmetry_horizontal(self):
        pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        obj = ArcObject(color=1, pixels=pixels, connectivity=4)
        assert obj.has_horizontal_symmetry is True

    def test_symmetry_vertical(self):
        pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        obj = ArcObject(color=1, pixels=pixels, connectivity=4)
        assert obj.has_vertical_symmetry is True


class TestAdjacency:
    def test_adjacent_objects(self):
        obj_a = ArcObject(color=1, pixels=np.array([[0, 0], [0, 1]]), connectivity=4)
        obj_b = ArcObject(color=2, pixels=np.array([[1, 0], [1, 1]]), connectivity=4)
        assert obj_a.is_adjacent_to(obj_b) is True

    def test_non_adjacent_objects(self):
        obj_a = ArcObject(color=1, pixels=np.array([[0, 0]]), connectivity=4)
        obj_b = ArcObject(color=2, pixels=np.array([[2, 2]]), connectivity=4)
        assert obj_a.is_adjacent_to(obj_b) is False

    def test_adjacency_matrix(self):
        data = np.array([
            [1, 1, 0, 2],
            [0, 0, 0, 2],
        ], dtype=np.int8)
        objects = extract_objects(Grid(data))
        adj = build_adjacency_matrix(objects)
        # Two objects, not adjacent
        assert adj.shape == (2, 2)


class TestBackgroundColor:
    def test_default_background(self):
        data = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        assert get_background_color(Grid(data)) == 0

    def test_non_zero_background(self):
        data = np.array([
            [5, 5, 5],
            [5, 1, 5],
            [5, 5, 5],
        ], dtype=np.int8)
        assert get_background_color(Grid(data)) == 5
