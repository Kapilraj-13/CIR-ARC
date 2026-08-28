from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set
from scipy import ndimage

from cir_arc.core.grid import Grid


@dataclass
class ArcObject:
    """One connected region in a grid."""
    color: int
    pixels: np.ndarray          # shape (N, 2), each row is (row, col)
    connectivity: int           # 4 or 8

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Returns (min_row, min_col, max_row, max_col) inclusive."""
        rows = self.pixels[:, 0]
        cols = self.pixels[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())

    @property
    def centroid(self) -> Tuple[float, float]:
        return float(self.pixels[:, 0].mean()), float(self.pixels[:, 1].mean())

    @property
    def height(self) -> int:
        r0, _, r1, _ = self.bounding_box
        return r1 - r0 + 1

    @property
    def width(self) -> int:
        _, c0, _, c1 = self.bounding_box
        return c1 - c0 + 1

    @property
    def is_square(self) -> bool:
        return self.height == self.width

    @property
    def is_rectangle(self) -> bool:
        return self.size == self.height * self.width

    @property
    def has_horizontal_symmetry(self) -> bool:
        return self._check_symmetry("horizontal")

    @property
    def has_vertical_symmetry(self) -> bool:
        return self._check_symmetry("vertical")

    def _check_symmetry(self, axis: str) -> bool:
        mask = self.to_mask()
        if axis == "horizontal":
            return np.array_equal(mask, np.flipud(mask))
        else:
            return np.array_equal(mask, np.fliplr(mask))

    def to_mask(self) -> np.ndarray:
        """Return boolean mask of object's bounding box."""
        r0, c0, r1, c1 = self.bounding_box
        mask = np.zeros((self.height, self.width), dtype=bool)
        for r, c in self.pixels:
            mask[r - r0, c - c0] = True
        return mask

    def to_subgrid(self, grid: Grid) -> Grid:
        """Extract the object's bounding box from the grid."""
        r0, c0, r1, c1 = self.bounding_box
        return grid.crop(r0, c0, r1 + 1, c1 + 1)

    def is_adjacent_to(self, other: ArcObject) -> bool:
        """True if objects share a border (4-connected, ignoring colors)."""
        self_set: Set[Tuple[int, int]] = set(map(tuple, self.pixels.tolist()))
        for r, c in other.pixels:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in self_set:
                    return True
        return False

    def __repr__(self) -> str:
        r0, c0, r1, c1 = self.bounding_box
        return (f"ArcObject(color={self.color}, size={self.size}, "
                f"bbox=({r0},{c0})-({r1},{c1}))")


def extract_objects(
    grid: Grid,
    background_color: int = 0,
    connectivity: int = 4
) -> List[ArcObject]:
    """
    Extract all connected objects from a grid.

    Parameters
    ----------
    grid            : Grid to extract from
    background_color: Color treated as background (not extracted)
    connectivity    : 4 (orthogonal) or 8 (including diagonal)

    Returns
    -------
    List of ArcObject, sorted by size descending.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    struct = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        if connectivity == 4
        else np.ones((3, 3), dtype=bool)
    )

    data = grid.data
    objects: List[ArcObject] = []

    for color in range(10):
        if color == background_color:
            continue

        mask = (data == color)
        if not mask.any():
            continue

        labeled, n_components = ndimage.label(mask, structure=struct)

        for label_id in range(1, n_components + 1):
            pixel_positions = np.argwhere(labeled == label_id)
            objects.append(ArcObject(
                color=color,
                pixels=pixel_positions,
                connectivity=connectivity
            ))

    objects.sort(key=lambda o: o.size, reverse=True)
    return objects


def build_adjacency_matrix(objects: List[ArcObject]) -> np.ndarray:
    """
    Return NxN boolean matrix where entry [i,j] is True
    if objects[i] and objects[j] are adjacent.
    """
    n = len(objects)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if objects[i].is_adjacent_to(objects[j]):
                adj[i, j] = adj[j, i] = True
    return adj


def get_background_color(grid: Grid) -> int:
    """
    Infer background color as the most frequent color.
    Falls back to 0 if tied.
    """
    data = grid.data
    counts = np.bincount(data.flatten(), minlength=10)
    return int(counts.argmax())
