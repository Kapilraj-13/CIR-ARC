from __future__ import annotations
import numpy as np
import json
from typing import List, Tuple, Optional

# ARC color palette (for display only)
ARC_COLORS = {
    0: (0, 0, 0),        # black (background)
    1: (0, 0, 255),      # blue
    2: (255, 0, 0),      # red
    3: (0, 128, 0),      # green
    4: (255, 255, 0),    # yellow
    5: (128, 128, 128),  # grey
    6: (255, 0, 255),    # magenta/fuchsia
    7: (255, 165, 0),    # orange
    8: (0, 255, 255),    # azure/cyan
    9: (128, 0, 0),      # maroon
}

MAX_H = 30
MAX_W = 30
N_COLORS = 10


class GridValidationError(Exception):
    pass


class Grid:
    """
    Canonical in-memory representation of an ARC grid.
    Wraps a numpy int8 array of shape (H, W), values 0-9.
    """

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.int8)
        self._validate(data)
        self._data = data

    @staticmethod
    def _validate(data: np.ndarray) -> None:
        if data.ndim != 2:
            raise GridValidationError(
                f"Grid must be 2D, got shape {data.shape}"
            )
        h, w = data.shape
        if h < 1 or h > MAX_H or w < 1 or w > MAX_W:
            raise GridValidationError(
                f"Grid size {h}x{w} out of range (1-{MAX_H} x 1-{MAX_W})"
            )
        if data.min() < 0 or data.max() > 9:
            raise GridValidationError(
                f"Grid values must be 0-9, got min={data.min()} max={data.max()}"
            )

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()  # always return a copy — no accidental mutation

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def colors_used(self) -> List[int]:
        return sorted(np.unique(self._data).tolist())

    @property
    def n_colors(self) -> int:
        return len(self.colors_used)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self._data, other._data)

    def __hash__(self) -> int:
        return hash(self._data.tobytes())

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape}, colors={self.colors_used})"

    # ── Serialization ──────────────────────────────────────────────────────

    @classmethod
    def from_list(cls, nested_list: List[List[int]]) -> Grid:
        """Load from ARC JSON format (nested Python lists)."""
        return cls(np.array(nested_list, dtype=np.int8))

    def to_list(self) -> List[List[int]]:
        """Export to ARC JSON format."""
        return self._data.tolist()

    @classmethod
    def from_json_string(cls, s: str) -> Grid:
        return cls.from_list(json.loads(s))

    def to_json_string(self) -> str:
        return json.dumps(self.to_list())

    # ── Grid operations ────────────────────────────────────────────────────

    def copy(self) -> Grid:
        return Grid(self._data.copy())

    def crop(self, r0: int, c0: int, r1: int, c1: int) -> Grid:
        """Crop to rows [r0:r1], cols [c0:c1]."""
        return Grid(self._data[r0:r1, c0:c1].copy())

    def pad(self, top: int, bottom: int, left: int, right: int,
            fill: int = 0) -> Grid:
        """Pad grid with fill color."""
        return Grid(
            np.pad(self._data, ((top, bottom), (left, right)),
                   constant_values=fill).astype(np.int8)
        )

    def resize_canvas(self, new_h: int, new_w: int, fill: int = 0) -> Grid:
        """Place this grid top-left on a new canvas of given size."""
        canvas = np.full((new_h, new_w), fill, dtype=np.int8)
        h = min(self.height, new_h)
        w = min(self.width, new_w)
        canvas[:h, :w] = self._data[:h, :w]
        return Grid(canvas)

    def recolor(self, color_map: dict) -> Grid:
        """Apply a color remapping. Keys and values must be 0-9."""
        result = self._data.copy()
        temp = self._data.copy()
        for src, dst in color_map.items():
            result[temp == src] = dst
        return Grid(result)

    def rotate_90(self) -> Grid:
        return Grid(np.rot90(self._data, k=1).copy())

    def rotate_180(self) -> Grid:
        return Grid(np.rot90(self._data, k=2).copy())

    def rotate_270(self) -> Grid:
        return Grid(np.rot90(self._data, k=3).copy())

    def reflect_horizontal(self) -> Grid:
        return Grid(np.flipud(self._data).copy())

    def reflect_vertical(self) -> Grid:
        return Grid(np.fliplr(self._data).copy())

    def reflect_diagonal(self) -> Grid:
        return Grid(self._data.T.copy())

    def reflect_antidiagonal(self) -> Grid:
        return Grid(np.fliplr(np.flipud(self._data.T)).copy())

    # ── Display ────────────────────────────────────────────────────────────

    def to_ascii(self) -> str:
        """Simple ASCII display for debugging."""
        return "\n".join(
            " ".join(str(v) for v in row)
            for row in self._data.tolist()
        )

    def show(self, title: str = "", ax=None) -> None:
        """Matplotlib display (requires matplotlib)."""
        import matplotlib.pyplot as plt

        palette = np.array(
            [ARC_COLORS[i] for i in range(N_COLORS)], dtype=np.uint8
        )
        img = palette[self._data]

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(self.width * 0.5 + 1, self.height * 0.5 + 1)
            )
            show_after = True
        else:
            show_after = False

        ax.imshow(img)
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        if title:
            ax.set_title(title)

        if show_after:
            plt.tight_layout()
            plt.show()

    def show_on_ax(self, ax, title: str = "") -> None:
        """Render grid on a provided matplotlib axis."""
        self.show(title=title, ax=ax)


def show_task(task, title: str = "") -> None:
    """Display all train pairs and first test pair side by side."""
    import matplotlib.pyplot as plt
    n = len(task.train_pairs)
    fig, axes = plt.subplots(2, n + 1, figsize=((n + 1) * 3, 6))
    if n + 1 == 1:
        axes = axes.reshape(2, 1)
    for i, pair in enumerate(task.train_pairs):
        pair.input.show_on_ax(axes[0][i], title=f"Train {i+1} Input")
        if pair.output is not None:
            pair.output.show_on_ax(axes[1][i], title=f"Train {i+1} Output")
        else:
            axes[1][i].axis("off")
    task.test_pairs[0].input.show_on_ax(axes[0][n], title="Test Input")
    if task.test_pairs[0].output is not None:
        task.test_pairs[0].output.show_on_ax(axes[1][n], title="Test Output")
    else:
        axes[1][n].axis("off")
    plt.suptitle(title or task.task_id)
    plt.tight_layout()
    plt.show()
