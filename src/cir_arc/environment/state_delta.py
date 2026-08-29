from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData, MultiLayerGrid


@dataclass(frozen=True)
class GridObject:
    """A connected component extracted from a multi-layer grid."""
    object_id: str
    color: int
    layer: int
    pixel_count: int
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    centroid: Tuple[float, float]     # (row_center, col_center)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


def find_objects(
    source: Union[FrameData, MultiLayerGrid, List[np.ndarray], np.ndarray],
    background_color: int = 0,
    connectivity: int = 4,
) -> List[GridObject]:
    """Extract connected components across all layers."""
    if isinstance(source, FrameData):
        layers = source.grid.layers
    elif isinstance(source, MultiLayerGrid):
        layers = source.layers
    elif isinstance(source, list):
        layers = [np.asarray(l, dtype=np.int16) for l in source]
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            layers = [source]
        else:
            layers = [source[i] for i in range(source.shape[0])]
    else:
        return []

    objects: List[GridObject] = []

    for layer_idx, layer in enumerate(layers):
        grid = np.asarray(layer, dtype=np.int16)
        if grid.ndim != 2:
            continue

        height, width = grid.shape
        visited = np.zeros((height, width), dtype=bool)

        for r in range(height):
            for c in range(width):
                color = int(grid[r, c])
                if color == background_color or visited[r, c]:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                pixels: List[Tuple[int, int]] = []

                while stack:
                    cr, cc = stack.pop()
                    pixels.append((cr, cc))

                    neighbors = [(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)]
                    if connectivity == 8:
                        neighbors.extend([(cr + 1, cc + 1), (cr + 1, cc - 1), (cr - 1, cc + 1), (cr - 1, cc - 1)])

                    for nr, nc in neighbors:
                        if 0 <= nr < height and 0 <= nc < width:
                            if not visited[nr, nc] and int(grid[nr, nc]) == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                if not pixels:
                    continue

                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                centroid = (float(sum(rows)) / len(rows), float(sum(cols)) / len(cols))
                object_id = f"L{layer_idx}C{color}@{bbox[0]},{bbox[1]}"

                objects.append(
                    GridObject(
                        object_id=object_id,
                        color=color,
                        layer=layer_idx,
                        pixel_count=len(pixels),
                        bbox=bbox,
                        centroid=centroid,
                    )
                )

    objects.sort(key=lambda o: (-o.pixel_count, o.layer, o.color, o.bbox))
    return objects


@dataclass
class PropertyMutation:
    object_id: str
    property_name: str
    old_value: Any
    new_value: Any


@dataclass
class StateDelta:
    frame_hash_before: str
    frame_hash_after: str
    action_taken: Action
    has_mutation: bool
    pixel_diff_count: int
    objects_before: List[GridObject] = field(default_factory=list)
    objects_after: List[GridObject] = field(default_factory=list)
    spawned_objects: List[GridObject] = field(default_factory=list)
    destroyed_objects: List[GridObject] = field(default_factory=list)
    moved_objects: List[Tuple[GridObject, GridObject, Tuple[float, float]]] = field(default_factory=list)
    mutations: List[PropertyMutation] = field(default_factory=list)

    @property
    def is_identity(self) -> bool:
        return not self.has_mutation and self.pixel_diff_count == 0


def compute_state_delta(
    frame_before: FrameData,
    frame_after: FrameData,
    action: Action,
    background_color: int = 0,
) -> StateDelta:
    """Compute state delta between two consecutive frames."""
    hash_b = frame_before.hash()
    hash_a = frame_after.hash()

    if hash_b == hash_a:
        objs = find_objects(frame_before, background_color=background_color)
        return StateDelta(
            frame_hash_before=hash_b,
            frame_hash_after=hash_a,
            action_taken=action,
            has_mutation=False,
            pixel_diff_count=0,
            objects_before=objs,
            objects_after=objs,
        )

    # Compute pixel differences across layers
    diff_pixels = 0
    max_layers = max(frame_before.grid.num_layers, frame_after.grid.num_layers)
    for l in range(max_layers):
        lb = frame_before.grid.layers[l] if l < frame_before.grid.num_layers else None
        la = frame_after.grid.layers[l] if l < frame_after.grid.num_layers else None
        if lb is None or la is None:
            active_layer = la if lb is None else lb
            if active_layer is not None:
                diff_pixels += int(np.count_nonzero(active_layer != background_color))
        else:
            # check shapes
            if lb.shape == la.shape:
                diff_pixels += int(np.count_nonzero(lb != la))
            else:
                diff_pixels += max(lb.size, la.size)

    objs_before = find_objects(frame_before, background_color=background_color)
    objs_after = find_objects(frame_after, background_color=background_color)

    # Match objects by color, layer, and pixel count
    unmatched_after = list(objs_after)
    matched_pairs: List[Tuple[GridObject, GridObject]] = []

    for ob in objs_before:
        best_match: Optional[GridObject] = None
        best_dist = float("inf")
        for oa in unmatched_after:
            if ob.color == oa.color and ob.layer == oa.layer and ob.pixel_count == oa.pixel_count:
                dist = (ob.centroid[0] - oa.centroid[0]) ** 2 + (ob.centroid[1] - oa.centroid[1]) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_match = oa

        if best_match is not None:
            matched_pairs.append((ob, best_match))
            unmatched_after.remove(best_match)

    matched_before_ids = {pair[0].object_id for pair in matched_pairs}
    destroyed = [ob for ob in objs_before if ob.object_id not in matched_before_ids]
    spawned = list(unmatched_after)

    moved: List[Tuple[GridObject, GridObject, Tuple[float, float]]] = []
    mutations: List[PropertyMutation] = []

    for ob, oa in matched_pairs:
        d_row = oa.centroid[0] - ob.centroid[0]
        d_col = oa.centroid[1] - ob.centroid[1]
        if abs(d_row) > 1e-4 or abs(d_col) > 1e-4:
            moved.append((ob, oa, (d_row, d_col)))
            mutations.append(
                PropertyMutation(
                    object_id=ob.object_id,
                    property_name="centroid",
                    old_value=ob.centroid,
                    new_value=oa.centroid,
                )
            )

    has_mutation = diff_pixels > 0 or len(moved) > 0 or len(spawned) > 0 or len(destroyed) > 0

    return StateDelta(
        frame_hash_before=hash_b,
        frame_hash_after=hash_a,
        action_taken=action,
        has_mutation=has_mutation,
        pixel_diff_count=diff_pixels,
        objects_before=objs_before,
        objects_after=objs_after,
        spawned_objects=spawned,
        destroyed_objects=destroyed,
        moved_objects=moved,
        mutations=mutations,
    )
