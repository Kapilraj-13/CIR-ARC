"""Dataset, target map generation, and variable grid batch collate utilities for CIR-ARC Phase 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import Dataset

from cir_arc.core.grid import Grid
from cir_arc.core.objects import ArcObject, extract_objects


def compute_boundary_map_from_grid(grid_array: np.ndarray, background_color: int = 0) -> np.ndarray:
    """Computes a binary boundary map from a 2D ARC discrete grid.

    A cell is marked as boundary (1.0) if it is part of an object (non-background) and
    has at least one 4-neighbor that is background or belongs to a different colored object.

    Args:
        grid_array: 2D numpy array with discrete color indices (0-9).
        background_color: Background color index (default: 0).

    Returns:
        2D float32 numpy array of shape (H, W) with values in {0.0, 1.0}.
    """
    H, W = grid_array.shape
    boundary_map = np.zeros((H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            color = grid_array[r, c]
            if color == background_color:
                continue

            # Check 4-connected neighbors
            is_boundary = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    is_boundary = True
                    break
                elif grid_array[nr, nc] != color:
                    is_boundary = True
                    break

            if is_boundary:
                boundary_map[r, c] = 1.0

    return boundary_map


class SyntheticArcDataset(Dataset):
    """PyTorch Dataset loading synthetic ARC tasks for object-centric perception.

    Loads task JSON files from synthetic data directories (e.g. data/synthetic/train/),
    extracts ground-truth ArcObject instances for supervised property heads,
    computes boundary and objectness maps, and returns PyTorch Tensors representing grids.

    Args:
        data_dir: Path to directory containing task JSON files.
        split: Optional split name ('train', 'test', etc.).
        max_samples: Optional limit on total number of samples loaded.
    """

    def __init__(
        self,
        data_dir: str = "data/synthetic/train",
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.file_paths: List[str] = []

        if os.path.exists(data_dir):
            all_files = sorted(
                str(p)
                for p in Path(data_dir).rglob("*.json")
                if not p.name.startswith(".")
            )
            if max_samples is not None and max_samples > 0:
                self.file_paths = all_files[:max_samples]
            else:
                self.file_paths = all_files

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a task JSON and extract input grid, ground-truth objects, and boundary maps.

        Args:
            idx: Sample index in dataset.

        Returns:
            Dict containing:
                - input_grid / grid: LongTensor of shape (H, W)
                - boundary_map: FloatTensor of shape (H, W)
                - objectness_map: FloatTensor of shape (H, W)
                - gt_objects / objects: List[ArcObject]
                - height / H: int
                - width / W: int
                - task_id: str
        """
        filepath = self.file_paths[idx]
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract grid array from train examples (or fallback to test / input)
        if "train" in data and len(data["train"]) > 0:
            grid_list = data["train"][0]["input"]
        elif "test" in data and len(data["test"]) > 0:
            grid_list = data["test"][0]["input"]
        elif "input" in data:
            grid_list = data["input"]
        else:
            grid_list = [[0]]

        arr = np.array(grid_list, dtype=np.int8)
        H, W = arr.shape

        # Extract ground truth objects using Phase 1 core logic
        grid_obj = Grid(arr)
        gt_objects = extract_objects(grid_obj, background_color=0)

        # Compute ground truth boundary and objectness maps
        boundary_np = compute_boundary_map_from_grid(arr, background_color=0)
        objectness_np = (arr != 0).astype(np.float32)

        tensor_grid = torch.from_numpy(arr).long()
        boundary_tensor = torch.from_numpy(boundary_np).float()
        objectness_tensor = torch.from_numpy(objectness_np).float()
        task_id = data.get("task_id", os.path.splitext(os.path.basename(filepath))[0])

        return {
            "input_grid": tensor_grid,
            "grid": tensor_grid,
            "boundary_map": boundary_tensor,
            "objectness_map": objectness_tensor,
            "gt_objects": gt_objects,
            "objects": gt_objects,
            "height": H,
            "H": H,
            "width": W,
            "W": W,
            "task_id": task_id,
        }


# Alias for backward and naming compatibility
SyntheticDataset = SyntheticArcDataset


def collate_variable_grids(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function padding variable-sized ARC grids to batch maximum dimensions.

    Pads grids to (B, max_H, max_W) using mask/pad token 10, constructs binary spatial masks
    (1.0 for valid pixels, 0.0 for padded pixels), aggregates boundary and objectness targets,
    and aggregates ground-truth object lists.

    Args:
        batch: List of dictionaries returned by SyntheticArcDataset.__getitem__.

    Returns:
        Dict containing:
            - input_grids: LongTensor of shape (B, max_H, max_W)
            - input_masks: FloatTensor of shape (B, max_H, max_W) with 1.0 at valid cells
            - boundary_targets: FloatTensor of shape (B, 1, max_H, max_W)
            - objectness_targets: FloatTensor of shape (B, 1, max_H, max_W)
            - gt_objects: List of List[ArcObject] per sample
            - heights: List[int]
            - widths: List[int]
            - task_ids: List[str]
    """
    B = len(batch)
    grids = [
        item["input_grid"] if "input_grid" in item else item["grid"]
        for item in batch
    ]
    gt_objects = [
        item["gt_objects"] if "gt_objects" in item else item.get("objects", [])
        for item in batch
    ]
    heights = [
        item["height"] if "height" in item else item.get("H", g.shape[0])
        for item, g in zip(batch, grids)
    ]
    widths = [
        item["width"] if "width" in item else item.get("W", g.shape[1])
        for item, g in zip(batch, grids)
    ]
    task_ids = [item.get("task_id", "") for item in batch]

    max_H = max(g.shape[0] for g in grids)
    max_W = max(g.shape[1] for g in grids)

    # Pad token 10 reserved for padding/masking
    padded_grids = torch.full((B, max_H, max_W), 10, dtype=torch.long)
    padded_masks = torch.zeros((B, max_H, max_W), dtype=torch.float32)
    boundary_targets = torch.zeros((B, 1, max_H, max_W), dtype=torch.float32)
    objectness_targets = torch.zeros((B, 1, max_H, max_W), dtype=torch.float32)

    for i, item in enumerate(batch):
        g = item["input_grid"] if "input_grid" in item else item["grid"]
        h, w = g.shape
        padded_grids[i, :h, :w] = g
        padded_masks[i, :h, :w] = 1.0

        if "boundary_map" in item:
            boundary_targets[i, 0, :h, :w] = item["boundary_map"]
        else:
            b_np = compute_boundary_map_from_grid(g.cpu().numpy().astype(np.int8))
            boundary_targets[i, 0, :h, :w] = torch.from_numpy(b_np)

        if "objectness_map" in item:
            objectness_targets[i, 0, :h, :w] = item["objectness_map"]
        else:
            objectness_targets[i, 0, :h, :w] = (g != 0).float()

    return {
        "input_grids": padded_grids,
        "grids": padded_grids,
        "grid": padded_grids,
        "input_masks": padded_masks,
        "masks": padded_masks,
        "mask": padded_masks,
        "boundary_targets": boundary_targets,
        "boundary_map": boundary_targets,
        "objectness_targets": objectness_targets,
        "objectness_map": objectness_targets,
        "gt_objects": gt_objects,
        "objects": gt_objects,
        "heights": heights,
        "H": heights,
        "widths": widths,
        "W": widths,
        "task_ids": task_ids,
    }


if __name__ == "__main__":
    print("Running SyntheticArcDataset and collate smoke tests...")

    dummy_samples = [
        {
            "grid": torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.long),
            "objects": [],
            "height": 2,
            "width": 3,
            "boundary_map": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
            "objectness_map": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
        },
        {
            "grid": torch.tensor([[2, 2], [2, 2], [0, 0]], dtype=torch.long),
            "objects": [],
            "height": 3,
            "width": 2,
            "boundary_map": torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
            "objectness_map": torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        },
    ]

    col = collate_variable_grids(dummy_samples)
    assert col["input_grids"].shape == (2, 3, 3)
    assert col["input_masks"].shape == (2, 3, 3)
    assert col["boundary_targets"].shape == (2, 1, 3, 3)
    assert col["objectness_targets"].shape == (2, 1, 3, 3)
    print("Collate tests passed successfully!")
