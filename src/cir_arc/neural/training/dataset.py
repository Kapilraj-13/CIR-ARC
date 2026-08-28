"""Dataset and variable grid batch collate utilities for CIR-ARC Phase 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from cir_arc.core.grid import Grid
from cir_arc.core.objects import ArcObject, extract_objects


class SyntheticArcDataset(Dataset):
    """PyTorch Dataset loading synthetic ARC tasks for object-centric perception.

    Loads task JSON files from synthetic data directories (e.g. data/synthetic/train/),
    extracts ground-truth ArcObject instances for supervised property heads,
    and returns PyTorch LongTensors representing the 2D discrete grids.

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
        """Load a task JSON and extract input grid and ground-truth objects.

        Args:
            idx: Sample index in dataset.

        Returns:
            Dict containing:
                - input_grid / grid: LongTensor of shape (H, W)
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

        tensor_grid = torch.from_numpy(arr).long()
        task_id = data.get("task_id", os.path.splitext(os.path.basename(filepath))[0])

        return {
            "input_grid": tensor_grid,
            "grid": tensor_grid,
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
    (1.0 for valid pixels, 0.0 for padded pixels), and aggregates ground-truth object lists.

    Args:
        batch: List of dictionaries returned by SyntheticArcDataset.__getitem__.

    Returns:
        Dict containing:
            - input_grids: LongTensor of shape (B, max_H, max_W)
            - input_masks: FloatTensor of shape (B, max_H, max_W) with 1.0 at valid cells
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

    for i, g in enumerate(grids):
        h, w = g.shape
        padded_grids[i, :h, :w] = g
        padded_masks[i, :h, :w] = 1.0

    return {
        "input_grids": padded_grids,
        "grids": padded_grids,
        "grid": padded_grids,
        "input_masks": padded_masks,
        "masks": padded_masks,
        "mask": padded_masks,
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

    # Test collate function on synthetic sample dicts
    sample1 = {
        "input_grid": torch.randint(0, 10, (4, 7), dtype=torch.long),
        "height": 4,
        "width": 7,
        "gt_objects": [ArcObject(color=1, pixels=np.array([[0, 0]]))],
        "task_id": "test_task_1",
    }
    sample2 = {
        "input_grid": torch.randint(0, 10, (9, 5), dtype=torch.long),
        "height": 9,
        "width": 5,
        "gt_objects": [ArcObject(color=2, pixels=np.array([[1, 1]]))],
        "task_id": "test_task_2",
    }

    batch = collate_variable_grids([sample1, sample2])
    assert batch["input_grids"].shape == (2, 9, 7), f"Unexpected shape {batch['input_grids'].shape}"
    assert batch["input_masks"].shape == (2, 9, 7), f"Unexpected mask shape {batch['input_masks'].shape}"
    assert (batch["input_masks"][0, :4, :7] == 1.0).all()
    assert (batch["input_masks"][0, 4:, :] == 0.0).all()
    assert (batch["input_masks"][1, :9, :5] == 1.0).all()
    assert (batch["input_masks"][1, :, 5:] == 0.0).all()
    print("collate_variable_grids passed smoke test successfully!")

    # Test dataset loading if synthetic directory exists
    train_dir = "data/synthetic/train"
    if os.path.exists(train_dir):
        ds = SyntheticArcDataset(data_dir=train_dir, max_samples=10)
        print(f"Loaded SyntheticArcDataset with {len(ds)} items.")
        if len(ds) > 0:
            first = ds[0]
            assert "input_grid" in first and "gt_objects" in first
            print(f"First item: shape {first['input_grid'].shape}, {len(first['gt_objects'])} objects.")

    print("All dataset smoke tests passed successfully!")
