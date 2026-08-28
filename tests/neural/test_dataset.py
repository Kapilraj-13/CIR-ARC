"""Unit tests for SyntheticArcDataset and variable grid collate function (Phase 2)."""

import os
import pytest
import torch
from torch.utils.data import DataLoader

from cir_arc.core.objects import ArcObject

# Safe import for progressive testability during milestone builds
dataset_mod = pytest.importorskip("cir_arc.neural.training.dataset")
SyntheticArcDataset = dataset_mod.SyntheticArcDataset
collate_variable_grids = dataset_mod.collate_variable_grids


def test_synthetic_dataset_instantiation():
    """Verify SyntheticArcDataset initializes and finds tasks in data/synthetic/train."""
    train_dir = "data/synthetic/train"
    if not os.path.exists(train_dir):
        pytest.skip(f"{train_dir} does not exist in workspace")

    dataset = SyntheticArcDataset(data_dir=train_dir)
    assert len(dataset) > 0


def test_synthetic_dataset_getitem_structure():
    """Verify __getitem__ returns a dictionary with input_grid, gt_objects, and metadata."""
    train_dir = "data/synthetic/train"
    if not os.path.exists(train_dir):
        pytest.skip(f"{train_dir} does not exist in workspace")

    dataset = SyntheticArcDataset(data_dir=train_dir)
    item = dataset[0]

    assert isinstance(item, dict)
    assert "input_grid" in item
    assert "gt_objects" in item
    assert "height" in item
    assert "width" in item

    assert isinstance(item["input_grid"], torch.Tensor)
    assert item["input_grid"].dtype == torch.long
    assert item["input_grid"].dim() == 2  # (H, W)

    h, w = item["input_grid"].shape
    assert item["height"] == h
    assert item["width"] == w

    assert isinstance(item["gt_objects"], list)
    for obj in item["gt_objects"]:
        assert isinstance(obj, ArcObject)
        assert 0 <= obj.color <= 9
        assert len(obj.pixels) > 0


def test_collate_variable_grids_padding():
    """Verify collate_variable_grids dynamically pads heterogeneous grid shapes to max(H), max(W)."""
    # Sample 1: 4x7 grid
    sample1 = {
        "input_grid": torch.randint(0, 10, (4, 7), dtype=torch.long),
        "height": 4,
        "width": 7,
        "gt_objects": [ArcObject(color=1, pixels=[(0, 0)])],
        "task_id": "task_1",
    }
    # Sample 2: 9x5 grid
    sample2 = {
        "input_grid": torch.randint(0, 10, (9, 5), dtype=torch.long),
        "height": 9,
        "width": 5,
        "gt_objects": [ArcObject(color=2, pixels=[(1, 1)])],
        "task_id": "task_2",
    }

    batch = collate_variable_grids([sample1, sample2])

    assert "input_grids" in batch
    assert "input_masks" in batch
    assert "gt_objects" in batch

    # Max H is 9, Max W is 7 -> batch shape (2, 9, 7)
    assert batch["input_grids"].shape == (2, 9, 7)
    assert batch["input_masks"].shape == (2, 9, 7)

    # Sample 1 mask: 1.0 for :4, :7 and 0.0 elsewhere
    assert (batch["input_masks"][0, :4, :7] == 1.0).all()
    assert (batch["input_masks"][0, 4:, :] == 0.0).all()

    # Sample 2 mask: 1.0 for :9, :5 and 0.0 elsewhere
    assert (batch["input_masks"][1, :9, :5] == 1.0).all()
    assert (batch["input_masks"][1, :, 5:] == 0.0).all()

    # Grid values inside valid region should match original
    assert torch.equal(batch["input_grids"][0, :4, :7], sample1["input_grid"])
    assert torch.equal(batch["input_grids"][1, :9, :5], sample2["input_grid"])


def test_dataloader_batch_iteration():
    """Verify DataLoader successfully batches and iterates through SyntheticArcDataset."""
    train_dir = "data/synthetic/train"
    if not os.path.exists(train_dir):
        pytest.skip(f"{train_dir} does not exist in workspace")

    dataset = SyntheticArcDataset(data_dir=train_dir)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_variable_grids,
    )

    batch = next(iter(loader))
    assert "input_grids" in batch
    assert "input_masks" in batch
    assert batch["input_grids"].dim() == 3  # (B, H_max, W_max)
    assert batch["input_grids"].shape[0] <= 4
    assert batch["input_masks"].shape == batch["input_grids"].shape
