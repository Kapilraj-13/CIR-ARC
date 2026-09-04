"""PyTorch Dataset and variable-size collate utility for 5-layer ARC Trajectories.

Features:
- Explicit boolean valid_mask: Separates genuine ARC color palette (0-9) from padding.
  Does NOT encode padding as color 10.
- Trajectory batching: Provides (grid_t, action, grid_next) alongside ground truth
  events, action effects, and mechanics evidence for causal transition training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from cir_arc.generators.trajectory_dataset import TrajectoryStep


class TrajectoryArcDataset(Dataset):
    """Loads 5-layer trajectory steps for causal dynamics and mechanics learning."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        in_memory_steps: Optional[List[TrajectoryStep]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.steps: List[TrajectoryStep] = in_memory_steps or []

        if data_dir is not None and os.path.exists(data_dir):
            files = sorted(Path(data_dir).glob("traj_*.json"))
            if max_samples:
                files = files[:max_samples]
            for f in files:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    self.steps.append(TrajectoryStep(**d))

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        step = self.steps[idx]
        grid_t = np.array(step.grid_t, dtype=np.int64)
        mask_t = np.array(step.valid_mask_t, dtype=bool)
        grid_next = np.array(step.grid_next, dtype=np.int64)
        mask_next = np.array(step.valid_mask_next, dtype=bool)

        return {
            "grid_t": torch.from_numpy(grid_t),
            "valid_mask_t": torch.from_numpy(mask_t),
            "action": int(step.action),
            "grid_next": torch.from_numpy(grid_next),
            "valid_mask_next": torch.from_numpy(mask_next),
            "events": step.events,
            "action_effect": step.action_effect,
            "mechanics_evidence": step.mechanics_evidence,
            "layer": int(step.layer),
            "is_negative": bool(step.is_negative_example),
            "H": grid_t.shape[0],
            "W": grid_t.shape[1],
        }


def collate_trajectory_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates variable-sized trajectory steps with explicit boolean valid_mask.

    Pads grids with background color 0 and sets valid_mask to False on padded boundaries.
    """
    max_h = max(item["H"] for item in batch)
    max_w = max(item["W"] for item in batch)
    B = len(batch)

    padded_grid_t = torch.zeros((B, max_h, max_w), dtype=torch.long)
    padded_mask_t = torch.zeros((B, max_h, max_w), dtype=torch.bool)
    padded_grid_next = torch.zeros((B, max_h, max_w), dtype=torch.long)
    padded_mask_next = torch.zeros((B, max_h, max_w), dtype=torch.bool)

    actions = torch.zeros(B, dtype=torch.long)
    layers = torch.zeros(B, dtype=torch.long)
    is_neg = torch.zeros(B, dtype=torch.bool)
    heights = []
    widths = []

    all_events = []
    all_effects = []
    all_evidence = []

    for b, item in enumerate(batch):
        h, w = item["H"], item["W"]
        heights.append(h)
        widths.append(w)

        padded_grid_t[b, :h, :w] = item["grid_t"]
        padded_mask_t[b, :h, :w] = item["valid_mask_t"]
        padded_grid_next[b, :h, :w] = item["grid_next"]
        padded_mask_next[b, :h, :w] = item["valid_mask_next"]

        actions[b] = item["action"]
        layers[b] = item["layer"]
        is_neg[b] = item["is_negative"]

        all_events.append(item["events"])
        all_effects.append(item["action_effect"])
        all_evidence.append(item["mechanics_evidence"])

    return {
        "grid_t": padded_grid_t,
        "valid_mask_t": padded_mask_t,
        "action": actions,
        "grid_next": padded_grid_next,
        "valid_mask_next": padded_mask_next,
        "heights": heights,
        "widths": widths,
        "events": all_events,
        "action_effects": all_effects,
        "mechanics_evidence": all_evidence,
        "layers": layers,
        "is_negative": is_neg,
    }
