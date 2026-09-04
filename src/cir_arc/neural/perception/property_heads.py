"""Property prediction heads module for object slots in CIR-ARC Phase 2 / 2.5.

Implements parallel lightweight MLPs predicting structured properties from slot representations:
1. color: 10-class logits (B, K, 10)
2. shape: 8-class logits (B, K, 8)
3. size: 1D normalized scalar in [0, 1] (B, K, 1)
4. position: 2D normalized (row, col) coordinates in [0, 1] (B, K, 2)
5. orientation: 4-class logits (B, K, 4)
6. symmetry: 4 binary logits (B, K, 4)
7. extent: 4D normalized bounding box (min_r, min_c, max_r, max_c) in [0, 1] (B, K, 4)
8. has_holes: 1D binary logit for topological hole/void detection (B, K, 1)
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class PropertyHeads(nn.Module):
    """Parallel MLP property prediction heads operating on object slot representations."""

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 64,
        num_colors: int = 10,
        num_shapes: int = 8,
        num_orientations: int = 4,
        num_symmetries: int = 4,
        include_extent: bool = False,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.include_extent = include_extent

        # 1. Color Head: 10-class logits
        self.color_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_colors),
        )

        # 2. Shape Head: 8-class logits
        self.shape_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_shapes),
        )

        # 3. Size Head: 1D normalized scalar in [0, 1]
        self.size_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 4. Position Head: 2D normalized (row, col) coordinates in [0, 1]
        self.position_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

        # 5. Orientation Head: 4-class logits (0, 90, 180, 270 degrees)
        self.orientation_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_orientations),
        )

        # 6. Symmetry Head: 4 binary logits (horizontal, vertical, main diag, anti diag)
        self.symmetry_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_symmetries),
        )

        # Optional v2 heads
        if include_extent:
            self.extent_head = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),
                nn.Sigmoid(),
            )
            self.hole_head = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.extent_head = None
            self.hole_head = None

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict structured properties for each slot representation.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).

        Returns:
            Dictionary containing predicted properties per slot.
        """
        res = {
            "color": self.color_head(slots),
            "shape": self.shape_head(slots),
            "size": self.size_head(slots),
            "position": self.position_head(slots),
            "orientation": self.orientation_head(slots),
            "symmetry": self.symmetry_head(slots),
        }

        if self.include_extent and self.extent_head is not None and self.hole_head is not None:
            res["extent"] = self.extent_head(slots)
            res["has_holes"] = self.hole_head(slots)
        else:
            # Deterministic geometric extent derived from centroid position and size
            pos = res["position"]
            sz = res["size"]
            half_sz = sz * 0.5
            min_r = (pos[..., 0:1] - half_sz).clamp(0.0, 1.0)
            min_c = (pos[..., 1:2] - half_sz).clamp(0.0, 1.0)
            max_r = (pos[..., 0:1] + half_sz).clamp(0.0, 1.0)
            max_c = (pos[..., 1:2] + half_sz).clamp(0.0, 1.0)
            res["extent"] = torch.cat([min_r, min_c, max_r, max_c], dim=-1)
            res["has_holes"] = torch.zeros((slots.shape[0], slots.shape[1], 1), device=slots.device)

        return res


if __name__ == "__main__":
    heads = PropertyHeads(slot_dim=128)
    assert sum(p.numel() for p in heads.parameters()) == 51421
    print("PropertyHeads exact baseline parameters verified: 51,421")
