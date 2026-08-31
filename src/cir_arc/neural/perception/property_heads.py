"""Property prediction heads module for object slots in CIR-ARC Phase 2.5.

Implements parallel lightweight MLPs predicting structured properties from slot representations:
1. color: 10-class logits (B, K, 10)
2. shape: 8-class logits (B, K, 8)
3. size: 1D normalized scalar in [0, 1] (B, K, 1)
4. position: 2D normalized (row, col) coordinates in [0, 1] (B, K, 2)
5. bbox: 4D normalized (min_r, min_c, max_r, max_c) in [0, 1] (B, K, 4)
6. dimensions: 4D normalized (width, height, area, perimeter) in [0, 1] (B, K, 4)
7. aspect_ratio: 1D normalized scalar in [0, 1] (B, K, 1)
8. orientation: 4-class logits (B, K, 4)
9. symmetry: 4 binary logits (B, K, 4)
10. holes: 1D binary void presence probability in [0, 1] (B, K, 1)
11. identity: 64D L2-normalized metric embedding vector (B, K, 64)
12. presence: 1D slot confidence / object presence probability (B, K, 1)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PropertyHeads(nn.Module):
    """Parallel MLP property prediction heads operating on object slot representations.

    Translates abstract D-dimensional slot representations into structured, interpretable
    symbolic attributes aligned with ArcObject definitions and StructuredObject models.

    Args:
        slot_dim: Dimensionality of input slot representations (default: 128).
        hidden_dim: Hidden dimension of the intermediate MLP layer (default: 64).
        num_colors: Number of color classes (default: 10).
        num_shapes: Number of shape categories (default: 8).
        num_orientations: Number of orientation bins (default: 4).
        num_symmetries: Number of binary symmetry axes (default: 4).
        identity_dim: Dimensionality of metric identity embedding (default: 64).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 64,
        num_colors: int = 10,
        num_shapes: int = 8,
        num_orientations: int = 4,
        num_symmetries: int = 4,
        identity_dim: int = 64,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.identity_dim = identity_dim

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

        # 5. Bounding Box Head: 4D normalized (min_r, min_c, max_r, max_c) in [0, 1]
        self.bbox_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        # 6. Dimensions Head: 4D normalized (width, height, area, perimeter) in [0, 1]
        self.dimensions_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        # 7. Aspect Ratio Head: 1D scalar
        self.aspect_ratio_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 8. Orientation Head: 4-class logits (0, 90, 180, 270 degrees)
        self.orientation_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_orientations),
        )

        # 9. Symmetry Head: 4 binary logits (H-sym, V-sym, MainDiag-sym, AntiDiag-sym)
        self.symmetry_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_symmetries),
        )

        # 10. Holes / Topology Head: binary probability of enclosed voids
        self.holes_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 11. Metric Identity Head: 64-dim L2-normalized embedding
        self.identity_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, identity_dim),
        )

        # 12. Slot Presence / Confidence Head: calibrated probability in [0, 1]
        self.presence_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict structured object properties from slot representations.

        Args:
            slots: Slot representation tensor of shape (B, K, slot_dim).

        Returns:
            Dictionary containing:
                - "color": (B, K, 10) logits
                - "shape": (B, K, 8) logits
                - "size": (B, K, 1) normalized scalar in [0, 1]
                - "position": (B, K, 2) normalized (row, col) coordinates in [0, 1]
                - "bbox": (B, K, 4) normalized (min_r, min_c, max_r, max_c) in [0, 1]
                - "dimensions": (B, K, 4) normalized (width, height, area, perimeter) in [0, 1]
                - "aspect_ratio": (B, K, 1) normalized scalar in [0, 1]
                - "orientation": (B, K, 4) logits
                - "symmetry": (B, K, 4) binary logits
                - "holes": (B, K, 1) void probability in [0, 1]
                - "identity": (B, K, identity_dim) L2-normalized embedding
                - "presence": (B, K, 1) slot presence confidence in [0, 1]
        """
        ident = self.identity_head(slots)
        norm_ident = F.normalize(ident, p=2, dim=-1)

        return {
            "color": self.color_head(slots),
            "shape": self.shape_head(slots),
            "size": self.size_head(slots),
            "position": self.position_head(slots),
            "bbox": self.bbox_head(slots),
            "dimensions": self.dimensions_head(slots),
            "aspect_ratio": self.aspect_ratio_head(slots),
            "orientation": self.orientation_head(slots),
            "symmetry": self.symmetry_head(slots),
            "holes": self.holes_head(slots),
            "identity": norm_ident,
            "presence": self.presence_head(slots),
        }
