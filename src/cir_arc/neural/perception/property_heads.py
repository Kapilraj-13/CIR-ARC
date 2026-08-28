"""Property prediction heads module for object slots in CIR-ARC.

Implements 6 parallel lightweight MLPs predicting structured properties from slot representations:
1. color: 10-class logits (B, K, 10)
2. shape: 8-class logits (B, K, 8)
3. size: 1D normalized scalar in [0, 1] (B, K, 1)
4. position: 2D normalized (row, col) coordinates in [0, 1] (B, K, 2)
5. orientation: 4-class logits (B, K, 4)
6. symmetry: 4 binary logits (B, K, 4)
"""

from typing import Dict
import torch
import torch.nn as nn


class PropertyHeads(nn.Module):
    """Parallel MLP property prediction heads operating on object slot representations.

    Translates abstract D-dimensional slot representations into structured, interpretable
    symbolic attributes aligned with ArcObject definitions.

    Args:
        slot_dim: Dimensionality of input slot representations (default: 128).
        hidden_dim: Hidden dimension of the intermediate MLP layer (default: 64).
        num_colors: Number of color classes (default: 10).
        num_shapes: Number of shape categories (default: 8).
        num_orientations: Number of orientation bins (default: 4).
        num_symmetries: Number of binary symmetry axes (default: 4).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 64,
        num_colors: int = 10,
        num_shapes: int = 8,
        num_orientations: int = 4,
        num_symmetries: int = 4,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim

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

        # 6. Symmetry Head: 4 binary logits (H-sym, V-sym, MainDiag-sym, AntiDiag-sym)
        self.symmetry_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_symmetries),
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
                - "orientation": (B, K, 4) logits
                - "symmetry": (B, K, 4) binary logits
        """
        return {
            "color": self.color_head(slots),
            "shape": self.shape_head(slots),
            "size": self.size_head(slots),
            "position": self.position_head(slots),
            "orientation": self.orientation_head(slots),
            "symmetry": self.symmetry_head(slots),
        }


if __name__ == "__main__":
    print("Running PropertyHeads smoke test...")
    model = PropertyHeads()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PropertyHeads parameter count: {param_count}")
    assert param_count == 51421, f"Expected 51,421 parameters, got {param_count}"

    test_shapes = [(4, 24, 128), (1, 24, 128), (2, 24, 128)]
    for shape in test_shapes:
        B, K, D = shape
        dummy_slots = torch.randn(shape)
        props = model(dummy_slots)

        assert "color" in props and props["color"].shape == (B, K, 10)
        assert "shape" in props and props["shape"].shape == (B, K, 8)
        assert "size" in props and props["size"].shape == (B, K, 1)
        assert "position" in props and props["position"].shape == (B, K, 2)
        assert "orientation" in props and props["orientation"].shape == (B, K, 4)
        assert "symmetry" in props and props["symmetry"].shape == (B, K, 4)

        # Range checks for Sigmoid activations
        assert (props["size"] >= 0.0).all() and (props["size"] <= 1.0).all(), "Size not in [0, 1]"
        assert (props["position"] >= 0.0).all() and (props["position"] <= 1.0).all(), "Position not in [0, 1]"

        print(f"Passed test for {shape} -> all 6 property head outputs validated successfully")

    print("All PropertyHeads smoke tests passed successfully!")
