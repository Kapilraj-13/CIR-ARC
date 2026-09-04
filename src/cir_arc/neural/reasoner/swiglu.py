"""SwiGLU Feed-Forward Network for CIR-ARC Reasoner.

Configuration:
- d_model = 768
- d_ff = 1856 (calibrated for exact ~120.18M total reasoner budget)
- Parameters: W_gate (768x1856) + W_up (768x1856) + W_down (1856x768) = 4,276,224
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network layer.

    Formula:
        SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, d_model: int = 768, d_ff: int = 1856, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Standard LLaMA-style SwiGLU without bias
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # 768 -> 1856: 1,425,408
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)    # 768 -> 1856: 1,425,408
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # 1856 -> 768: 1,425,408
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].

        Returns:
            Output tensor [batch_size, seq_len, d_model].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden)
        return out
