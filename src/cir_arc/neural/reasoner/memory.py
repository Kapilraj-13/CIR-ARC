"""Dual Memory System for the CIR-ARC ~120.18M Reasoner.

Contains:
1. Reasoning Workspace (R_t): 128 ephemeral latent tokens rewritten during reasoning iterations.
2. Working Memory (M_t): 128 persistent recurrent cognitive tokens updated via Cross-Attention:
       Query from current state/reasoning, Key/Value from working memory.
3. Episodic Memory Retrieval: Content-based retrieval of historical compressed event tuples.

Total parameters: 3,544,832.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemorySystem(nn.Module):
    """Integrates Ephemeral Reasoning Workspace, Persistent Working Memory, and Episodic Retrieval."""

    def __init__(
        self,
        d_model: int = 768,
        num_reasoning: int = 128,
        num_wm: int = 128,
        retrieval_dim: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_reasoning = num_reasoning
        self.num_wm = num_wm

        # 1. Ephemeral Reasoning Workspace Tokens (128 x 768 = 98,304)
        self.reasoning_tokens = nn.Parameter(torch.randn(num_reasoning, d_model) * 0.02)

        # 2. Persistent Working Memory Base Tokens (128 x 768 = 98,304)
        self.wm_tokens = nn.Parameter(torch.randn(num_wm, d_model) * 0.02)

        # Working Memory Cross-Attention Update (4 x 590,592 + 1,536 = 2,363,904)
        self.wm_q = nn.Linear(d_model, d_model)
        self.wm_k = nn.Linear(d_model, d_model)
        self.wm_v = nn.Linear(d_model, d_model)
        self.wm_out = nn.Linear(d_model, d_model)
        self.wm_norm = nn.LayerNorm(d_model)

        # 3. Episodic Memory Retrieval Projections (196,864 + 196,864 + 590,592 = 984,320)
        self.retrieval_q = nn.Linear(d_model, retrieval_dim)
        self.retrieval_k = nn.Linear(d_model, retrieval_dim)
        self.retrieval_v = nn.Linear(d_model, d_model)

        self.scale_wm = 1.0 / math.sqrt(d_model)
        self.scale_ret = 1.0 / math.sqrt(retrieval_dim)

    def get_initial_reasoning_workspace(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns fresh ephemeral reasoning workspace tokens [B, 128, d_model]."""
        return self.reasoning_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def get_initial_working_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns initial persistent working memory tokens [B, 128, d_model]."""
        return self.wm_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def update_working_memory(
        self,
        current_state: torch.Tensor,
        prev_wm: torch.Tensor,
    ) -> torch.Tensor:
        """Updates persistent working memory via cross-attention.

        Args:
            current_state: Fused cognitive state tokens [B, S, d_model].
            prev_wm: Previous working memory state [B, 128, d_model].

        Returns:
            Updated working memory [B, 128, d_model].
        """
        # Queries from current state/reasoning representation
        q = self.wm_q(prev_wm)           # [B, 128, d_model]
        k = self.wm_k(current_state)      # [B, S, d_model]
        v = self.wm_v(current_state)      # [B, S, d_model]

        # Cross-attention weights
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale_wm
        attn_weights = F.softmax(attn_scores, dim=-1)

        attended = torch.bmm(attn_weights, v)
        updated_wm = prev_wm + self.wm_out(attended)
        return self.wm_norm(updated_wm)

    def retrieve_episodic_memory(
        self,
        query_state: torch.Tensor,
        episodic_keys: Optional[torch.Tensor] = None,
        episodic_values: Optional[torch.Tensor] = None,
        top_k: int = 8,
    ) -> Optional[torch.Tensor]:
        """Retrieves relevant historical events from episodic memory.

        Args:
            query_state: Current cognitive query [B, 1, d_model].
            episodic_keys: Historical event state descriptors [B, T, d_model].
            episodic_values: Historical event compressed memories [B, T, d_model].
            top_k: Number of historical memories to retrieve.

        Returns:
            Retrieved memory tokens [B, top_k, d_model] or None if no episodic history.
        """
        if episodic_keys is None or episodic_values is None or episodic_keys.shape[1] == 0:
            return None

        q = self.retrieval_q(query_state)    # [B, 1, 256]
        k = self.retrieval_k(episodic_keys)  # [B, T, 256]
        v = self.retrieval_v(episodic_values)# [B, T, d_model]

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale_ret  # [B, 1, T]
        k_actual = min(top_k, scores.shape[-1])
        topk_scores, topk_indices = torch.topk(scores, k=k_actual, dim=-1)  # [B, 1, k_actual]
        topk_weights = F.softmax(topk_scores, dim=-1)

        # Gather top-k values
        # topk_indices: [B, 1, k_actual] -> expand for gather
        idx_expanded = topk_indices.expand(-1, -1, self.d_model)
        # Re-index values
        batch_size = query_state.shape[0]
        retrieved_list = []
        for b in range(batch_size):
            b_vals = episodic_values[b]  # [T, d_model]
            b_idx = topk_indices[b, 0]   # [k_actual]
            retrieved_list.append(b_vals[b_idx])
        retrieved = torch.stack(retrieved_list, dim=0)  # [B, k_actual, d_model]
        return retrieved
