"""Input token encoders and dense projections for the CIR-ARC ~120.18M Reasoner.

Fuses:
1. SymbolicSceneState entities, relations, events, mechanics beliefs, global state, uncertainty
   (with explicit UNKNOWN / OTHER fallback channels for OOD novel mechanics).
2. DenseLatentState slots, spatial tokens (compressed to 128 tokens), relation latents, temporal latents.
3. Adaptive Gated Neuro-Symbolic Fusion: z = g * z_dense + (1 - g) * z_symbolic.

Total parameters: 5,095,936.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.neural.reasoner.config import ReasonerConfig


class SymbolicEntityEncoder(nn.Module):
    """Encodes structured objects into 768-D tokens.

    Total parameters: 465,792.
    """

    def __init__(self, d_model: int = 768, num_colors: int = 12, num_shapes: int = 10, num_orientations: int = 5) -> None:
        super().__init__()
        self.color_emb = nn.Embedding(num_colors, 64)
        self.geom_mlp = nn.Sequential(
            nn.Linear(11, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.shape_emb = nn.Embedding(num_shapes, 64)
        self.orient_emb = nn.Embedding(num_orientations, 32)
        self.sym_proj = nn.Linear(4, 32)
        self.hole_emb = nn.Embedding(2, 16)
        self.kinematics_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.affordance_proj = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.uncertainty_proj = nn.Linear(4, 32)
        self.identity_proj = nn.Linear(64, 64)

        in_dim = 64 + 128 + 64 + 32 + 32 + 16 + 64 + 64 + 32 + 64  # 560
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        color: torch.Tensor,
        geom: torch.Tensor,
        shape: torch.Tensor,
        orientation: torch.Tensor,
        symmetries: torch.Tensor,
        holes: torch.Tensor,
        kinematics: torch.Tensor,
        affordances: torch.Tensor,
        uncertainties: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        """Encode object attributes into token representations."""
        c = self.color_emb(color.clamp(0, self.color_emb.num_embeddings - 1))
        g = self.geom_mlp(geom)
        s = self.shape_emb(shape.clamp(0, self.shape_emb.num_embeddings - 1))
        o = self.orient_emb(orientation.clamp(0, self.orient_emb.num_embeddings - 1))
        sym = self.sym_proj(symmetries.float())
        h = self.hole_emb(holes.clamp(0, 1))
        k = self.kinematics_mlp(kinematics)
        aff = self.affordance_proj(affordances)
        unc = self.uncertainty_proj(uncertainties)
        ident = self.identity_proj(identity)

        concat = torch.cat([c, g, s, o, sym, h, k, aff, unc, ident], dim=-1)
        return self.fusion(concat)


class RelationTokenEncoder(nn.Module):
    """Encodes pairwise relational edges into 768-D tokens.

    Total parameters: 939,072.
    """

    def __init__(self, d_model: int = 768, num_relations: int = 16) -> None:
        super().__init__()
        self.sub_proj = nn.Linear(d_model, 256)
        self.obj_proj = nn.Linear(d_model, 256)
        self.rel_type_emb = nn.Embedding(num_relations, 128)
        self.conf_proj = nn.Linear(4, 64)
        self.out_proj = nn.Sequential(
            nn.Linear(256 + 256 + 128 + 64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        sub_vec: torch.Tensor,
        obj_vec: torch.Tensor,
        rel_type: torch.Tensor,
        conf_dist: torch.Tensor,
    ) -> torch.Tensor:
        s = self.sub_proj(sub_vec)
        o = self.obj_proj(obj_vec)
        r = self.rel_type_emb(rel_type.clamp(0, self.rel_type_emb.num_embeddings - 1))
        c = self.conf_proj(conf_dist)
        concat = torch.cat([s, o, r, c], dim=-1)
        return self.out_proj(concat)


class EventTokenEncoder(nn.Module):
    """Encodes transition events into 768-D tokens.

    Total parameters: 742,336.
    """

    def __init__(self, d_model: int = 768, num_events: int = 16) -> None:
        super().__init__()
        self.event_type_emb = nn.Embedding(num_events, 128)
        self.src_proj = nn.Linear(d_model, 192)
        self.tgt_proj = nn.Linear(d_model, 192)
        self.step_conf_proj = nn.Linear(4, 64)
        self.out_proj = nn.Sequential(
            nn.Linear(128 + 192 + 192 + 64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        event_type: torch.Tensor,
        src_vec: torch.Tensor,
        tgt_vec: torch.Tensor,
        step_conf: torch.Tensor,
    ) -> torch.Tensor:
        e = self.event_type_emb(event_type.clamp(0, self.event_type_emb.num_embeddings - 1))
        s = self.src_proj(src_vec)
        t = self.tgt_proj(tgt_vec)
        sc = self.step_conf_proj(step_conf)
        concat = torch.cat([e, s, t, sc], dim=-1)
        return self.out_proj(concat)


class MechanicsBeliefEncoder(nn.Module):
    """Encodes online mechanics hypothesis parameters into 768-D token.

    Total parameters: 201,984.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(11, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, mechanics_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(mechanics_vec)


class GlobalStateEncoder(nn.Module):
    """Encodes HUD and global environment state into 768-D token.

    Total parameters: 218,624.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.inv_emb = nn.Embedding(16, 64)
        self.mlp = nn.Sequential(
            nn.Linear(8 + 64, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, hud_vec: torch.Tensor, inventory_token_id: torch.Tensor) -> torch.Tensor:
        inv = self.inv_emb(inventory_token_id.clamp(0, 15))
        concat = torch.cat([hud_vec, inv], dim=-1)
        return self.mlp(concat)


class ActionEffectEncoder(nn.Module):
    """Encodes predicted causal action effects into 768-D token.

    Total parameters: 202,752.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(14, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, effect_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(effect_vec)


class UncertaintyEncoder(nn.Module):
    """Encodes uncertainty summary metrics into 768-D token.

    Total parameters: 101,248.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, uncertainty_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(uncertainty_vec)


class DenseLatentProjections(nn.Module):
    """Projects continuous neural features from the 3.5M Perception model.

    Compresses 2D dense spatial maps into 128 tokens via adaptive pooling.
    Applies learned gated Neuro-Symbolic fusion:
        z = g * z_dense + (1 - g) * z_symbolic.

    Total parameters: 2,224,128.
    """

    def __init__(
        self,
        slot_dim: int = 224,
        feat_dim: int = 224,
        d_model: int = 768,
        num_spatial_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.num_spatial_tokens = num_spatial_tokens

        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )
        self.temp_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def compress_spatial_features(self, spatial_map: torch.Tensor) -> torch.Tensor:
        """Compress dense (B, H, W, feat_dim) or (B, N, feat_dim) into exactly num_spatial_tokens."""
        if spatial_map.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            b, h, w, c = spatial_map.shape
            x = spatial_map.permute(0, 3, 1, 2)
            # Adaptive pool to fixed length, e.g. 16x8 = 128
            target_h = 16
            target_w = self.num_spatial_tokens // target_h
            pooled = F.adaptive_avg_pool2d(x, (target_h, target_w))  # (B, C, target_h, target_w)
            compressed = pooled.flatten(2).transpose(1, 2)  # (B, 128, C)
            return compressed
        elif spatial_map.dim() == 3:
            b, n, c = spatial_map.shape
            if n == self.num_spatial_tokens:
                return spatial_map
            # 1D adaptive average pooling: (B, C, N) -> (B, C, 128)
            x = spatial_map.transpose(1, 2)
            pooled = F.adaptive_avg_pool1d(x, self.num_spatial_tokens)
            return pooled.transpose(1, 2)
        return spatial_map

    def fuse_neuro_symbolic(self, z_dense: torch.Tensor, z_symbolic: torch.Tensor) -> torch.Tensor:
        """Adaptive Gated Fusion between dense continuous signals and discrete symbolic tokens."""
        concat = torch.cat([z_dense, z_symbolic], dim=-1)
        g = self.gate(concat)
        return g * z_dense + (1.0 - g) * z_symbolic
