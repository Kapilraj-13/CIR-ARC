"""Trainer module and PerceptionModel integration for CIR-ARC Phase 2.

Encapsulates:
1. PerceptionModel: Full neural perception model (Embedding -> MultiScale CNN -> Proposal Slot Attention
   -> Relational Set Transformer -> PropertyHeads -> ReconstructionDecoder).
2. Trainer: Multi-objective training loop with AdamW optimizer, CosineAnnealingLR,
   gradient clipping, and checkpoint persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.core.objects import ArcObject
from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import MultiScaleCNNStem
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.relation_encoder import SlotRelationEncoder
from cir_arc.neural.perception.property_heads import PropertyHeads
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder
from cir_arc.neural.perception.relation_graph import extract_ground_truth_relations
from cir_arc.neural.world_state import StructuredObject, SpatialRelation, RelationGraph, WorldState
from cir_arc.neural.losses.reconstruction import reconstruction_loss
from cir_arc.neural.losses.property import (
    color_loss,
    position_loss,
    size_loss,
    shape_loss,
    objectness_loss,
)
from cir_arc.neural.losses.boundary import (
    boundary_loss,
    cell_objectness_loss,
)
from cir_arc.neural.losses.matching import hungarian_matching
from cir_arc.neural.losses.diversity import diversity_loss, objectness_sparsity_loss


class PerceptionModel(nn.Module):
    """End-to-end multi-scale neural perception model for CIR-ARC Phase 2.

    Chains:
    1. ColorEmbedding: (B, H, W) -> (B, H, W, embed_dim)
    2. MultiScaleCNNStem: (B, H, W, embed_dim) -> (B, H*W, 128) + (boundary_map, cell_objectness)
    3. ProposalSlotAttention: (B, H*W, 128) -> slots (B, 24, 128), objectness (B, 24), attn (B, 24, H*W)
    4. SlotRelationEncoder: (B, 24, 128) -> refined_slots (B, 24, 128)
    5. PropertyHeads: refined_slots -> {color, shape, size, position, orientation, symmetry}
    6. ReconstructionDecoder: refined_slots -> color logits (B, H, W, 10)
    """

    def __init__(
        self,
        num_colors: int = 11,
        embed_dim: int = 48,
        stem_hidden_dim: int = 64,
        stem_out_dim: int = 128,
        n_slots: int = 24,
        slot_dim: int = 128,
        feat_dim: int = 128,
        n_iter: int = 3,
        relation_layers: int = 2,
        relation_heads: int = 4,
        max_h: int = 30,
        max_w: int = 30,
        prop_hidden_dim: int = 64,
        num_shapes: int = 8,
        num_orientations: int = 4,
        num_symmetries: int = 4,
        recon_num_colors: int = 10,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.max_h = max_h
        self.max_w = max_w

        # 1. Color Embedding
        self.embedding = ColorEmbedding(
            num_colors=num_colors,
            embed_dim=embed_dim,
        )

        # 2. Multi-Scale CNN Feature Stem
        self.cnn_stem = MultiScaleCNNStem(
            in_channels=embed_dim,
            hidden_channels=stem_hidden_dim,
            out_channels=stem_out_dim,
        )

        # 3. Proposal-guided Slot Attention
        self.slot_attention = SlotAttention(
            n_slots=n_slots,
            slot_dim=slot_dim,
            feat_dim=feat_dim,
            n_iter=n_iter,
            eps=1e-8,
            hidden_dim=256,
            proposal_init=True,
        )

        # 4. Relational Set Transformer Object Refinement
        self.relation_encoder = SlotRelationEncoder(
            slot_dim=slot_dim,
            num_heads=relation_heads,
            mlp_hidden_dim=256,
            num_layers=relation_layers,
            dropout=0.0,
        )

        # 5. Symbolic Property Prediction Heads
        self.property_heads = PropertyHeads(
            slot_dim=slot_dim,
            hidden_dim=prop_hidden_dim,
            num_colors=10,
            num_shapes=num_shapes,
            num_orientations=num_orientations,
            num_symmetries=num_symmetries,
        )

        # 6. Spatial Reconstruction Decoder
        self.decoder = ReconstructionDecoder(
            slot_dim=slot_dim,
            max_h=max_h,
            max_w=max_w,
            num_colors=recon_num_colors,
        )

    def forward(
        self,
        grid: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through perception pipeline.

        Args:
            grid: LongTensor of shape (B, H, W) containing discrete color indices (0-10).
            mask: Optional Tensor of shape (B, H, W) or (B, H*W) with 1.0/True at valid cells.

        Returns:
            Dict containing slots, objectness, attn_maps, boundary_map, cell_objectness, props, recon_logits.
        """
        B, H, W = grid.shape

        # Step 1: Color embedding (B, H, W, embed_dim)
        x_embed = self.embedding(grid)

        # Step 2: Multi-scale spatial CNN encoder + proposal maps
        tokens, boundary_map, cell_objectness = self.cnn_stem(x_embed, return_maps=True)

        # Step 3: Proposal-guided Slot Attention competitive binding
        if mask is not None:
            flat_mask = mask.reshape(B, H * W) if mask.dim() == 3 else mask
        else:
            flat_mask = None

        slots, objectness, attn_maps = self.slot_attention(
            tokens,
            mask=flat_mask,
            cell_objectness=cell_objectness,
        )

        # Step 4: Relational Set Transformer refinement
        refined_slots = self.relation_encoder(slots, objectness=objectness)

        # Step 5: Symbolic property prediction
        props = self.property_heads(refined_slots)

        # Step 6: Reconstruction decoder back to 2D grid logits
        recon_logits = self.decoder(
            refined_slots,
            H=H,
            W=W,
        )

        return {
            "slots": refined_slots,
            "raw_slots": slots,
            "objectness": objectness,
            "attn_maps": attn_maps,
            "boundary_map": boundary_map,
            "cell_objectness": cell_objectness,
            "props": props,
            "recon_logits": recon_logits,
        }

    def to_world_state(
        self,
        grid: Union[torch.Tensor, np.ndarray],
        obj_threshold: float = 0.4,
        rel_threshold: float = 0.5,
        frame_index: int = 0,
    ) -> WorldState:
        """Translates a raw 2D ARC grid into a complete, structured WorldState dataclass."""
        self.eval()
        device = next(self.parameters()).device

        if isinstance(grid, np.ndarray):
            raw_grid_np = grid.copy()
            grid_t = torch.from_numpy(grid).long().to(device)
        else:
            raw_grid_np = grid.detach().cpu().numpy()
            grid_t = grid.to(device)

        if grid_t.dim() == 2:
            grid_t = grid_t.unsqueeze(0)

        H, W = raw_grid_np.shape[-2], raw_grid_np.shape[-1]
        with torch.no_grad():
            out = self.forward(grid_t)

        slots = out["slots"][0].cpu().numpy()                       # (K, D)
        objectness = out["objectness"][0].cpu().numpy()             # (K,)
        props = out["props"]
        colors = props["color"][0].argmax(dim=-1).cpu().numpy()     # (K,)
        positions = props["position"][0].cpu().numpy()              # (K, 2)
        sizes = props["size"][0].cpu().numpy().reshape(-1)          # (K,)
        shapes = props["shape"][0].argmax(dim=-1).cpu().numpy()     # (K,)
        orientations = props["orientation"][0].argmax(dim=-1).cpu().numpy() # (K,)
        symmetries = (torch.sigmoid(props["symmetry"][0]) >= 0.5).cpu().numpy() # (K, 4)
        attn_maps = out["attn_maps"][0].cpu().numpy().reshape(self.n_slots, H, W) # (K, H, W)

        # Construct StructuredObject list for active slots
        objects: List[StructuredObject] = []
        arc_objects: List[ArcObject] = []

        for k in range(self.n_slots):
            conf = float(objectness[k])
            if conf < obj_threshold:
                continue

            # Compute exact bounding box and pixels from slot spatial attention map
            slot_attn = attn_maps[k]
            peak = float(slot_attn.max()) if slot_attn.size > 0 else 0.0
            thresh = max(peak * 0.4, 0.1)
            slot_mask = slot_attn >= thresh

            active_pixels = np.argwhere(slot_mask)
            if len(active_pixels) == 0:
                # Fallback to centroid point
                r_c = int(np.clip(positions[k, 0] * (H - 1), 0, H - 1))
                c_c = int(np.clip(positions[k, 1] * (W - 1), 0, W - 1))
                active_pixels = np.array([[r_c, c_c]])

            r_min = float(active_pixels[:, 0].min() / max(H - 1, 1))
            r_max = float(active_pixels[:, 0].max() / max(H - 1, 1))
            c_min = float(active_pixels[:, 1].min() / max(W - 1, 1))
            c_max = float(active_pixels[:, 1].max() / max(W - 1, 1))
            width = float((active_pixels[:, 1].max() - active_pixels[:, 1].min() + 1) / max(W, 1))
            height = float((active_pixels[:, 0].max() - active_pixels[:, 0].min() + 1) / max(H, 1))
            area = float(len(active_pixels) / max(H * W, 1))
            perimeter = float(2 * (width + height))
            aspect_ratio = float(width / max(height, 1e-4))

            obj = StructuredObject(
                slot_id=k,
                color=int(colors[k]),
                confidence=conf,
                centroid=(float(positions[k, 0]), float(positions[k, 1])),
                bbox=(r_min, c_min, r_max, c_max),
                width=width,
                height=height,
                area=area,
                perimeter=perimeter,
                aspect_ratio=aspect_ratio,
                shape_class=int(shapes[k]),
                orientation=int(orientations[k]),
                symmetries=(bool(symmetries[k, 0]), bool(symmetries[k, 1]), bool(symmetries[k, 2]), bool(symmetries[k, 3])),
                has_holes=False,
                mask=slot_mask,
                identity_vector=slots[k, :64],
            )
            objects.append(obj)

            # Construct ArcObject for exact relational graph computation
            arc_obj = ArcObject(
                color=int(colors[k]),
                pixels=active_pixels,
            )
            arc_objects.append(arc_obj)

        # Compute exact 14-predicate Relational Graph from detected objects
        gt_rel_mat = extract_ground_truth_relations(arc_objects, H=H, W=W)
        from cir_arc.neural.world_state import RELATION_TYPES

        active_relations: List[SpatialRelation] = []
        for i in range(gt_rel_mat.shape[0]):
            for j in range(gt_rel_mat.shape[1]):
                if i == j:
                    continue
                for r_idx, rel_name in enumerate(RELATION_TYPES):
                    conf = float(gt_rel_mat[i, j, r_idx])
                    if conf >= rel_threshold:
                        src_id = objects[i].slot_id if i < len(objects) else i
                        tgt_id = objects[j].slot_id if j < len(objects) else j
                        active_relations.append(
                            SpatialRelation(
                                source_id=src_id,
                                relation_type=rel_name,
                                target_id=tgt_id,
                                confidence=conf,
                            )
                        )

        relation_graph = RelationGraph(
            adj_matrix=gt_rel_mat,
            edge_list=active_relations,
        )

        return WorldState(
            objects=objects,
            relations=active_relations,
            relation_graph=relation_graph,
            raw_grid=raw_grid_np,
            grid_shape=(H, W),
            frame_index=frame_index,
        )


class Trainer:
    """Multi-objective training manager for PerceptionModel."""

    def __init__(
        self,
        model: PerceptionModel,
        config: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        clip_grad_norm: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.config = config or {}

        train_cfg = self.config.get("training", {})
        self.lr = float(lr if lr is not None else train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.clip_grad_norm = float(clip_grad_norm if clip_grad_norm is not None else train_cfg.get("clip_grad_norm", 1.0))

        device_setting = device if device is not None else train_cfg.get("device")
        if isinstance(device_setting, torch.device):
            self.device = device_setting
        elif device_setting == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_setting == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        weights_cfg = dict(self.config.get("loss_weights", {}))
        if loss_weights is not None:
            weights_cfg.update(loss_weights)

        self.recon_weight = float(weights_cfg.get("recon_weight", 1.5))
        self.color_weight = float(weights_cfg.get("color_weight", 1.0))
        self.pos_weight = float(weights_cfg.get("pos_weight", 1.0))
        self.size_weight = float(weights_cfg.get("size_weight", 0.5))
        self.shape_weight = float(weights_cfg.get("shape_weight", 0.5))
        self.obj_weight = float(weights_cfg.get("obj_weight", 1.0))
        self.bound_weight = float(weights_cfg.get("bound_weight", 0.5))
        self.cell_obj_weight = float(weights_cfg.get("cell_obj_weight", 0.5))
        self.div_weight = float(weights_cfg.get("div_weight", 0.01))
        self.sparse_weight = float(weights_cfg.get("sparse_weight", 0.01))

        total_epochs = int(self.config.get("training", {}).get("epochs", 30))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs, 1),
            eta_min=1e-5,
        )

        self.step = 0
        self.epoch = 0

    def step_scheduler(self) -> float:
        """Advance LR scheduler by one epoch and return current learning rate."""
        self.scheduler.step()
        self.epoch += 1
        return self.get_current_lr()

    def get_current_lr(self) -> float:
        """Return the current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single forward and backward optimization step.

        Args:
            batch: Dictionary containing input_grids, input_masks, gt_objects, boundary_targets, etc.

        Returns:
            Dict of float loss metrics.
        """
        grids = batch["input_grids"] if "input_grids" in batch else batch["grid"]
        masks = batch.get("input_masks") if "input_masks" in batch else batch.get("mask")
        gt_objects = batch.get("gt_objects") if "gt_objects" in batch else batch.get("objects", [])
        heights = batch.get("heights") if "heights" in batch else batch.get("H", [grids.shape[1]] * grids.shape[0])
        widths = batch.get("widths") if "widths" in batch else batch.get("W", [grids.shape[2]] * grids.shape[0])
        boundary_targets = batch.get("boundary_targets") if "boundary_targets" in batch else batch.get("boundary_map")
        objectness_targets = batch.get("objectness_targets") if "objectness_targets" in batch else batch.get("objectness_map")

        grids = grids.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        if boundary_targets is not None:
            boundary_targets = boundary_targets.to(self.device)
        if objectness_targets is not None:
            objectness_targets = objectness_targets.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(grids, mask=masks)
        slots = outputs["slots"]
        objectness = outputs["objectness"]
        props = outputs["props"]
        recon_logits = outputs["recon_logits"]
        pred_boundary = outputs["boundary_map"]
        pred_cell_obj = outputs["cell_objectness"]

        # 1. Grid Reconstruction Loss
        loss_recon = reconstruction_loss(recon_logits, grids, mask=masks)

        # 2. Hungarian Matching for Object Supervision
        B = grids.shape[0]
        matches_batch: List[List[Tuple[int, int]]] = []

        for b in range(B):
            sample_objs = gt_objects[b] if b < len(gt_objects) else []
            sample_props = {k: v[b] for k, v in props.items()}
            sample_h = heights[b] if b < len(heights) else grids.shape[1]
            sample_w = widths[b] if b < len(widths) else grids.shape[2]

            matches = hungarian_matching(
                pred_props=sample_props,
                gt_objects=sample_objs,
                H=sample_h,
                W=sample_w,
            )
            matches_batch.append(matches)

        # 3. Property Losses
        loss_color = color_loss(props["color"], gt_objects, matches_batch)
        loss_pos = position_loss(props["position"], gt_objects, matches_batch, H=heights, W=widths)
        loss_size = size_loss(props["size"], gt_objects, matches_batch, H=heights, W=widths)
        loss_shape = shape_loss(props["shape"], gt_objects, matches_batch)
        loss_obj = objectness_loss(objectness, matches_batch)

        # 4. Diversity & Sparsity Regularizations
        loss_div = diversity_loss(slots)
        loss_sparse = objectness_sparsity_loss(objectness)

        # 5. Boundary & Cell Objectness Losses
        if boundary_targets is not None:
            loss_bound = boundary_loss(pred_boundary, boundary_targets, mask=masks)
        else:
            loss_bound = torch.tensor(0.0, device=self.device)

        if objectness_targets is not None:
            loss_cell_obj = cell_objectness_loss(pred_cell_obj, objectness_targets, mask=masks)
        else:
            loss_cell_obj = torch.tensor(0.0, device=self.device)

        # Total Weighted Multi-Objective Loss
        total_loss = (
            self.recon_weight * loss_recon
            + self.color_weight * loss_color
            + self.pos_weight * loss_pos
            + self.size_weight * loss_size
            + self.shape_weight * loss_shape
            + self.obj_weight * loss_obj
            + self.bound_weight * loss_bound
            + self.cell_obj_weight * loss_cell_obj
            + self.div_weight * loss_div
            + self.sparse_weight * loss_sparse
        )

        total_loss.backward()

        if self.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        self.step += 1

        return {
            "loss": float(total_loss.item()),
            "loss_recon": float(loss_recon.item()),
            "recon_loss": float(loss_recon.item()),
            "loss_color": float(loss_color.item()),
            "color_loss": float(loss_color.item()),
            "loss_pos": float(loss_pos.item()),
            "pos_loss": float(loss_pos.item()),
            "loss_size": float(loss_size.item()),
            "size_loss": float(loss_size.item()),
            "loss_shape": float(loss_shape.item()),
            "shape_loss": float(loss_shape.item()),
            "loss_obj": float(loss_obj.item()),
            "obj_loss": float(loss_obj.item()),
            "loss_bound": float(loss_bound.item()),
            "loss_cell_obj": float(loss_cell_obj.item()),
            "loss_div": float(loss_div.item()),
            "loss_sparse": float(loss_sparse.item()),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model weights, optimizer state, and trainer metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model weights and optimizer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
