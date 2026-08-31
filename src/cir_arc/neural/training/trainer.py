"""Unified multi-scale perception model and training infrastructure for CIR-ARC Phase 2.5."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import MultiScaleCNNStem, CNNStem
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.relation_encoder import SlotRelationEncoder
from cir_arc.neural.perception.relation_graph import RelationalGraphHead, extract_ground_truth_relations
from cir_arc.neural.perception.property_heads import PropertyHeads
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder, SlotMaskDecoder
from cir_arc.neural.world_state import (
    StructuredObject,
    SpatialRelation,
    RelationGraph,
    WorldState,
    RELATION_TYPES,
    NUM_RELATIONS,
)

from cir_arc.neural.losses.matching import hungarian_matching
from cir_arc.neural.losses.reconstruction import reconstruction_loss
from cir_arc.neural.losses.property import (
    color_loss,
    position_loss,
    size_loss,
    shape_loss,
    orientation_loss,
    symmetry_loss,
    objectness_loss,
    bbox_loss,
    dimensions_loss,
)
from cir_arc.neural.losses.diversity import (
    diversity_loss,
    objectness_sparsity_loss,
)
from cir_arc.neural.losses.boundary import (
    boundary_loss,
    cell_objectness_loss,
)
from cir_arc.neural.losses.mask import (
    slot_mask_loss,
    mask_exclusivity_loss,
)
from cir_arc.neural.losses.relation import relation_loss
from cir_arc.neural.losses.identity import object_identity_contrastive_loss


class PerceptionModel(nn.Module):
    """Unified multi-scale object-centric perception pipeline for ARC grids (~1.0M parameters).

    Chains:
    1. ColorEmbedding: (B, H, W) -> (B, H, W, 48)
    2. MultiScaleCNNStem: (B, H, W, 48) -> tokens (B, H*W, 128), boundary_map (B, 1, H, W), cell_obj (B, 1, H, W)
    3. Proposal SlotAttention: (B, H*W, 128) + cell_obj -> slots (B, 24, 128), objectness (B, 24), attn_maps (B, 24, H*W)
    4. SlotRelationEncoder: slots (B, 24, 128) -> refined_slots (B, 24, 128) via Set Transformer self-attention
    5. RelationalGraphHead: refined_slots -> relation_logits (B, 24, 24, 14)
    6. SlotMaskDecoder: refined_slots (B, 24, 128) -> slot_masks (B, 24, H, W)
    7. PropertyHeads: refined_slots (B, 24, 128) -> props dict (color, bbox, dims, shape, orient, sym, id, presence)
    8. ReconstructionDecoder: refined_slots + slot_masks -> recon_logits (B, H, W, 10)

    Args:
        num_colors: Number of color embedding classes (default: 11).
        embed_dim: Color embedding dimension (default: 48).
        hidden_channels: CNN stem intermediate channels (default: 64).
        stem_out_dim: CNN stem output token dimension (default: 128).
        n_slots: Number of object slots (default: 24).
        slot_dim: Slot representation dimension (default: 128).
        n_iter: Number of slot attention competitive iterations (default: 3).
        relation_layers: Number of Set Transformer attention layers (default: 2).
        relation_heads: Number of attention heads in relation encoder (default: 4).
        max_h: Maximum grid height (default: 30).
        max_w: Maximum grid width (default: 30).
        prop_hidden_dim: Hidden dimension for property heads (default: 64).
        num_shapes: Number of shape categories (default: 8).
        num_orientations: Number of orientation classes (default: 4).
        num_symmetries: Number of symmetry axes (default: 4).
        recon_num_colors: Number of target color classes for decoder (default: 10).
    """

    def __init__(
        self,
        num_colors: int = 11,
        embed_dim: int = 48,
        hidden_channels: int = 64,
        stem_out_dim: int = 128,
        n_slots: int = 24,
        slot_dim: int = 128,
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
        self.embedding = ColorEmbedding(num_colors=num_colors, embed_dim=embed_dim)

        # 2. Multi-Scale Hierarchical CNN with Proposal Heads
        self.cnn_stem = MultiScaleCNNStem(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            out_channels=stem_out_dim,
            num_groups=8,
        )

        # 3. Proposal-guided Slot Attention
        self.slot_attention = SlotAttention(
            n_slots=n_slots,
            slot_dim=slot_dim,
            feat_dim=stem_out_dim,
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

        # 5. Relational Graph Head
        self.relation_head = RelationalGraphHead(
            slot_dim=slot_dim,
            hidden_dim=128,
            num_relations=NUM_RELATIONS,
        )

        # 6. Spatial Slot Mask Decoder
        self.mask_decoder = SlotMaskDecoder(
            slot_dim=slot_dim,
            max_h=max_h,
            max_w=max_w,
            hidden_dim=64,
        )

        # 7. Symbolic Property Prediction Heads
        self.property_heads = PropertyHeads(
            slot_dim=slot_dim,
            hidden_dim=prop_hidden_dim,
            num_colors=10,
            num_shapes=num_shapes,
            num_orientations=num_orientations,
            num_symmetries=num_symmetries,
            identity_dim=64,
        )

        # 8. Spatial Reconstruction Decoder
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
        """Forward pass through full multi-scale perception pipeline.

        Args:
            grid: LongTensor of shape (B, H, W) containing discrete color indices (0-10).
            mask: Optional Tensor of shape (B, H, W) or (B, H*W) with 1.0/True at valid cells.

        Returns:
            Dict containing:
                - "slots": (B, n_slots, slot_dim)
                - "objectness": (B, n_slots)
                - "relation_logits": (B, n_slots, n_slots, 14)
                - "attn_maps": (B, n_slots, H*W)
                - "boundary_map": (B, 1, H, W)
                - "cell_objectness": (B, 1, H, W)
                - "slot_masks": (B, n_slots, H, W)
                - "props": dict with color, shape, size, position, bbox, dimensions, orientation, symmetry, identity, presence
                - "recon_logits": (B, H, W, 10)
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

        # Step 5: Relational Graph head predicting all pairwise relations
        relation_logits = self.relation_head(refined_slots, objectness=objectness)

        # Step 6: Spatial ownership mask decoding per slot
        slot_masks = self.mask_decoder(refined_slots, H=H, W=W)

        # Step 7: Symbolic property prediction
        props = self.property_heads(refined_slots)

        # Step 8: Reconstruction decoder back to 2D grid logits
        recon_logits = self.decoder(
            refined_slots,
            objectness=objectness,
            slot_masks=slot_masks,
            H=H,
            W=W,
        )

        return {
            "slots": refined_slots,
            "raw_slots": slots,
            "objectness": objectness,
            "relation_logits": relation_logits,
            "attn_maps": attn_maps,
            "boundary_map": boundary_map,
            "cell_objectness": cell_objectness,
            "slot_masks": slot_masks,
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
        bboxes = props["bbox"][0].cpu().numpy()                     # (K, 4)
        dims = props["dimensions"][0].cpu().numpy()                 # (K, 4)
        aspects = props["aspect_ratio"][0].cpu().numpy()            # (K, 1)
        shapes = props["shape"][0].argmax(dim=-1).cpu().numpy()     # (K,)
        orientations = props["orientation"][0].argmax(dim=-1).cpu().numpy() # (K,)
        symmetries = (torch.sigmoid(props["symmetry"][0]) >= 0.5).cpu().numpy() # (K, 4)
        holes = (props["holes"][0] >= 0.5).cpu().numpy()            # (K, 1)
        identities = props["identity"][0].cpu().numpy()             # (K, 64)
        masks = out["slot_masks"][0].cpu().numpy()                  # (K, H, W)

        # Construct StructuredObject list for active slots
        objects: List[StructuredObject] = []
        for k in range(self.n_slots):
            conf = float(objectness[k])
            if conf < obj_threshold:
                continue

            obj = StructuredObject(
                slot_id=k,
                color=int(colors[k]),
                confidence=conf,
                centroid=(float(positions[k, 0]), float(positions[k, 1])),
                bbox=(float(bboxes[k, 0]), float(bboxes[k, 1]), float(bboxes[k, 2]), float(bboxes[k, 3])),
                width=float(dims[k, 0]),
                height=float(dims[k, 1]),
                area=float(dims[k, 2]),
                perimeter=float(dims[k, 3]),
                aspect_ratio=float(aspects[k, 0]),
                shape_class=int(shapes[k]),
                orientation=int(orientations[k]),
                symmetries=(bool(symmetries[k, 0]), bool(symmetries[k, 1]), bool(symmetries[k, 2]), bool(symmetries[k, 3])),
                has_holes=bool(holes[k, 0]),
                mask=masks[k] >= 0.5,
                identity_vector=identities[k],
                raw_slot_vector=slots[k],
            )
            objects.append(obj)

        # Extract Relational Graph
        rel_graphs = self.relation_head.predict_graph(
            out["slots"],
            objectness=out["objectness"],
            threshold=rel_threshold,
            obj_threshold=obj_threshold,
        )
        rel_graph = rel_graphs[0]

        boundary_np = out["boundary_map"][0, 0].cpu().numpy() if "boundary_map" in out else None
        cell_obj_np = out["cell_objectness"][0, 0].cpu().numpy() if "cell_objectness" in out else None

        return WorldState(
            objects=objects,
            relations=rel_graph.edge_list,
            relation_graph=rel_graph,
            raw_grid=raw_grid_np if raw_grid_np.ndim == 2 else raw_grid_np[0],
            grid_shape=(H, W),
            frame_index=frame_index,
            boundary_map=boundary_np,
            cell_objectness=cell_obj_np,
            global_features=slots.mean(axis=0),
        )


class Trainer:
    """Trainer encapsulating multi-objective optimization, Hungarian matching, and checkpointing.

    Args:
        model: PerceptionModel instance.
        config: Optional configuration dictionary.
        lr: Learning rate (default: 1e-3).
        weight_decay: Weight decay for AdamW (default: 1e-4).
        clip_grad_norm: Maximum gradient norm for clipping (default: 1.0).
        loss_weights: Optional dictionary of loss component weights.
        device: torch.device or string.
    """

    def __init__(
        self,
        model: PerceptionModel,
        config: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.model = model
        self.config = config or {}
        self.lr = float(self.config.get("training", {}).get("learning_rate", lr))
        self.weight_decay = float(self.config.get("training", {}).get("weight_decay", weight_decay))
        self.clip_grad_norm = float(self.config.get("training", {}).get("clip_grad_norm", clip_grad_norm))

        if device is not None:
            self.device = torch.device(device)
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

        self.recon_weight = float(weights_cfg.get("recon_weight", 1.0))
        self.color_weight = float(weights_cfg.get("color_weight", 1.0))
        self.pos_weight = float(weights_cfg.get("pos_weight", 1.0))
        self.size_weight = float(weights_cfg.get("size_weight", 1.0))
        self.bbox_weight = float(weights_cfg.get("bbox_weight", 0.5))
        self.dims_weight = float(weights_cfg.get("dims_weight", 0.5))
        self.shape_weight = float(weights_cfg.get("shape_weight", 0.5))
        self.obj_weight = float(weights_cfg.get("obj_weight", 1.5))
        self.div_weight = float(weights_cfg.get("div_weight", 0.01))
        self.sparse_weight = float(weights_cfg.get("sparse_weight", 0.01))
        self.bound_weight = float(weights_cfg.get("bound_weight", 0.8))
        self.cell_obj_weight = float(weights_cfg.get("cell_obj_weight", 0.8))
        self.mask_weight = float(weights_cfg.get("mask_weight", 1.0))
        self.excl_weight = float(weights_cfg.get("excl_weight", 0.05))
        self.relation_weight = float(weights_cfg.get("relation_weight", 0.5))
        self.identity_weight = float(weights_cfg.get("identity_weight", 0.1))

        total_epochs = int(self.config.get("training", {}).get("epochs", 30))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs, 1),
            eta_min=1e-5,
        )

        self.step = 0
        self.epoch = 0

    def step_scheduler(self) -> float:
        """Advance learning rate scheduler and return current learning rate."""
        self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step across all multi-objective perception losses.

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
        pred_masks = outputs["slot_masks"]
        pred_relations = outputs["relation_logits"]

        # 1. Grid Reconstruction Loss
        loss_recon = reconstruction_loss(recon_logits, grids, mask=masks)

        # 2. Hungarian Matching for Object Supervision
        B = grids.shape[0]
        matches_batch: List[List[Tuple[int, int]]] = []
        gt_relation_targets: List[np.ndarray] = []

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

            # Compute GT relational matrix for sample objects
            gt_rel_mat = extract_ground_truth_relations(sample_objs, H=sample_h, W=sample_w)
            gt_relation_targets.append(gt_rel_mat)

        # 3. Property & Geometry Losses
        loss_color = color_loss(props["color"], gt_objects, matches_batch)
        loss_pos = position_loss(props["position"], gt_objects, matches_batch, H=heights, W=widths)
        loss_size = size_loss(props["size"], gt_objects, matches_batch, H=heights, W=widths)
        loss_shape = shape_loss(props["shape"], gt_objects, matches_batch)
        loss_obj = objectness_loss(objectness, matches_batch)
        loss_bbox = bbox_loss(props["bbox"], gt_objects, matches_batch, H=heights, W=widths)
        loss_dims = dimensions_loss(props["dimensions"], gt_objects, matches_batch, H=heights, W=widths)

        # 4. Relational Graph & Contrastive Object Identity Losses
        loss_relation = relation_loss(pred_relations, gt_relation_targets, matches_batch)
        loss_identity = object_identity_contrastive_loss(props["identity"], matches_batch)

        # 5. Diversity & Sparsity Regularizations
        loss_div = diversity_loss(slots)
        loss_sparse = objectness_sparsity_loss(objectness)

        # 6. Boundary & Cell Objectness Losses
        if boundary_targets is not None:
            loss_bound = boundary_loss(pred_boundary, boundary_targets, mask=masks)
        else:
            loss_bound = torch.tensor(0.0, device=self.device)

        if objectness_targets is not None:
            loss_cell_obj = cell_objectness_loss(pred_cell_obj, objectness_targets, mask=masks)
        else:
            loss_cell_obj = torch.tensor(0.0, device=self.device)

        # 7. Spatial Mask Loss & Exclusivity Loss
        loss_mask = slot_mask_loss(pred_masks, gt_objects, matches_batch)
        loss_excl = mask_exclusivity_loss(pred_masks, objectness=objectness)

        # Total Weighted Multi-Objective Loss
        total_loss = (
            self.recon_weight * loss_recon
            + self.color_weight * loss_color
            + self.pos_weight * loss_pos
            + self.size_weight * loss_size
            + self.bbox_weight * loss_bbox
            + self.dims_weight * loss_dims
            + self.shape_weight * loss_shape
            + self.obj_weight * loss_obj
            + self.relation_weight * loss_relation
            + self.identity_weight * loss_identity
            + self.div_weight * loss_div
            + self.sparse_weight * loss_sparse
            + self.bound_weight * loss_bound
            + self.cell_obj_weight * loss_cell_obj
            + self.mask_weight * loss_mask
            + self.excl_weight * loss_excl
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
            "loss_bbox": float(loss_bbox.item()),
            "loss_dims": float(loss_dims.item()),
            "loss_shape": float(loss_shape.item()),
            "shape_loss": float(loss_shape.item()),
            "loss_obj": float(loss_obj.item()),
            "obj_loss": float(loss_obj.item()),
            "loss_relation": float(loss_relation.item()),
            "loss_identity": float(loss_identity.item()),
            "loss_bound": float(loss_bound.item()),
            "loss_cell_obj": float(loss_cell_obj.item()),
            "loss_mask": float(loss_mask.item()),
            "loss_excl": float(loss_excl.item()),
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
                "step": self.step,
                "epoch": self.epoch,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model weights and trainer state from checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
