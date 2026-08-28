"""Unified perception model and training infrastructure for CIR-ARC Phase 2."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import CNNStem
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.property_heads import PropertyHeads
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder

from cir_arc.neural.losses.matching import hungarian_matching
from cir_arc.neural.losses.reconstruction import reconstruction_loss
from cir_arc.neural.losses.property import (
    color_loss,
    position_loss,
    size_loss,
    objectness_loss,
)
from cir_arc.neural.losses.diversity import (
    diversity_loss,
    objectness_sparsity_loss,
)


class PerceptionModel(nn.Module):
    """Unified object-centric perception pipeline for ARC grids.

    Chains:
    1. ColorEmbedding: (B, H, W) -> (B, H, W, 48)
    2. CNNStem: (B, H, W, 48) -> (B, H*W, 128)
    3. SlotAttention: (B, H*W, 128) -> slots (B, 24, 128), objectness (B, 24), attn_maps (B, 24, H*W)
    4. PropertyHeads: slots (B, 24, 128) -> props dict
    5. ReconstructionDecoder: slots (B, 24, 128) -> recon_logits (B, H, W, 10)

    Args:
        num_colors: Number of color embedding classes (default: 11).
        embed_dim: Color embedding dimension (default: 48).
        hidden_channels: CNN stem intermediate channels (default: 64).
        stem_out_dim: CNN stem output token dimension (default: 128).
        n_slots: Number of object slots (default: 24).
        slot_dim: Slot representation dimension (default: 128).
        n_iter: Slot attention iterations (default: 3).
        prop_hidden_dim: Property head MLP hidden dimension (default: 64).
        num_shapes: Number of shape categories (default: 8).
        num_orientations: Number of orientation bins (default: 4).
        num_symmetries: Number of symmetry axes (default: 4).
        max_h: Maximum grid height (default: 30).
        max_w: Maximum grid width (default: 30).
        recon_num_colors: Number of reconstruction color classes (default: 10).
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
        prop_hidden_dim: int = 64,
        num_shapes: int = 8,
        num_orientations: int = 4,
        num_symmetries: int = 4,
        max_h: int = 30,
        max_w: int = 30,
        recon_num_colors: int = 10,
    ) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.stem_out_dim = stem_out_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.max_h = max_h
        self.max_w = max_w

        self.embedding = ColorEmbedding(num_colors=num_colors, embed_dim=embed_dim)
        self.cnn_stem = CNNStem(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            out_channels=stem_out_dim,
            num_groups=8,
        )
        self.slot_attention = SlotAttention(
            n_slots=n_slots,
            slot_dim=slot_dim,
            feat_dim=stem_out_dim,
            n_iter=n_iter,
            eps=1e-8,
            hidden_dim=256,
        )
        self.property_heads = PropertyHeads(
            slot_dim=slot_dim,
            hidden_dim=prop_hidden_dim,
            num_colors=10,
            num_shapes=num_shapes,
            num_orientations=num_orientations,
            num_symmetries=num_symmetries,
        )
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
        """Forward pass through full perception pipeline.

        Args:
            grid: LongTensor of shape (B, H, W) containing discrete color indices (0-10).
            mask: Optional Tensor of shape (B, H, W) or (B, H*W) with 1.0/True at valid cells.

        Returns:
            Dict containing:
                - "slots": (B, n_slots, slot_dim)
                - "objectness": (B, n_slots)
                - "attn_maps": (B, n_slots, H*W)
                - "props": dict with color, shape, size, position, orientation, symmetry
                - "recon_logits": (B, H, W, 10)
        """
        B, H, W = grid.shape

        # Step 1: Color embedding (B, H, W, embed_dim)
        x_embed = self.embedding(grid)

        # Step 2: Spatial CNN encoder (B, H*W, stem_out_dim)
        tokens = self.cnn_stem(x_embed)

        # Step 3: Slot Attention competitive binding
        if mask is not None:
            flat_mask = mask.reshape(B, H * W) if mask.dim() == 3 else mask
        else:
            flat_mask = None

        slots, objectness, attn_maps = self.slot_attention(tokens, mask=flat_mask)

        # Step 4: Symbolic property prediction
        props = self.property_heads(slots)

        # Step 5: Reconstruction decoder back to 2D grid logits
        recon_logits = self.decoder(slots, objectness=objectness, H=H, W=W)

        return {
            "slots": slots,
            "objectness": objectness,
            "attn_maps": attn_maps,
            "props": props,
            "recon_logits": recon_logits,
        }


class Trainer:
    """Trainer encapsulating optimization, loss computation, Hungarian matching, and checkpointing.

    Args:
        model: PerceptionModel instance.
        config: Optional configuration dictionary.
        lr: Learning rate (default: 1e-3).
        weight_decay: Weight decay for AdamW (default: 1e-4).
        clip_grad_norm: Maximum gradient norm for clipping (default: 1.0).
        loss_weights: Optional dictionary of loss component weights.
        device: Target execution device ('cpu', 'cuda', or torch.device).
    """

    def __init__(
        self,
        model: PerceptionModel,
        config: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.config = config or {}

        train_cfg = self.config.get("training", {})
        self.lr = train_cfg.get("learning_rate", lr) if lr == 1e-3 and "learning_rate" in train_cfg else lr
        self.weight_decay = train_cfg.get("weight_decay", weight_decay) if weight_decay == 1e-4 and "weight_decay" in train_cfg else weight_decay
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", clip_grad_norm) if clip_grad_norm == 1.0 and "clip_grad_norm" in train_cfg else clip_grad_norm

        if device is not None:
            self.device = torch.device(device)
        elif "device" in train_cfg:
            cfg_dev = train_cfg["device"]
            if cfg_dev == "cuda" and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(cfg_dev)
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
        self.obj_weight = float(weights_cfg.get("obj_weight", 0.5))
        self.div_weight = float(weights_cfg.get("div_weight", 0.01))
        self.sparse_weight = float(weights_cfg.get("sparse_weight", 0.01))

        self.step = 0
        self.epoch = 0

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training optimization step.

        Computes forward pass, Hungarian matching against ground-truth objects,
        reconstruction, property, diversity, and sparsity losses, backward pass,
        gradient clipping, and optimizer step.

        Args:
            batch: Dictionary containing input_grids/grid, input_masks/mask, gt_objects, etc.

        Returns:
            Dict of float loss metrics.
        """
        grids = batch["input_grids"] if "input_grids" in batch else batch["grid"]
        masks = batch.get("input_masks") if "input_masks" in batch else batch.get("mask")
        gt_objects = batch.get("gt_objects") if "gt_objects" in batch else batch.get("objects", [])
        heights = batch.get("heights") if "heights" in batch else batch.get("H", [grids.shape[1]] * grids.shape[0])
        widths = batch.get("widths") if "widths" in batch else batch.get("W", [grids.shape[2]] * grids.shape[0])

        grids = grids.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(grids, mask=masks)
        slots = out["slots"]
        objectness = out["objectness"]
        props = out["props"]
        recon_logits = out["recon_logits"]

        B = grids.shape[0]

        # Hungarian matching per sample in batch
        batch_matches: List[List[Tuple[int, int]]] = []
        for b in range(B):
            b_props = {k: v[b] for k, v in props.items()}
            b_gt = gt_objects[b] if b < len(gt_objects) else []
            H_b = heights[b] if isinstance(heights, list) else int(heights)
            W_b = widths[b] if isinstance(widths, list) else int(widths)
            matches_b = hungarian_matching(b_props, b_gt, H=H_b, W=W_b)
            batch_matches.append(matches_b)

        # Losses
        l_recon = reconstruction_loss(recon_logits, grids, mask=masks)
        c_loss = color_loss(props["color"], gt_objects, batch_matches)
        p_loss = position_loss(props["position"], gt_objects, batch_matches, H=heights, W=widths)
        s_loss = size_loss(props["size"], gt_objects, batch_matches, H=heights, W=widths)
        o_loss = objectness_loss(objectness, batch_matches)
        l_div = diversity_loss(slots)
        l_sparse = objectness_sparsity_loss(objectness)

        total_loss = (
            self.recon_weight * l_recon
            + self.color_weight * c_loss
            + self.pos_weight * p_loss
            + self.size_weight * s_loss
            + self.obj_weight * o_loss
            + self.div_weight * l_div
            + self.sparse_weight * l_sparse
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()
        self.step += 1

        return {
            "loss": float(total_loss.item()),
            "recon_loss": float(l_recon.item()),
            "color_loss": float(c_loss.item()),
            "pos_loss": float(p_loss.item()),
            "size_loss": float(s_loss.item()),
            "obj_loss": float(o_loss.item()),
            "div_loss": float(l_div.item()),
            "sparse_loss": float(l_sparse.item()),
        }

    def train_epoch(self, dataloader: Any) -> Dict[str, float]:
        """Train over an entire epoch dataloader.

        Args:
            dataloader: Iterable yielding batch dicts.

        Returns:
            Dict of epoch average float metrics.
        """
        try:
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        except ImportError:
            pbar = dataloader

        epoch_metrics: Dict[str, float] = {}
        count = 0

        for batch in pbar:
            step_metrics = self.train_step(batch)
            for k, v in step_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            count += 1

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": f"{step_metrics['loss']:.4f}"})

        self.epoch += 1
        return {k: v / max(count, 1) for k, v in epoch_metrics.items()}

    def save_checkpoint(self, filepath: str) -> None:
        """Save model state, optimizer state, config, and counters to disk."""
        parent_dir = os.path.dirname(os.path.abspath(filepath))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "step": self.step,
            "epoch": self.epoch,
        }
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model state, optimizer state, and training counters from disk."""
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if "step" in state:
                self.step = state["step"]
            if "epoch" in state:
                self.epoch = state["epoch"]
            if "config" in state and state["config"]:
                self.config = state["config"]
        elif isinstance(state, dict):
            self.model.load_state_dict(state)
        return state


if __name__ == "__main__":
    print("Running PerceptionModel & Trainer smoke tests...")
    model = PerceptionModel()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PerceptionModel parameter count: {param_count}")
    assert 200000 <= param_count <= 500000, f"Parameter count {param_count} outside [200K, 500K]"

    # Uniform forward pass
    B, H, W = 4, 10, 10
    grids = torch.randint(0, 10, (B, H, W), dtype=torch.long)
    out = model(grids)
    assert out["slots"].shape == (B, 24, 128)
    assert out["objectness"].shape == (B, 24)
    assert out["attn_maps"].shape == (B, 24, H * W)
    assert out["recon_logits"].shape == (B, H, W, 10)
    print("Uniform batch forward pass passed.")

    # Heterogeneous batch forward pass
    B_het = 3
    H_max, W_max = 15, 15
    grids_het = torch.randint(0, 10, (B_het, H_max, W_max), dtype=torch.long)
    masks_het = torch.zeros((B_het, H_max, W_max), dtype=torch.float32)
    masks_het[0, :5, :5] = 1.0
    masks_het[1, :8, :12] = 1.0
    masks_het[2, :15, :15] = 1.0
    out_het = model(grids_het, mask=masks_het)
    assert out_het["recon_logits"].shape == (B_het, H_max, W_max, 10)
    print("Heterogeneous batch forward pass passed.")

    # Trainer 1-step test
    trainer = Trainer(model=model, lr=1e-3)
    sample_batch = {
        "input_grids": torch.randint(0, 10, (2, 8, 8), dtype=torch.long),
        "input_masks": torch.ones((2, 8, 8), dtype=torch.float32),
        "gt_objects": [[], []],
        "heights": [8, 8],
        "widths": [8, 8],
    }
    metrics = trainer.train_step(sample_batch)
    assert "loss" in metrics and isinstance(metrics["loss"], float)
    print(f"Trainer step passed: loss={metrics['loss']:.4f}")

    # Checkpoint roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)
        model_loaded = PerceptionModel()
        trainer_loaded = Trainer(model=model_loaded, lr=1e-3)
        trainer_loaded.load_checkpoint(ckpt_path)

        model.eval()
        model_loaded.eval()
        test_in = torch.randint(0, 10, (2, 6, 6), dtype=torch.long)
        with torch.no_grad():
            out1 = model(test_in)
            out2 = model_loaded(test_in)
        assert torch.allclose(out1["recon_logits"], out2["recon_logits"], atol=1e-6)
        print("Checkpoint roundtrip passed.")

    print("All PerceptionModel & Trainer smoke tests passed successfully!")
