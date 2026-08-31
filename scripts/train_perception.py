"""
Training and Evaluation runner for Phase 2 Neural Perception on ARC tasks.
Usage:
    python scripts/train_perception.py --config configs/phase2.yaml --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cir_arc.neural.training.dataset import SyntheticArcDataset, collate_variable_grids
from cir_arc.neural.training.trainer import PerceptionModel, Trainer
from cir_arc.neural.evaluation.perception_metrics import compute_perception_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIR-ARC Perception Model")
    parser.add_argument("--config", type=str, default="configs/phase2.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--data_dir", type=str, default="data/synthetic/train", help="Train data directory")
    parser.add_argument("--held_out_dir", type=str, default="data/synthetic/held_out", help="Validation data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phase2", help="Directory to save checkpoints")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate(model: PerceptionModel, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_metrics: Dict[str, list] = {}

    with torch.no_grad():
        for batch in val_loader:
            grids = batch["input_grids"].to(device)
            masks = batch["input_masks"].to(device)
            gt_objects = batch["gt_objects"]
            heights = batch["heights"]
            widths = batch["widths"]

            outputs = model(grids, mask=masks)
            metrics = compute_perception_metrics(
                pred_logits=outputs["recon_logits"],
                target_grid=grids,
                objectness=outputs["objectness"],
                pred_props=outputs["props"],
                gt_objects_batch=gt_objects,
                mask=masks,
                heights=heights,
                widths=widths,
                pred_masks=outputs.get("slot_masks"),
                pred_relations=outputs.get("relation_logits"),
            )

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(v)

    return {k: float(sum(v) / max(len(v), 1)) for k, v in all_metrics.items()}


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs

    exp_name = args.exp_name or config.get("experiment", {}).get("name", "phase2_perception")
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device_str = config.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"=== CIR-ARC Phase 2 Perception Training [{exp_name}] ===")
    print(f"Target execution device: {device}")

    # Dataset loading
    train_dir = args.data_dir
    held_out_dir = args.held_out_dir

    if not os.path.exists(train_dir):
        print(f"Warning: Train dataset directory '{train_dir}' not found. Generating sample data...")
        from cir_arc.generators.single_rule import GENERATOR_REGISTRY
        os.makedirs(train_dir, exist_ok=True)
        for name, GenCls in GENERATOR_REGISTRY.items():
            gen = GenCls()
            tasks = gen.generate_batch(50, seed=42)
            out_p = Path(train_dir) / name
            out_p.mkdir(parents=True, exist_ok=True)
            for t in tasks:
                t.save(out_p / f"{t.task_id}.json")

    train_ds = SyntheticArcDataset(data_dir=train_dir)
    val_ds = SyntheticArcDataset(data_dir=held_out_dir) if (os.path.exists(held_out_dir) and len(os.listdir(held_out_dir)) > 0) else None

    if val_ds is None or len(val_ds) == 0:
        # Automatically split 20% of train_ds for validation so validation metrics are ALWAYS evaluated
        if len(train_ds) > 10:
            val_size = max(int(0.2 * len(train_ds)), 1)
            train_size = len(train_ds) - val_size
            train_ds, val_ds = torch.utils.data.random_split(
                train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            print(f"Dataset split: {train_size} train, {val_size} validation examples.")
        else:
            print(f"Loaded {len(train_ds)} training examples.")
    else:
        print(f"Loaded {len(train_ds)} training examples.")
        print(f"Loaded {len(val_ds)} validation examples.")

    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("training", {}).get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_grids,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_variable_grids,
            num_workers=num_workers,
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    # Initialize model and trainer
    model_cfg = config.get("model", {})
    model = PerceptionModel(
        num_colors=model_cfg.get("num_colors", 11),
        embed_dim=model_cfg.get("embed_dim", 48),
        hidden_channels=model_cfg.get("stem_hidden_dim", 64),
        stem_out_dim=model_cfg.get("stem_out_dim", 128),
        n_slots=model_cfg.get("n_slots", 24),
        slot_dim=model_cfg.get("slot_dim", 128),
        n_iter=model_cfg.get("n_iter", 3),
        relation_layers=model_cfg.get("relation_layers", 2),
        relation_heads=model_cfg.get("relation_heads", 4),
        prop_hidden_dim=model_cfg.get("prop_hidden_dim", 64),
        num_shapes=model_cfg.get("num_shapes", 8),
        num_orientations=model_cfg.get("num_orientations", 4),
        num_symmetries=model_cfg.get("num_symmetries", 4),
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    trainer = Trainer(model=model, config=config, device=device)

    epochs = config.get("training", {}).get("epochs", 30)
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        trainer.epoch = epoch
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            metrics = trainer.train_step(batch)
            total_loss += metrics.get("loss", 0.0)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        log_str = f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {avg_loss:.4f}"

        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            f1 = val_metrics.get("object_f1", 0.0)
            recon_acc = val_metrics.get("recon_acc", 0.0)
            col_acc = val_metrics.get("color_acc", 0.0)
            pos_mae = val_metrics.get("pos_mae", 0.0)
            mask_iou_v = val_metrics.get("mask_iou", 0.0)
            rel_acc_v = val_metrics.get("relation_acc", 0.0)
            log_str += f" | Val F1: {f1:.3f} | Recon: {recon_acc:.3f} | Mask IoU: {mask_iou_v:.3f} | Rel Acc: {rel_acc_v:.3f} | Pos MAE: {pos_mae:.3f}"

            if f1 > best_f1:
                best_f1 = f1
                best_path = checkpoint_dir / "best_model.pt"
                trainer.save_checkpoint(str(best_path))

        current_lr = trainer.step_scheduler()
        log_str += f" | LR: {current_lr:.5f}"

        print(log_str)

        # Periodic save
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            trainer.save_checkpoint(str(ckpt_path))

    print(f"\nTraining Complete! Best Object F1: {best_f1:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
