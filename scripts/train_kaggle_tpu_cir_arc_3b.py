#!/usr/bin/env python3
"""Production Distributed Kaggle TPU Training Script for CIR-ARC-3B (3.000B Parameters).

Target Hardware: Kaggle TPU VM v3-8 (8 cores, 128 GB HBM, 16 GB per core).
Distributed Runtime: PyTorch-XLA (torch_xla) with PJRT_DEVICE=TPU.

Memory Optimization Invariants:
1. Native bfloat16 on Matrix Units (MXUs) via XLA_USE_BF16=1.
2. Gradient Checkpointing on the 24-Layer GQA-SwiGLU Reasoning Trunk.
3. Memory-Efficient Adafactor Optimizer (factored second moments) or FSDP sharding
   to keep optimizer states < 4 GB per core, avoiding OOM on 16 GB HBM cores.
4. Distributed SPMD ParallelLoader over compressed shards (data/shards/train_shard_*.jsonl.gz).
5. Multi-task loss suite balancing Action Policy, Safety Gate, Dynamics, and Novelty.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Configure runtime environment
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# PyTorch-XLA conditional imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from cir_arc.neural.models.cir_arc_3b.config import CirArc3BConfig
from cir_arc.neural.models.cir_arc_3b.model import CirArc3B

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TPU-Trainer-3B")


@dataclass
class TPUTrainingConfig:
    """Hyperparameters and distributed configuration for Kaggle TPU v3-8."""

    # Architecture
    d_model: int = 2560
    n_layers: int = 24
    n_q_heads: int = 20
    n_kv_heads: int = 5
    head_dim: int = 128
    d_ff: int = 6912
    num_slots: int = 64
    slot_dim: int = 1024
    action_space_size: int = 4102

    # Distributed Training
    num_cores: int = 8
    batch_size_per_core: int = 2           # 2 * 8 = 16 per forward step
    gradient_accumulation_steps: int = 8   # Effective batch size = 128
    num_epochs: int = 10
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True    # Vital for 3B parameter model
    use_adafactor: bool = True             # Factored optimizer state (< 4 GB)
    use_fsdp: bool = True                  # Shards model across 8 TPU cores

    # Paths
    shards_dir: str = "data/shards"
    checkpoint_dir: str = "checkpoints/cir_arc_3b"
    save_interval_steps: int = 500
    max_samples_per_shard_scan: int = 100_000


class ShardedTransitionDataset(Dataset):
    """Reads multi-source training transitions from compressed shards or sublevels.jsonl."""

    def __init__(self, shards_dir: str, split: str = "train", max_samples: int = 100_000) -> None:
        self.records: List[Dict[str, Any]] = []

        # Candidate paths prioritizing user dataset
        search_dirs = [
            shards_dir,
            "/kaggle/input/datasets/kapilrajr/CIR ARC-MODEL-DATA",
            "/kaggle/input/CIR ARC-MODEL-DATA",
            "/kaggle/input/datasets/kapilrajr/cir-arc-model-data",
            "/kaggle/input/cir-arc-model-data",
            "data/procedural_sublevels",
            "data/shards",
        ]
        search_dirs = [d for d in search_dirs if d and os.path.exists(d)]

        candidate_files = []
        for sdir in search_dirs:
            # Check for direct sublevels files (e.g. train_sublevels.jsonl)
            target_jsonl = os.path.join(sdir, f"{split}_sublevels.jsonl")
            if os.path.exists(target_jsonl):
                candidate_files.append(target_jsonl)
            # Check for sharded files
            candidate_files.extend(glob.glob(os.path.join(sdir, f"*{split}*shard_*.jsonl.gz")))

        candidate_files = sorted(list(set(candidate_files)))

        if candidate_files:
            logger.info("Loading '%s' data from %d file(s):", split, len(candidate_files))
            for cfile in candidate_files:
                sz_mb = os.path.getsize(cfile) / 1e6
                logger.info("  --> %s (%.2f MB)", os.path.basename(cfile), sz_mb)
                opener = gzip.open(cfile, "rt", encoding="utf-8") if cfile.endswith(".gz") else open(cfile, "r", encoding="utf-8")
                with opener as f:
                    for line in f:
                        if len(self.records) >= max_samples:
                            break
                        line = line.strip()
                        if line:
                            try:
                                self.records.append(json.loads(line))
                            except Exception:
                                continue
                if len(self.records) >= max_samples:
                    break
            logger.info("Successfully loaded %d samples for split '%s'.", len(self.records), split)
        else:
            logger.warning("No files found for split '%s'. Generating in-memory fallback transitions.", split)
            for i in range(200):
                self.records.append({
                    "grid": [[0] * 16 for _ in range(16)],
                    "next_grid": [[0] * 16 for _ in range(16)],
                    "action": i % 8,
                    "is_negative_example": False,
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        gt = rec.get("grid") if "grid" in rec else rec.get("grid_t", [[0] * 16 for _ in range(16)])
        gn = rec.get("next_grid") if "next_grid" in rec else rec.get("grid_next", gt)

        tensor_t = torch.zeros(11, 32, 32, dtype=torch.float32)
        tensor_n = torch.zeros(11, 32, 32, dtype=torch.float32)

        ht = min(len(gt), 32)
        wt = min(len(gt[0]) if ht > 0 else 0, 32)
        for r in range(ht):
            for c in range(wt):
                color = int(gt[r][c]) % 10
                tensor_t[color, r, c] = 1.0

        hn = min(len(gn), 32)
        wn = min(len(gn[0]) if hn > 0 else 0, 32)
        for r in range(hn):
            for c in range(wn):
                color = int(gn[r][c]) % 10
                tensor_n[color, r, c] = 1.0

        action = int(rec.get("action", 0)) % 4102
        is_neg = 1.0 if rec.get("is_negative_example", False) else 0.0

        return {
            "grid_t": tensor_t,
            "grid_next": tensor_n,
            "action": torch.tensor(action, dtype=torch.long),
            "is_negative": torch.tensor(is_neg, dtype=torch.float32),
        }


def compute_cir_arc_multitask_loss(
    model_out: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    config: TPUTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-task loss balancing policy, dynamics, causal attribution, and safety."""
    policy_logits = model_out["policy_logits"]
    target_action = batch["action"].to(policy_logits.device)
    l_policy = F.cross_entropy(policy_logits, target_action)

    entrapment_pred = model_out["entrapment_risk"].squeeze(-1)
    target_risk = batch["is_negative"].to(entrapment_pred.device)
    l_safety = F.binary_cross_entropy(entrapment_pred, target_risk)

    predicted_state = model_out["world_model"]["predicted_state"]
    actual_state = model_out["cognitive_state"].detach()
    l_dynamics = F.mse_loss(predicted_state, actual_state)

    novelty_score = model_out["hypothesis"]["novelty_score"]
    l_novelty = (novelty_score ** 2).mean() * 0.05

    total_loss = l_policy + 0.5 * l_safety + 0.25 * l_dynamics + l_novelty

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_policy": float(l_policy.item()),
        "loss_safety": float(l_safety.item()),
        "loss_dynamics": float(l_dynamics.item()),
    }
    return total_loss, metrics


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_worker(index: int, cfg: TPUTrainingConfig) -> None:
    """Worker executed on each TPU core or fallback device."""
    device = xm.xla_device() if HAS_XLA else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_master = (xm.is_master_ordinal() if HAS_XLA else (index == 0))

    if is_master:
        logger.info("[Core %d] Master worker initialized on device: %s", index, device)

    # Initialize model
    model_config = CirArc3BConfig()
    model = CirArc3B(model_config).to(device)
    if HAS_XLA:
        model = model.to(torch.bfloat16)

    # Optional FSDP sharding
    if HAS_XLA and cfg.use_fsdp:
        try:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
            model = FSDP(model)
            if is_master:
                logger.info("FSDP sharding enabled across 8 TPU cores.")
        except Exception as e:
            if is_master:
                logger.warning("FSDP init notice: %s", e)

    # Datasets and Distributed Loaders
    train_dataset = ShardedTransitionDataset(cfg.shards_dir, split="train", max_samples=cfg.max_samples_per_shard_scan)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size() if HAS_XLA else 1,
        rank=xm.get_ordinal() if HAS_XLA else 0,
        shuffle=True,
    )
    raw_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_per_core,
        sampler=sampler,
        num_workers=0,
    )
    loader = pl.ParallelLoader(raw_loader, [device]).per_device_loader(device) if HAS_XLA else raw_loader

    # Memory-efficient Optimizer Selection
    if cfg.use_adafactor:
        try:
            from torch.optim import Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=cfg.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=cfg.weight_decay,
            )
            if is_master:
                logger.info("Using memory-efficient Adafactor optimizer (< 4 GB state).")
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = (len(train_dataset) // (cfg.batch_size_per_core * cfg.num_cores)) * cfg.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, max(1, total_steps))

    global_step = 0
    model.train()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        if is_master:
            logger.info("=== Epoch %d/%d Starting ===", epoch + 1, cfg.num_epochs)
        epoch_start = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0

        for step, batch in enumerate(loader):
            grid_t = batch["grid_t"].to(device)
            grid_next = batch["grid_next"].to(device)
            action = batch["action"].to(device)

            if HAS_XLA:
                grid_t = grid_t.to(torch.bfloat16)
                grid_next = grid_next.to(torch.bfloat16)

            out = model(
                grid_t=grid_t,
                grid_next=grid_next,
                action=action,
                gradient_checkpointing=cfg.gradient_checkpointing,
            )

            loss, metrics = compute_cir_arc_multitask_loss(out, batch, cfg)
            scaled_loss = loss / cfg.gradient_accumulation_steps
            scaled_loss.backward()
            accum_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if HAS_XLA:
                    xm.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_master and global_step % 50 == 0:
                    lr_curr = scheduler.get_last_lr()[0]
                    logger.info(
                        "Step %05d | Total Loss: %.4f | Policy: %.4f | Safety: %.4f | LR: %.6f",
                        global_step,
                        metrics["loss_total"],
                        metrics["loss_policy"],
                        metrics["loss_safety"],
                        lr_curr,
                    )

                if is_master and global_step % cfg.save_interval_steps == 0:
                    ckpt_file = os.path.join(cfg.checkpoint_dir, f"cir_arc_3b_step{global_step}.pt")
                    if HAS_XLA:
                        xm.save(model.state_dict(), ckpt_file)
                    else:
                        torch.save(model.state_dict(), ckpt_file)
                    logger.info("Saved periodic checkpoint to %s", ckpt_file)

        epoch_time = time.time() - epoch_start
        if is_master:
            logger.info("Epoch %d completed in %.2fs", epoch + 1, epoch_time)

    if is_master:
        final_ckpt = os.path.join(cfg.checkpoint_dir, "cir_arc_3b_final.pt")
        if HAS_XLA:
            xm.save(model.state_dict(), final_ckpt)
        else:
            torch.save(model.state_dict(), final_ckpt)
        logger.info("Training complete. Final checkpoint saved to: %s", final_ckpt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Kaggle TPU training harness for CIR-ARC-3B.")
    parser.add_argument("--data-dir", "--shards-dir", dest="data_dir", type=str, default="/kaggle/input/datasets/kapilrajr/cir-arc-model-data", help="Path to data or shards")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/cir_arc_3b", help="Checkpoint output dir")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per TPU core")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    cfg = TPUTrainingConfig(
        shards_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size_per_core=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )

    if HAS_XLA:
        logger.info("Launching distributed training across all TPU cores with xmp.spawn...")
        xmp.spawn(_train_worker, args=(cfg,))
    else:
        logger.warning("torch_xla not detected. Executing local single-process simulation on device.")
        _train_worker(0, cfg)


if __name__ == "__main__":
    main()
