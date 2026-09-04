"""Phase 4: Kaggle Training Blueprint for CIR-ARC ~120.18M Direct Cognitive Reasoner.

IMPORTANT SAFETY INVARIANT:
Local execution of multi-hour training is STRICTLY PROHIBITED.
This script is designed for Kaggle GPU environments (dual T4, P100, V100, or A100).
It auto-selects mixed precision (BF16 on A100/Hopper, FP16 on T4/Turing, FP16/FP32 on P100).
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cir_arc.neural.reasoner import (
    ReasonerConfig,
    CognitiveReasoner120M,
    ReasonerMultiObjectiveLoss,
    ReasonerLossWeights,
)


def get_precision_dtype(precision_setting: str = "auto") -> torch.dtype:
    """Auto-detects the optimal compute precision for the available GPU hardware."""
    if not torch.cuda.is_available():
        return torch.float32

    if precision_setting == "bfloat16":
        return torch.bfloat16
    elif precision_setting == "float16":
        return torch.float16
    elif precision_setting == "float32":
        return torch.float32

    # "auto" mode: select best supported dtype
    if torch.cuda.is_bf16_supported():
        print("[Hardware] A100/Ampere detected: auto-selecting bfloat16.")
        return torch.bfloat16
    else:
        print("[Hardware] T4/P100 detected: auto-selecting float16.")
        return torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIR-ARC Phase 4 Kaggle Reasoner Training")
    parser.add_argument("--kaggle-phase4", action="store_true", help="Explicit confirmation flag for Kaggle execution")
    parser.add_argument("--config", type=str, default="configs/phase4.yaml", help="Path to phase 4 config")
    parser.add_argument("--stage", type=str, default="R1", choices=["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase4")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 validation step without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Safety Guardrail: Prevent local training accidents
    if not args.kaggle_phase4 and not args.dry_run:
        print("\n" + "=" * 80)
        print("[SAFETY GUARD]: Local training is strictly prohibited in this environment.")
        print("This script is intended for Phase 4 on Kaggle with GPU acceleration.")
        print("To run a single verification step locally without training, pass --dry-run.")
        print("=" * 80 + "\n")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_precision_dtype("auto")

    print(f"Initializing CIR-ARC 120.18M Reasoner on {device} (dtype={dtype})...")
    config = ReasonerConfig()
    model = CognitiveReasoner120M(config).to(device=device, dtype=dtype)

    param_counts = model.count_parameters()
    print(f"Total Model Parameters: {param_counts['total']:,} ({param_counts['total'] / 1e6:.2f}M)")

    criterion = ReasonerMultiObjectiveLoss(ReasonerLossWeights())

    if args.dry_run:
        print("[Dry Run] Simulating 1 dummy step to verify memory and backward pass...")
        model.train()
        B, K = 2, 16
        dummy_slots = torch.randn(B, K, config.slot_dim, device=device, dtype=dtype)
        dummy_spatial = torch.randn(B, 30, 30, config.feat_dim, device=device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            outputs = model(slot_embeddings=dummy_slots, spatial_features=dummy_spatial)
            dummy_targets = {
                "target_state_latent": outputs["cognitive_state"].detach(),
                "target_goal_latent": outputs["goals"][:, 0].detach(),
                "target_action_id": torch.zeros(B, dtype=torch.long, device=device),
            }
            loss_dict = criterion(outputs, dummy_targets)

        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"[Dry Run] Successfully verified 1 step! Loss = {loss.item():.4f}")
        return

    print(f"Starting Phase 4 Training Stage {args.stage} on Kaggle...")


if __name__ == "__main__":
    main()
