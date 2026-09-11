"""Parameter and Tensor Shape Verification Script for CIR-ARC-3B.

Audits parameter counts and tensor shapes offline without running heavy training.
Verifies all 10 cognitive faculties against the audited 3,000,000,000 parameter budget.
"""

from __future__ import annotations

import sys
import torch

from cir_arc.neural.models.cir_arc_3b.config import CirArc3BConfig
from cir_arc.neural.models.cir_arc_3b.model import CirArc3B


def verify_parameter_counts(use_meta: bool = False) -> bool:
    """Audit parameter counts down to individual weights."""
    config = CirArc3BConfig()
    print("=" * 70)
    print(f"CIR-ARC-3B ARCHITECTURAL PARAMETER AUDIT {'(META DEVICE)' if use_meta else ''}")
    print("=" * 70)

    # Instantiate model
    print("Instantiating CIR-ARC-3B architecture...")
    if use_meta and hasattr(torch, "device"):
        with torch.device("meta"):
            model = CirArc3B(config)
    else:
        model = CirArc3B(config)
    counts = model.count_parameters()

    expected = {
        "perception_slot_tracker": 140_000_000,
        "game_model_self_model": 100_000_000,
        "hypothesis_synthesizer": 140_000_000,
        "world_model_ensemble": 320_000_000,
        "causal_program_graph": 130_000_000,
        "belief_mpc_planner": 180_000_000,
        "action_heads_safety_gate": 115_000_000,
        "memory_falsification_engine": 190_000_000,
        "token_interface": 17_641_280,
        "reasoning_trunk": 1_667_358_720,
        "total": 3_000_000_000,
    }

    all_passed = True
    print(f"{'Module Name':<35} | {'Actual Params':<15} | {'Target Params':<15} | {'Status'}")
    print("-" * 75)

    for key, target in expected.items():
        actual = counts[key]
        status = "PASSED" if actual == target else f"FAILED (diff: {actual - target:+,d})"
        if actual != target:
            all_passed = False
        print(f"{key:<35} | {actual:15,d} | {target:15,d} | {status}")

    print("=" * 75)
    if all_passed:
        print("[SUCCESS] All 10 cognitive faculties strictly adhere to the 3.000B parameter budget!")
    else:
        print("[ERROR] Parameter discrepancies detected!")

    return all_passed


def verify_forward_shapes() -> bool:
    """Verify input-output tensor shapes with dummy inputs."""
    print("\n" + "=" * 70)
    print("CIR-ARC-3B FORWARD PASS & TENSOR SHAPE VERIFICATION")
    print("=" * 70)

    config = CirArc3BConfig()
    model = CirArc3B(config)
    model.eval()

    B = 1
    C = config.num_input_colors  # 11
    H = config.grid_size         # 32
    W = config.grid_size         # 32

    dummy_grid_t = torch.randn(B, C, H, W)
    dummy_grid_next = torch.randn(B, C, H, W)
    dummy_action = torch.tensor([1], dtype=torch.long)

    print(f"Input grid_t shape:    {tuple(dummy_grid_t.shape)}")
    print(f"Input grid_next shape: {tuple(dummy_grid_next.shape)}")
    print(f"Input action shape:    {tuple(dummy_action.shape)}")

    with torch.no_grad():
        out = model(grid_t=dummy_grid_t, grid_next=dummy_grid_next, action=dummy_action)

    print("\nOutput Tensors:")
    print(f"  cognitive_state:  {tuple(out['cognitive_state'].shape)} (Expected: {(B, config.d_model)})")
    print(f"  policy_logits:    {tuple(out['policy_logits'].shape)} (Expected: {(B, config.action_space_size)})")
    print(f"  pointer_logits:   {tuple(out['pointer_logits'].shape)} (Expected: {(B, 1024)})")
    print(f"  entrapment_risk:  {tuple(out['entrapment_risk'].shape)} (Expected: {(B, 1)})")

    assert out["cognitive_state"].shape == (B, config.d_model), "Cognitive state shape mismatch"
    assert out["policy_logits"].shape == (B, config.action_space_size), "Policy logits shape mismatch"
    assert out["pointer_logits"].shape == (B, 1024), "Pointer logits shape mismatch"
    assert out["entrapment_risk"].shape == (B, 1), "Entrapment risk shape mismatch"

    print("[SUCCESS] All tensor dimensions and shapes match architectural specifications!")
    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Audit CIR-ARC-3B parameters and tensor shapes.")
    parser.add_argument("--meta", action="store_true", help="Use PyTorch meta device (0 RAM usage, fast)")
    parser.add_argument("--skip-forward", action="store_true", help="Skip forward pass check")
    args = parser.parse_args()

    param_ok = verify_parameter_counts(use_meta=args.meta)
    shape_ok = True
    if not args.skip_forward and not args.meta:
        shape_ok = verify_forward_shapes()
    if not (param_ok and shape_ok):
        sys.exit(1)
    print("\n[VERIFICATION COMPLETE] CIR-ARC-3B is mathematically and structurally verified.")


if __name__ == "__main__":
    main()
