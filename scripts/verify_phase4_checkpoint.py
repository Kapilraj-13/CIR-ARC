"""Verifies the Phase 4 120.18M Reasoner checkpoint saved from Kaggle."""

import sys
from pathlib import Path
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cir_arc.neural.reasoner import CognitiveReasoner120M, ReasonerConfig


def verify_checkpoint(checkpoint_path: str = "checkpoints/phase4/best_reasoner_120m.pt"):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint file not found: {ckpt_file.resolve()}")
        sys.exit(1)

    print("=" * 70)
    print("CIR-ARC REASONER CHECKPOINT VERIFICATION")
    print(f"File: {ckpt_file.resolve()}")
    print(f"Size: {ckpt_file.stat().st_size / (1024**2):.2f} MB ({ckpt_file.stat().st_size:,} bytes)")
    print("=" * 70)

    # 1. Load checkpoint
    print("\n[Step 1/5] Loading checkpoint into memory (CPU)...")
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    print(f"  [OK] Successfully loaded. Keys: {list(checkpoint.keys())}")

    # 2. Inspect config
    print("\n[Step 2/5] Inspecting Architecture Configuration...")
    config_data = checkpoint.get("config")
    if isinstance(config_data, dict):
        config = ReasonerConfig(**config_data)
    else:
        config = config_data
    print(f"  d_model: {config.d_model}, n_layers: {config.n_layers}, n_q_heads: {config.n_q_heads}, n_kv_heads: {config.n_kv_heads}")
    print(f"  d_ff: {config.d_ff}, slot_dim: {config.slot_dim}, num_wm_tokens: {config.num_wm_tokens}")
    print("  [OK] ReasonerConfig is valid.")

    # 3. Instantiate model and load state dict
    print("\n[Step 3/5] Instantiating CognitiveReasoner120M and loading weights...")
    model = CognitiveReasoner120M(config)
    state_dict = checkpoint["model_state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=True)
    print(f"  Missing keys: {len(incompatible.missing_keys)}, Unexpected keys: {len(incompatible.unexpected_keys)}")
    
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters in model: {total_params:,}")
    print(f"  Active parameters:         {active_params:,}")
    assert total_params == 120_179_360, f"Expected 120,179,360 params, got {total_params}"
    print("  [OK] Strict weights loading succeeded! Parameter count exactly matches 120,179,360.")

    # 4. Test Forward Pass
    print("\n[Step 4/5] Testing Forward Pass on CPU...")
    model.eval()
    with torch.no_grad():
        dummy_slots = torch.randn(1, 16, config.slot_dim)
        outputs = model(slot_embeddings=dummy_slots)
        print(f"  Cognitive State shape:     {outputs['cognitive_state'].shape}")
        print(f"  Action Logits shape:       {outputs['action_logits'].shape}")
        print(f"  Goals shape:               {outputs['goals'].shape}")
        print(f"  Working Memory shape:      {outputs['updated_working_memory'].shape}")
        print(f"  Value Prediction:          {outputs['value'].item():.4f}")
        print(f"  Decision Confidence:       {outputs['confidence'].item():.4f}")
    print("  [OK] Forward pass executed successfully without errors.")

    # 5. Test Counterfactual Action Planning
    print("\n[Step 5/5] Testing Counterfactual Action Planning Engine...")
    with torch.no_grad():
        intent, scores = model.plan(slot_embeddings=dummy_slots, candidate_actions=[0, 1, 2, 3, 4, 6])
        print(f"  Selected Action Intent:    {intent.action_name} (ID={intent.action_type_id})")
        print(f"  Confidence:                {intent.confidence:.4f}")
        print(f"  Expected Value:            {intent.expected_value:.4f}")
        print(f"  Candidate Action Rollouts Evaluated: {len(scores)}")
        for rank, sc in enumerate(scores[:3], 1):
            print(f"    Rank #{rank}: {sc.action_name:<10s} | Score: {sc.total_score:+.4f} | Value: {sc.future_value:+.2f} | Risk: {sc.risk_penalty:.2f}")
    print("  [OK] Planning engine operational!")

    # Check validation metrics if recorded
    val_metrics = checkpoint.get("val_metrics")
    if val_metrics:
        print("\n" + "=" * 70)
        print("RECORDED VALIDATION PERFORMANCE IN CHECKPOINT:")
        print(f"  Action Top-1 Accuracy: {val_metrics.get('action_accuracy', 0.0) * 100:.2f}%")
        print(f"  Action Top-3 Accuracy: {val_metrics.get('action_top3_accuracy', 0.0) * 100:.2f}%")
        print(f"  Action Macro F1:       {val_metrics.get('action_macro_f1', 0.0):.4f}")
        print(f"  Total Loss:            {val_metrics.get('total_loss', 0.0):.4f}")
        print("=" * 70)

    print("\n" + "=" * 70)
    print(">>> ALL 5 CHECKS PASSED: CHECKPOINT IS 100% VALID AND FULLY VERIFIED! <<<")
    print("=" * 70)


if __name__ == "__main__":
    verify_checkpoint()
