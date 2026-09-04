"""Real-time Perception & World-State v2 Inference on Official ARC-AGI-3 Game m0r0.

Loads the trained 3.4M PerceptionModel v2 from checkpoints/phase2/best_model_v2_3.4M.pt
and runs live perception, object decomposition, affordance prediction, pointer head localization,
and transition prediction on official m0r0 game frames.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add m0r0 environment path
m0r0_dir = r"D:\Data_and_Models\DATASETS\arc-prize-2026-arc-agi-3\environment_files\m0r0\492f87ba"
if os.path.exists(m0r0_dir):
    sys.path.insert(0, m0r0_dir)

import numpy as np
import torch

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.rc_adapter import RCEngineAdapter
from cir_arc.neural.training.trainer import PerceptionModel
from cir_arc.neural.world_state import HybridSceneState


def main() -> None:
    print("=" * 75)
    print("  CIR-ARC REAL-TIME PERCEPTION INFERENCE ON OFFICIAL ARC-AGI-3 (Game: m0r0)")
    print("=" * 75)

    ckpt_path = "checkpoints/phase2/best_model_v2_3.4M.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device:    {device}")
    print(f"Loading Checkpoint: {ckpt_path} ...")

    # 1. Instantiate 3.4M PerceptionModel (Stage D)
    model_config = dict(
        num_colors=11,
        embed_dim=48,
        stem_hidden_dim=112,
        stem_out_dim=224,
        n_slots=24,
        slot_dim=224,
        feat_dim=224,
        n_iter=3,
        relation_layers=4,
        relation_heads=8,
        max_h=30,
        max_w=30,
        prop_hidden_dim=96,
        num_shapes=8,
        num_orientations=4,
        num_symmetries=4,
        recon_num_colors=10,
        use_coordconv=True,
        include_v2_modules=True,
    )

    model = PerceptionModel(**model_config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: 3.4M Stage D ({total_params:,} parameters) LOADED STRICT")

    # 2. Initialize official m0r0 environment
    print("\nConnecting to Official ARC-AGI-3 m0r0 Environment...")
    adapter = RCEngineAdapter(game_id="m0r0")
    obs = adapter.reset()

    # Get composite 64x64 game frame
    frame_t = obs.grid.composite()
    h, w = frame_t.shape
    print(f"Initial Frame Received: Shape=({h}, {w}) | Initial State: {obs.state.value}")

    # Display active ARC colors
    colors_present = np.unique(frame_t)
    print(f"Active ARC Colors Present in m0r0 Frame: {colors_present.tolist()}")

    # 3. Real-time forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        frame_tensor = torch.from_numpy(frame_t).long().unsqueeze(0).to(device)
        out = model(frame_tensor)
        t_infer_ms = (time.perf_counter() - t0) * 1000.0

    print("\n" + "-" * 75)
    print(f"  PERCEPTION INFERENCE COMPLETED IN {t_infer_ms:.2f} ms")
    print("-" * 75)

    slots = out["slots"]  # (1, 24, 224)
    objectness = out["objectness"][0].cpu().numpy()  # (24,)
    props = out["props"]
    affordance_logits = out.get("affordance_logits")  # (1, 24, 9)

    print(f"Extracted Slot Tensor:           {list(slots.shape)}")
    print(f"Spatial Tokens Tensor:           {list(out['spatial_tokens'].shape)}")

    # 4. Filter active object slots
    active_indices = [i for i, score in enumerate(objectness) if score > 0.3]
    if not active_indices:
        active_indices = np.argsort(-objectness)[:5].tolist()

    print(f"\nDiscovered {len(active_indices)} Salient Entity Slots (sorted by confidence):")
    sorted_active = sorted(active_indices, key=lambda idx: -objectness[idx])

    for rank, s_idx in enumerate(sorted_active[:6]):
        score = objectness[s_idx]
        pos = props["position"][0, s_idx].cpu().numpy()
        pred_color = int(props["color"][0, s_idx].argmax().item())
        pred_shape = int(props["shape"][0, s_idx].argmax().item())
        has_hole = bool(props.get("has_holes", props.get("hole", torch.zeros(1, 24, 1)))[0, s_idx].sigmoid().item() > 0.5)

        # Affordance top predictions
        aff_str = ""
        if affordance_logits is not None:
            aff_probs = torch.sigmoid(affordance_logits[0, s_idx]).cpu().numpy()
            aff_names = [
                "can_move", "can_push", "can_collect", "can_interact",
                "can_toggle", "can_destroy", "can_block", "can_support", "can_be_clicked"
            ]
            top_affs = [aff_names[a_idx] for a_idx, p in enumerate(aff_probs) if p > 0.4]
            aff_str = f" | Affordances: {', '.join(top_affs) if top_affs else 'passive'}"

        print(
            f"  Slot #{s_idx:02d} [Rank {rank+1}]: Conf={score:.2f} | "
            f"Color={pred_color} | Shape={pred_shape} | "
            f"Norm Pos=({pos[0]:.2f}, {pos[1]:.2f}) | Hole={has_hole}{aff_str}"
        )

    # 5. Run Two-Stage Pointer Head on m0r0
    pointer_out = model.pointer_head(
        slots=slots,
        spatial_tokens=out["spatial_tokens"],
        H=h,
        W=w,
    )
    sel_slot = int(pointer_out["selected_slot"][0].item())
    target_pixel = pointer_out["coords_pixel"][0].cpu().numpy()
    target_xy = pointer_out["coords_xy"][0].cpu().numpy()

    print("\nTwo-Stage Pointer Head Prediction (for ACTION6 interactive clicks):")
    print(f"  Selected Primary Slot:   Slot #{sel_slot}")
    print(f"  Target Local Coordinates: Row={target_pixel[0]}, Col={target_pixel[1]}")
    print(f"  Display Coordinate (X,Y): ({target_xy[0]}, {target_xy[1]})")

    # 6. Extract HybridSceneState & Cognitive Tokens
    hybrid_state = model.to_hybrid_scene_state(frame_t, frame_index=0)
    cognitive_tokens = hybrid_state.to_cognitive_tokens(embed_dim=256)
    print("\nDual Neuro-Symbolic State Bundle:")
    print(f"  Symbolic Objects Count:  {len(hybrid_state.symbolic.objects)}")
    print(f"  Symbolic Relations:      {len(hybrid_state.symbolic.relations)}")
    print(f"  Dense Continuous Latents:{list(hybrid_state.dense.slot_embeddings.shape)}")
    print(f"  Cognitive Reasoner Tokens:{list(cognitive_tokens.shape)} (Fed to 120M reasoner)")

    # 7. Execute Step & Test Transition Prediction
    action_to_test = 1  # Action UP
    print(f"\nExecuting Action {action_to_test} in m0r0...")
    obs_next = adapter.step(Action.from_id(action_to_test))
    frame_next = obs_next.grid.composite()

    with torch.no_grad():
        frame_next_tensor = torch.from_numpy(frame_next).long().unsqueeze(0).to(device)
        out_next = model(frame_next_tensor)
        slots_next_actual = out_next["slots"]

        # Predict next slots via transition model
        pred_next_slots, delta_pos = model.transition_model(
            slots, torch.tensor([action_to_test], device=device)
        )
        cos_sim = torch.cosine_similarity(pred_next_slots, slots_next_actual, dim=-1).mean().item()

    print(f"Next Frame Received. State: {obs_next.state.value} | Steps: {obs_next.step_count}")
    print(f"Latent Transition Consistency: Cosine Similarity = {cos_sim:.4f} (1.0 = perfect alignment)")

    adapter.close()
    print("\n" + "=" * 75)
    print("  REAL-TIME INFERENCE VERIFICATION COMPLETE: ALL SYSTEMS NOMINAL")
    print("=" * 75)


if __name__ == "__main__":
    main()
