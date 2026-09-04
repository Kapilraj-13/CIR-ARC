"""Automated parameter verification and functional tests for the CIR-ARC ~120.18M Reasoner."""

import pytest
import torch

from cir_arc.neural.reasoner import (
    ReasonerConfig,
    CognitiveReasoner120M,
    ActionIntent,
)


def test_exact_parameter_count():
    """Verify that the model matches the audited 120,179,360 parameter specification down to single weights."""
    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    counts = model.count_parameters()

    print("\nAudited Parameter Counts:")
    for k, v in counts.items():
        print(f"  {k:20s}: {v:12,d} ({v / 1e6:.3f}M)")

    # 1. Transformer Trunk: 18 layers GQA (12Q/4KV) + SwiGLU 1856 + RMSNorm
    assert counts["transformer_trunk"] == 105_312_000, f"Expected 105,312,000 but got {counts['transformer_trunk']:,}"

    # 2. Input Fusion: Symbolic encoders + dense projections + gated fusion
    assert counts["input_fusion"] == 5_095_936, f"Expected 5,095,936 but got {counts['input_fusion']:,}"

    # 3. Memory System: 128 reasoning tokens + 128 working memory + cross-attention + retrieval
    assert counts["memory_system"] == 3_544_832, f"Expected 3,544,832 but got {counts['memory_system']:,}"

    # 4. Cognitive Heads: Goal + WorldModel + Value + Action + Verification
    assert counts["cognitive_heads"] == 6_226_592, f"Expected 6,226,592 but got {counts['cognitive_heads']:,}"

    # 5. Grand Total: Exactly 120,179,360 parameters (120.18M)
    assert counts["total"] == 120_179_360, f"Expected 120,179,360 but got {counts['total']:,}"


def test_forward_pass():
    """Verify forward reasoning execution with mock slot and spatial tensors."""
    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    model.eval()

    B = 2
    K = 16
    slot_dim = config.slot_dim      # 224
    feat_dim = config.feat_dim      # 224

    mock_slots = torch.randn(B, K, slot_dim)
    mock_spatial = torch.randn(B, 30, 30, feat_dim)

    with torch.no_grad():
        out = model(slot_embeddings=mock_slots, spatial_features=mock_spatial)

    assert "cognitive_state" in out
    assert out["cognitive_state"].shape == (B, config.d_model)
    assert out["updated_working_memory"].shape == (B, config.num_wm_tokens, config.d_model)
    assert out["goals"].shape == (B, config.num_goal_hypotheses, config.d_model)
    assert out["goal_confidence"].shape == (B, config.num_goal_hypotheses)
    assert out["value"].shape == (B, 1)
    assert out["action_logits"].shape == (B, config.num_action_types)
    assert out["pointer_logits"].shape == (B, K)


def test_counterfactual_planning():
    """Verify candidate action evaluation and ActionIntent emission."""
    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    model.eval()

    B = 1
    K = 8
    mock_slots = torch.randn(B, K, config.slot_dim)
    candidate_actions = [0, 1, 2, 3, 4, 6]  # MOVE_UP, DOWN, LEFT, RIGHT, ACTION, CLICK

    with torch.no_grad():
        intent, scores = model.plan(slot_embeddings=mock_slots, candidate_actions=candidate_actions)

    assert isinstance(intent, ActionIntent)
    assert intent.action_type_id in candidate_actions
    assert len(scores) == len(candidate_actions)
    assert scores[0].total_score >= scores[-1].total_score  # Verified sorted by score


def test_verification_error():
    """Verify state prediction verification and belief revision gating."""
    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    model.eval()

    pred_latent = torch.randn(1, config.d_model)
    obs_latent = torch.randn(1, config.d_model)

    with torch.no_grad():
        error, gate = model.verify_observation(pred_latent, obs_latent)

    assert isinstance(error, float)
    assert error >= 0.0
    assert gate.shape == (1, config.d_model)
    assert (gate >= 0.0).all() and (gate <= 1.0).all()


def test_forward_scene():
    """Verify forward_scene with a complete HybridSceneState dataclass."""
    import numpy as np
    from cir_arc.neural.world_state import (
        HybridSceneState,
        DenseLatentState,
        SymbolicSceneState,
        RelationGraph,
        MechanicsBelief,
        GlobalStateData,
        UncertaintySummary,
    )

    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    model.eval()

    K = 12
    dense = DenseLatentState(
        slot_embeddings=torch.randn(K, config.slot_dim),
        spatial_features=torch.randn(30, 30, config.feat_dim),
    )
    symbolic = SymbolicSceneState(
        frame_index=1,
        grid_shape=(30, 30),
        objects=[],
        relations=[],
        scene_graph=RelationGraph(adj_matrix=np.zeros((K, K, 14))),
        mechanics_beliefs=MechanicsBelief(gravity=(1.0, 0.0), gravity_confidence=0.8),
        global_state=GlobalStateData(score=10, lives=3),
        uncertainties=UncertaintySummary(overall_certainty=0.9),
    )
    scene = HybridSceneState(
        frame_index=1,
        grid_shape=(30, 30),
        raw_grid=np.zeros((30, 30), dtype=np.int32),
        symbolic=symbolic,
        dense=dense,
    )

    with torch.no_grad():
        out = model.forward_scene(scene)

    assert "cognitive_state" in out
    assert out["cognitive_state"].shape == (1, config.d_model)
    assert out["action_logits"].shape == (1, config.num_action_types)
    assert out["goals"].shape == (1, config.num_goal_hypotheses, config.d_model)


def test_multi_objective_loss():
    """Verify ReasonerMultiObjectiveLoss computation across all 9 loss components."""
    from cir_arc.neural.reasoner import ReasonerMultiObjectiveLoss, ReasonerLossWeights

    criterion = ReasonerMultiObjectiveLoss(ReasonerLossWeights())
    B = 2
    d_model = 768

    outputs = {
        "cognitive_state": torch.randn(B, d_model),
        "goals": torch.randn(B, 4, d_model),
        "predicted_next_latent": torch.randn(B, d_model),
        "predicted_event_logits": torch.randn(B, 14),
        "action_logits": torch.randn(B, 7),
        "pointer_logits": torch.randn(B, 12),
        "candidate_scores": torch.randn(B, 6),
        "value": torch.randn(B, 1),
        "plan_latent_sequence": torch.randn(B, 5, d_model),
        "prediction_error": torch.randn(B, 1),
    }

    targets = {
        "target_state_latent": torch.randn(B, d_model),
        "target_goal_latent": torch.randn(B, d_model),
        "target_next_latent": torch.randn(B, d_model),
        "target_event_ids": torch.randint(0, 14, (B,)),
        "target_action_id": torch.randint(0, 7, (B,)),
        "target_pointer_slot": torch.randint(0, 12, (B,)),
        "optimal_action_mask": torch.randint(0, 2, (B, 6)),
        "target_discounted_return": torch.randn(B, 1),
        "target_trajectory_latents": torch.randn(B, 5, d_model),
        "target_is_error": torch.randint(0, 2, (B, 1)),
    }

    losses = criterion(outputs, targets)
    assert "total_loss" in losses
    assert losses["total_loss"].item() > 0.0
    for key in ["loss_state", "loss_goal", "loss_dynamics", "loss_action", "loss_counterfactual", "loss_value", "loss_plan", "loss_verify", "loss_efficiency"]:
        assert key in losses


def test_loss_backward_gradient_flow():
    """Verify that gradients flow properly through all reasoner modules without NaNs."""
    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    model.train()

    B = 1
    K = 4
    slots = torch.randn(B, K, config.slot_dim)
    spatial = torch.randn(B, 16, 16, config.feat_dim)

    outputs = model(slot_embeddings=slots, spatial_features=spatial)
    # Simple loss to test gradient backprop through the entire 18-layer trunk & heads
    loss = outputs["cognitive_state"].sum() + outputs["action_logits"].sum() + outputs["value"].sum()
    loss.backward()

    # Verify gradients exist and are finite
    assert model.trunk.layers[0].self_attn.q_proj.weight.grad is not None
    assert not torch.isnan(model.trunk.layers[0].self_attn.q_proj.weight.grad).any()
    assert model.trunk.layers[-1].mlp.gate_proj.weight.grad is not None
    assert not torch.isnan(model.trunk.layers[-1].mlp.gate_proj.weight.grad).any()
    assert model.dense_projections.slot_proj[0].weight.grad is not None
    assert not torch.isnan(model.dense_projections.slot_proj[0].weight.grad).any()


def test_reasoning_arc_dataset_integration():
    """Verify that ReasoningArcDataset samples feed directly into CognitiveReasoner120M."""
    from cir_arc.neural.training.reasoning_dataset import ReasoningArcDataset, collate_reasoning_batch

    dataset = ReasoningArcDataset(data_dir="data/synthetic/train", max_samples=4)
    if len(dataset) == 0:
        pytest.skip("Synthetic dataset directory not found.")

    batch = collate_reasoning_batch([dataset[0], dataset[1]])

    assert batch["slot_embeddings"].shape == (2, 24, 224)
    assert batch["candidate_scores"].shape == (2, 7)
    assert batch["mechanics_vec"].shape == (2, 11)
    assert batch["target_is_error"].shape == (2, 1)

    config = ReasonerConfig()
    model = CognitiveReasoner120M(config)
    model.eval()

    with torch.no_grad():
        out = model(slot_embeddings=batch["slot_embeddings"])

    assert out["cognitive_state"].shape == (2, config.d_model)
    assert out["action_logits"].shape == (2, config.num_action_types)
    assert out["goals"].shape == (2, config.num_goal_hypotheses, config.d_model)


def test_reasoner_metrics_tracker():
    """Verify ReasonerMetricsTracker correctly aggregates F1, accuracy, losses, and ARC-AGI scorecards."""
    from cir_arc.neural.evaluation.reasoner_metrics import ReasonerMetricsTracker

    tracker = ReasonerMetricsTracker()
    tracker.reset()

    # 1. Update running losses
    tracker.update_losses({
        "total_loss": torch.tensor(2.45),
        "loss_action": torch.tensor(1.10),
        "loss_verify": torch.tensor(0.35),
    })

    # 2. Update batch predictions and targets
    B = 4
    d_model = 768
    outputs = {
        "cognitive_state": torch.randn(B, d_model),
        "action_logits": torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]),
        "prediction_error": torch.tensor([[2.0], [-2.0], [2.0], [-2.0]]),  # sigmoid -> [1, 0, 1, 0]
        "goals": torch.randn(B, 4, d_model),
        "candidate_scores": torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]),
        "predicted_next_latent": torch.randn(B, d_model),
    }
    targets = {
        "target_action_id": torch.tensor([0, 1, 1, 0]),  # 3 correct top1 (0, 1, 0), 1 incorrect (1 != 2)
        "target_is_error": torch.tensor([[1.0], [0.0], [0.0], [0.0]]),   # TP=1, FP=1, TN=2, FN=0
        "target_goal_latent": outputs["goals"][:, 0].clone(),             # Perfect cosine similarity = 1.0
        "optimal_action_mask": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "target_next_latent": outputs["predicted_next_latent"].clone(),    # MSE = 0.0
    }

    tracker.update_batch(outputs, targets)
    tracker.update_grid_matches(exact_matches=3, total_grids=4)

    metrics = tracker.compute()

    assert "total_loss" in metrics
    assert metrics["action_accuracy"] == 0.75  # 3 / 4
    assert metrics["action_macro_f1"] > 0.0
    assert metrics["verification_accuracy"] == 0.75  # (1 TP + 2 TN) / 4
    assert metrics["verification_precision"] == 0.5   # 1 / (1 + 1)
    assert metrics["verification_recall"] == 1.0      # 1 / (1 + 0)
    assert metrics["verification_f1"] > 0.0
    assert abs(metrics["goal_cosine_similarity"] - 1.0) < 1e-4
    assert metrics["counterfactual_ranking_accuracy"] == 1.0
    assert abs(metrics["dynamics_latent_mse"]) < 1e-5
    assert metrics["exact_grid_match_rate"] == 0.75

    # Verify formatted scorecard printing does not raise
    tracker.print_scorecard(epoch=1)


