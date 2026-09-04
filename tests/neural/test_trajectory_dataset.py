"""Comprehensive unit tests for the 5-Layer Trajectory Dataset, Explicit Masking, and Dual Consistency Loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cir_arc.generators.trajectory_dataset import ProceduralTrajectoryGenerator, TrajectoryStep
from cir_arc.neural.training.trajectory_dataset import TrajectoryArcDataset, collate_trajectory_batch
from cir_arc.neural.training.trainer import PerceptionModel, Trainer
from cir_arc.neural.losses.consistency import latent_transition_loss, neuro_symbolic_alignment_loss


class TestTrajectoryGeneratorLayers:
    """Validates generation of all 5 dataset layers and negative examples."""

    def test_generator_5_layers(self) -> None:
        gen = ProceduralTrajectoryGenerator(seed=123)

        s1 = gen.generate_layer1_static()
        assert s1.layer == 1
        assert len(s1.grid_t) == 15 and len(s1.grid_t[0]) == 15
        assert np.array(s1.grid_t).max() <= 9  # Colors strictly in 0-9

        s2 = gen.generate_layer2_temporal()
        assert s2.layer == 2
        assert len(s2.events) >= 1
        assert s2.events[0]["event_type"] == "MOVE"

        s3 = gen.generate_layer3_action_transition()
        assert s3.layer == 3
        assert s3.action in (1, 2, 3, 4)
        assert "success" in s3.action_effect

        s4 = gen.generate_layer4_mechanics_discovery()
        assert s4.layer == 4
        assert len(s4.mechanics_evidence) > 0

        s5 = gen.generate_layer5_novel_stress_test()
        assert s5.layer == 5
        assert "coupled_symmetry" in s5.mechanics_evidence

        s_neg = gen.generate_negative_example()
        assert s_neg.is_negative_example is True
        assert len(s_neg.ambiguity_notes) > 0

    def test_balanced_corpus_composition(self) -> None:
        gen = ProceduralTrajectoryGenerator(seed=42)
        corpus = gen.generate_balanced_corpus(n_samples=100)

        assert len(corpus) == 100
        layers = [s.layer for s in corpus]
        assert 1 in layers
        assert 2 in layers
        assert 3 in layers
        assert 4 in layers
        assert 5 in layers
        assert any(s.is_negative_example for s in corpus)


class TestTrajectoryDatasetAndCollate:
    """Validates PyTorch Dataset and explicit boolean valid_mask collation."""

    def test_dataset_and_mask_collation(self) -> None:
        gen = ProceduralTrajectoryGenerator(seed=99)
        corpus = [
            gen.generate_layer1_static(H=10, W=10),
            gen.generate_layer3_action_transition(H=14, W=14),
        ]

        dataset = TrajectoryArcDataset(in_memory_steps=corpus)
        assert len(dataset) == 2

        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_trajectory_batch)
        batch = next(iter(loader))

        assert batch["grid_t"].shape == (2, 14, 14)
        assert batch["valid_mask_t"].shape == (2, 14, 14)
        assert batch["grid_next"].shape == (2, 14, 14)
        assert batch["valid_mask_next"].shape == (2, 14, 14)
        assert batch["action"].shape == (2,)

        # Check that item 0 (10x10) has False in its padded border [10:14, 10:14]
        mask_0 = batch["valid_mask_t"][0]
        assert mask_0[:10, :10].all()
        assert not mask_0[10:, 10:].any()

        # Grid colors must be strictly within 0-9 (no fake color 10 for padding)
        assert batch["grid_t"].max() <= 9
        assert batch["grid_next"].max() <= 9


class TestMutualConsistencyLossAndTrainingStep:
    """Validates latent transition loss and mutual neuro-symbolic alignment training."""

    def test_latent_transition_and_alignment_losses(self) -> None:
        B, K, D = 2, 12, 64
        pred_slots = torch.randn(B, K, D, requires_grad=True)
        target_slots = torch.randn(B, K, D)

        loss_t = latent_transition_loss(pred_slots, target_slots)
        assert loss_t.item() > 0.0
        loss_t.backward()
        assert pred_slots.grad is not None

        prop_logits = {
            "color": torch.randn(B, K, 10),
            "shape": torch.randn(B, K, 8),
            "position": torch.sigmoid(torch.randn(B, K, 2)),
            "size": torch.sigmoid(torch.randn(B, K, 1)),
        }
        loss_align = neuro_symbolic_alignment_loss(pred_slots, prop_logits)
        assert 0.0 <= loss_align.item() <= 2.0

    def test_trainer_trajectory_step(self) -> None:
        model = PerceptionModel(
            stem_hidden_dim=32,
            stem_out_dim=64,
            slot_dim=64,
            feat_dim=64,
            prop_hidden_dim=32,
            include_v2_modules=True,
        )
        trainer = Trainer(model=model, device="cpu")

        gen = ProceduralTrajectoryGenerator(seed=77)
        corpus = [gen.generate_layer3_action_transition(H=12, W=12) for _ in range(2)]
        dataset = TrajectoryArcDataset(in_memory_steps=corpus)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_trajectory_batch)
        batch = next(iter(loader))

        metrics = trainer.train_trajectory_step(batch)
        assert "loss" in metrics
        assert "loss_transition" in metrics
        assert "loss_alignment" in metrics
        assert "loss_recon_t" in metrics
        assert metrics["loss"] > 0.0
