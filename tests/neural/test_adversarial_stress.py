"""Adversarial Stress Test Suite for CIR-ARC Phase 2 Perception System.

Adversarially stress-tests:
1. Extreme heterogeneous batch padding (1x1, 1x30, 30x1, 30x30).
2. Edge-case ground-truth object counts (M=0, M=1, M=24, M=40 > K).
3. Numerical stability under extreme inputs, slot collapse, zero masks, huge gradients.
4. Checkpoint serialization, cross-device loading, and eval mode determinism.
5. Multi-task loss numerical properties and gradient norm clipping.
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cir_arc.core.grid import Grid
from cir_arc.core.objects import ArcObject, extract_objects
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
    compute_property_losses,
)
from cir_arc.neural.losses.diversity import diversity_loss, objectness_sparsity_loss
from cir_arc.neural.training.dataset import collate_variable_grids
from cir_arc.neural.training.trainer import PerceptionModel, Trainer
from cir_arc.neural.evaluation.perception_metrics import (
    reconstruction_accuracy,
    object_detection_f1,
    color_accuracy,
    position_mae,
    size_mae,
    compute_perception_metrics,
)


class TestAdversarialExtremePadding:
    """Stress tests with extreme heterogeneous batch sizes and aspect ratios."""

    def test_extreme_heterogeneous_batch_padding(self):
        """Batch with 1x1, 30x30, 1x30, 30x1, 2x2 grids."""
        samples = [
            {
                "input_grid": torch.tensor([[5]], dtype=torch.long),
                "height": 1,
                "width": 1,
                "gt_objects": [ArcObject(color=5, pixels=np.array([[0, 0]]))],
                "task_id": "1x1",
            },
            {
                "input_grid": torch.randint(0, 10, (30, 30), dtype=torch.long),
                "height": 30,
                "width": 30,
                "gt_objects": [
                    ArcObject(color=1, pixels=np.array([[0, 0]])),
                    ArcObject(color=2, pixels=np.array([[29, 29]])),
                ],
                "task_id": "30x30",
            },
            {
                "input_grid": torch.randint(0, 10, (1, 30), dtype=torch.long),
                "height": 1,
                "width": 30,
                "gt_objects": [ArcObject(color=3, pixels=np.array([[0, 15]]))],
                "task_id": "1x30",
            },
            {
                "input_grid": torch.randint(0, 10, (30, 1), dtype=torch.long),
                "height": 30,
                "width": 1,
                "gt_objects": [ArcObject(color=4, pixels=np.array([[15, 0]]))],
                "task_id": "30x1",
            },
            {
                "input_grid": torch.randint(0, 10, (2, 2), dtype=torch.long),
                "height": 2,
                "width": 2,
                "gt_objects": [],
                "task_id": "2x2",
            },
        ]

        batch = collate_variable_grids(samples)
        assert batch["input_grids"].shape == (5, 30, 30)
        assert batch["input_masks"].shape == (5, 30, 30)

        # Verify spatial masks exactly match active regions
        assert batch["input_masks"][0, 0, 0] == 1.0
        assert batch["input_masks"][0].sum().item() == 1.0

        assert batch["input_masks"][1].sum().item() == 900.0

        assert batch["input_masks"][2, 0, :].sum().item() == 30.0
        assert batch["input_masks"][2, 1:, :].sum().item() == 0.0

        assert batch["input_masks"][3, :, 0].sum().item() == 30.0
        assert batch["input_masks"][3, :, 1:].sum().item() == 0.0

        assert batch["input_masks"][4, :2, :2].sum().item() == 4.0
        assert batch["input_masks"][4, 2:, :].sum().item() == 0.0

        # Forward pass through PerceptionModel
        model = PerceptionModel()
        out = model(batch["input_grids"], mask=batch["input_masks"])

        assert out["slots"].shape == (5, 24, 128)
        assert out["objectness"].shape == (5, 24)
        assert out["attn_maps"].shape == (5, 24, 900)
        assert out["recon_logits"].shape == (5, 30, 30, 10)

        # Check competitive binding invariant
        attn_sums = out["attn_maps"].sum(dim=1)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)

        # Check trainer step with this extreme batch
        trainer = Trainer(model=model, lr=1e-3)
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])
        assert not np.isinf(metrics["loss"])
        assert metrics["loss"] > 0.0


class TestAdversarialObjectCountEdges:
    """Stress tests with M=0, M=1, M=24, and M=40 > K object counts."""

    def test_empty_objects_m0(self):
        """All batch items have zero ground-truth objects (e.g. blank background)."""
        K = 24
        pred_props = {
            "color": torch.randn(K, 10, requires_grad=True),
            "position": torch.rand(K, 2, requires_grad=True),
            "size": torch.rand(K, 1, requires_grad=True),
        }
        objectness = torch.rand(K, requires_grad=True)

        matches = hungarian_matching(pred_props, [], H=10, W=10)
        assert matches == []

        loss_dict = compute_property_losses(
            pred_props=pred_props,
            objectness=objectness,
            gt_objects=[],
            matches=matches,
            H=10,
            W=10,
        )
        total_loss = loss_dict["total_property_loss"]
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

        total_loss.backward()
        assert pred_props["color"].grad is not None
        assert not torch.isnan(pred_props["color"].grad).any()
        assert objectness.grad is not None
        assert not torch.isnan(objectness.grad).any()

    def test_overloaded_objects_m40_exceeding_k24(self):
        """Grid with 40 distinct 1x1 objects, exceeding slot capacity K=24."""
        K = 24
        M = 40
        pred_props = {
            "color": torch.randn(K, 10, requires_grad=True),
            "position": torch.rand(K, 2, requires_grad=True),
            "size": torch.rand(K, 1, requires_grad=True),
        }
        objectness = torch.rand(K, requires_grad=True)

        gt_objects = [
            ArcObject(color=(i % 9) + 1, pixels=np.array([[i // 6, i % 6]]))
            for i in range(M)
        ]

        matches = hungarian_matching(pred_props, gt_objects, H=10, W=10)
        assert len(matches) == K  # Exactly 24 matched pairs (all slots assigned)

        # Verify slots are uniquely assigned (bijective subset)
        slot_indices = [s for s, _ in matches]
        assert len(set(slot_indices)) == K

        loss_dict = compute_property_losses(
            pred_props=pred_props,
            objectness=objectness,
            gt_objects=gt_objects,
            matches=matches,
            H=10,
            W=10,
        )
        total_loss = loss_dict["total_property_loss"]
        assert not torch.isnan(total_loss)
        assert total_loss.item() > 0.0

        total_loss.backward()
        assert pred_props["color"].grad is not None
        assert not torch.isnan(pred_props["color"].grad).any()

    def test_trainer_with_mixed_object_counts(self):
        """Batch containing M=0, M=1, M=10, and M=30."""
        samples = [
            {
                "input_grid": torch.zeros((10, 10), dtype=torch.long),
                "height": 10,
                "width": 10,
                "gt_objects": [],
                "task_id": "m0",
            },
            {
                "input_grid": torch.randint(0, 10, (10, 10), dtype=torch.long),
                "height": 10,
                "width": 10,
                "gt_objects": [ArcObject(color=2, pixels=np.array([[1, 1]]))],
                "task_id": "m1",
            },
            {
                "input_grid": torch.randint(0, 10, (10, 10), dtype=torch.long),
                "height": 10,
                "width": 10,
                "gt_objects": [
                    ArcObject(color=(i % 9) + 1, pixels=np.array([[i, i]]))
                    for i in range(10)
                ],
                "task_id": "m10",
            },
            {
                "input_grid": torch.randint(0, 10, (10, 10), dtype=torch.long),
                "height": 10,
                "width": 10,
                "gt_objects": [
                    ArcObject(color=(i % 9) + 1, pixels=np.array([[i // 3, i % 3]]))
                    for i in range(30)
                ],
                "task_id": "m30",
            },
        ]
        batch = collate_variable_grids(samples)
        model = PerceptionModel()
        trainer = Trainer(model=model, lr=1e-3)
        metrics = trainer.train_step(batch)

        assert not np.isnan(metrics["loss"])
        assert not np.isnan(metrics["color_loss"])
        assert not np.isnan(metrics["pos_loss"])
        assert not np.isnan(metrics["size_loss"])
        assert not np.isnan(metrics["obj_loss"])


class TestAdversarialNumericalStability:
    """Stress tests for numerical stability: extreme values, slot collapse, zero masks, grad clipping."""

    def test_extreme_reconstruction_logits(self):
        """Test reconstruction loss with extreme positive/negative logits (+-1000.0)."""
        B, H, W = 2, 5, 5
        target = torch.randint(0, 10, (B, H, W), dtype=torch.long)

        # Extreme logits
        logits_extreme = torch.zeros(B, H, W, 10, requires_grad=True)
        with torch.no_grad():
            logits_extreme[:, :, :, 0] = 1000.0
            logits_extreme[:, :, :, 1] = -1000.0

        loss = reconstruction_loss(logits_extreme, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        loss.backward()
        assert not torch.isnan(logits_extreme.grad).any()
        assert not torch.isinf(logits_extreme.grad).any()

    def test_slot_collapse_diversity_loss(self):
        """All slots are completely identical (cosine similarity = 1.0)."""
        B, K, D = 2, 24, 128
        # Create identical slot vectors
        single_slot = torch.randn(B, 1, D)
        identical_slots = single_slot.expand(B, K, D).clone().requires_grad_(True)

        loss = diversity_loss(identical_slots)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert torch.allclose(loss, torch.tensor(1.0), atol=1e-4)

        loss.backward()
        assert not torch.isnan(identical_slots.grad).any()
        assert not torch.isinf(identical_slots.grad).any()

    def test_zero_vector_diversity_loss(self):
        """Slots are exact zeros (testing eps in norm)."""
        B, K, D = 2, 24, 128
        zero_slots = torch.zeros(B, K, D, requires_grad=True)
        loss = diversity_loss(zero_slots)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        loss.backward()
        assert not torch.isnan(zero_slots.grad).any()

    def test_gradient_clipping_effectiveness(self):
        """Verify clip_grad_norm_ prevents exploding gradients under massive loss multiplier."""
        model = PerceptionModel()
        trainer = Trainer(model=model, lr=1e-3, clip_grad_norm=1.0)

        batch = {
            "input_grids": torch.randint(0, 10, (2, 8, 8), dtype=torch.long),
            "input_masks": torch.ones(2, 8, 8),
            "gt_objects": [[ArcObject(color=1, pixels=np.array([[0, 0]]))], []],
            "heights": [8, 8],
            "widths": [8, 8],
        }

        # Intentionally scale weights massively in trainer
        trainer.recon_weight = 1e6
        trainer.color_weight = 1e6
        metrics = trainer.train_step(batch)

        assert not np.isnan(metrics["loss"])
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

        # Check total parameter norm is finite
        total_norm = torch.norm(
            torch.stack([torch.norm(p.detach(), 2) for p in model.parameters()]), 2
        ).item()
        assert not np.isnan(total_norm)
        assert not np.isinf(total_norm)


class TestAdversarialCheckpointAndDeterminism:
    """Stress tests for checkpoint serialization, device transfer, and determinism."""

    def test_eval_mode_bit_exact_reproducibility(self):
        """In eval mode, repeated forward passes on identical input produce exact same outputs."""
        model = PerceptionModel()
        model.eval()

        test_in = torch.randint(0, 10, (3, 12, 12), dtype=torch.long)
        with torch.no_grad():
            out1 = model(test_in)
            out2 = model(test_in)

        assert torch.equal(out1["slots"], out2["slots"])
        assert torch.equal(out1["objectness"], out2["objectness"])
        assert torch.equal(out1["recon_logits"], out2["recon_logits"])

    def test_checkpoint_roundtrip_device_transfer(self):
        """Save checkpoint on CPU, load and execute in separate model instance."""
        model1 = PerceptionModel()
        trainer1 = Trainer(model=model1, lr=5e-4)

        # Do 2 train steps to modify weights and optimizer state
        batch = {
            "input_grids": torch.randint(0, 10, (2, 6, 6), dtype=torch.long),
            "input_masks": torch.ones(2, 6, 6),
            "gt_objects": [[ArcObject(color=1, pixels=np.array([[0, 0]]))], []],
            "heights": [6, 6],
            "widths": [6, 6],
        }
        trainer1.train_step(batch)
        trainer1.train_step(batch)
        assert trainer1.step == 2

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = os.path.join(tmpdir, "adv_checkpoint.pt")
            trainer1.save_checkpoint(ckpt_file)

            model2 = PerceptionModel()
            trainer2 = Trainer(model=model2, lr=5e-4)
            state = trainer2.load_checkpoint(ckpt_file)

            assert trainer2.step == 2
            model1.eval()
            model2.eval()

            eval_input = torch.randint(0, 10, (4, 7, 7), dtype=torch.long)
            with torch.no_grad():
                pred1 = model1(eval_input)
                pred2 = model2(eval_input)

            assert torch.allclose(pred1["slots"], pred2["slots"], atol=1e-6)
            assert torch.allclose(pred1["recon_logits"], pred2["recon_logits"], atol=1e-6)
            assert torch.allclose(pred1["objectness"], pred2["objectness"], atol=1e-6)
