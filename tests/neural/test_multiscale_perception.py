"""Comprehensive unit and invariant tests for Phase 2 multi-scale architecture upgrade."""

import pytest
import numpy as np
import torch

from cir_arc.core.objects import ArcObject
from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import MultiScaleCNNStem, ResidualDepthwiseSeparableConv
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.relation_encoder import SlotRelationEncoder, SetTransformerBlock
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder, SlotMaskDecoder
from cir_arc.neural.losses.boundary import boundary_loss, cell_objectness_loss
from cir_arc.neural.losses.mask import slot_mask_loss, mask_exclusivity_loss
from cir_arc.neural.training.trainer import PerceptionModel, Trainer


class TestMultiScaleCNNStem:
    def test_multiscale_output_shapes(self):
        stem = MultiScaleCNNStem(in_channels=48, hidden_channels=64, out_channels=128)
        test_shapes = [(2, 10, 10, 48), (1, 5, 8, 48), (3, 15, 15, 48)]

        for shape in test_shapes:
            B, H, W, _ = shape
            x = torch.randn(shape)
            tokens, bound_map, obj_map = stem(x, return_maps=True)

            assert tokens.shape == (B, H * W, 128)
            assert bound_map.shape == (B, 1, H, W)
            assert obj_map.shape == (B, 1, H, W)
            assert (bound_map >= 0.0).all() and (bound_map <= 1.0).all()
            assert (obj_map >= 0.0).all() and (obj_map <= 1.0).all()

    def test_legacy_forward_compatibility(self):
        stem = MultiScaleCNNStem()
        x = torch.randn(2, 6, 6, 48)
        tokens = stem(x, return_maps=False)
        assert tokens.shape == (2, 36, 128)


class TestSlotRelationEncoder:
    def test_relation_encoder_shape_and_norm(self):
        encoder = SlotRelationEncoder(slot_dim=128, num_heads=4, num_layers=2)
        slots = torch.randn(3, 24, 128)
        obj = torch.rand(3, 24)

        out = encoder(slots, objectness=obj)
        assert out.shape == (3, 24, 128)
        assert not torch.isnan(out).any()

    def test_permutation_equivariance(self):
        encoder = SlotRelationEncoder(slot_dim=128, num_heads=4, num_layers=2)
        encoder.eval()

        slots = torch.randn(2, 24, 128)
        perm = torch.randperm(24)

        with torch.no_grad():
            out_orig = encoder(slots)
            out_perm = encoder(slots[:, perm, :])

        out_expected = out_orig[:, perm, :]
        max_diff = torch.max(torch.abs(out_perm - out_expected)).item()
        assert max_diff < 1e-5, f"Permutation equivariance violated! Max diff: {max_diff}"


class TestSlotMaskDecoder:
    def test_mask_decoder_shapes_and_range(self):
        decoder = SlotMaskDecoder(slot_dim=128, max_h=30, max_w=30)
        slots = torch.randn(2, 24, 128)

        for H, W in [(6, 6), (10, 15), (20, 20)]:
            masks = decoder(slots, H=H, W=W)
            assert masks.shape == (2, 24, H, W)
            assert (masks >= 0.0).all() and (masks <= 1.0).all()


class TestProposalSlotAttention:
    def test_proposal_initialization(self):
        sa = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128)
        inputs = torch.randn(2, 100, 128)
        obj_map = torch.rand(2, 1, 10, 10)

        slots, objectness, attn_maps = sa(inputs, cell_objectness=obj_map)
        assert slots.shape == (2, 24, 128)
        assert objectness.shape == (2, 24)
        assert attn_maps.shape == (2, 24, 100)

        # Competitive binding sum constraint
        attn_sum = attn_maps.sum(dim=1)
        expected_sum = torch.ones_like(attn_sum)
        assert torch.allclose(attn_sum, expected_sum, atol=1e-5)


class TestAuxiliaryLosses:
    def test_boundary_loss(self):
        pred = torch.tensor([[[[0.9, 0.1], [0.1, 0.9]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = boundary_loss(pred, target)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_cell_objectness_loss(self):
        pred = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = cell_objectness_loss(pred, target)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_mask_loss_and_exclusivity(self):
        pred_masks = torch.rand(2, 24, 10, 10)
        obj = ArcObject(color=1, pixels=np.array([[2, 2], [2, 3], [3, 2]]))
        gt_objects = [[obj], [obj]]
        matches = [[(0, 0)], [(1, 0)]]

        m_loss = slot_mask_loss(pred_masks, gt_objects, matches)
        excl_loss = mask_exclusivity_loss(pred_masks)

        assert m_loss.item() >= 0.0
        assert excl_loss.item() >= 0.0


class TestPerceptionModelBudgetAndIntegration:
    def test_parameter_budget_target(self):
        model = PerceptionModel()
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 800000 <= param_count <= 1200000, (
            f"Parameter count {param_count:,} outside target budget [800K, 1.2M]!"
        )

    def test_full_pipeline_forward_and_backward(self):
        model = PerceptionModel()
        trainer = Trainer(model=model, lr=1e-3)

        sample_batch = {
            "input_grids": torch.randint(0, 10, (2, 8, 8), dtype=torch.long),
            "input_masks": torch.ones((2, 8, 8), dtype=torch.float32),
            "boundary_targets": torch.rand((2, 1, 8, 8)),
            "objectness_targets": torch.rand((2, 1, 8, 8)),
            "gt_objects": [
                [ArcObject(color=3, pixels=np.array([[1, 1], [1, 2]]))],
                [ArcObject(color=5, pixels=np.array([[3, 3]]))],
            ],
            "heights": [8, 8],
            "widths": [8, 8],
        }

        metrics = trainer.train_step(sample_batch)
        assert "loss" in metrics and isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0.0
        assert "loss_bound" in metrics
        assert "loss_mask" in metrics
        assert "loss_excl" in metrics
