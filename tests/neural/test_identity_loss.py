"""Unit tests for Relational and Contrastive Identity Loss functions."""

import numpy as np
import pytest
import torch

from cir_arc.neural.losses.relation import relation_loss
from cir_arc.neural.losses.identity import object_identity_contrastive_loss


def test_relation_loss_computation():
    """Verify relation_loss handles batched predictions and gradients."""
    pred_rel = torch.randn(2, 24, 24, 14, requires_grad=True)
    gt_targets = [np.ones((2, 2, 14), dtype=np.float32), np.zeros((3, 3, 14), dtype=np.float32)]
    matches = [[(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)]]

    loss = relation_loss(pred_rel, gt_targets, matches)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert pred_rel.grad is not None


def test_object_identity_contrastive_loss():
    """Verify object_identity_contrastive_loss computation and gradients."""
    identities = torch.randn(2, 24, 64, requires_grad=True)
    matches = [[(0, 0), (1, 1)], [(0, 0), (1, 1)]]

    loss = object_identity_contrastive_loss(identities, matches)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert identities.grad is not None
