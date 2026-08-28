"""End-to-End Acceptance Tests for CIR-ARC Phase 2 (Object-Centric Neural Perception).

Validates all formal acceptance criteria specified in ORIGINAL_REQUEST.md:
- Architecture Verification (parameter budget, attention invariants, batch handling)
- Training Readiness (dataset loading, 1-step optimization, checkpoint roundtrip)
- Integration & Configuration (phase2.yaml validation)
- Phase 1 Non-Regression
"""

import os
import tempfile
import yaml
import pytest
import torch

from cir_arc.core.grid import Grid
from cir_arc.core.objects import extract_objects
from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.cnn_stem import CNNStem
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.perception.property_heads import PropertyHeads
from cir_arc.neural.perception.reconstruction import ReconstructionDecoder

# Safe imports for progressive testability
trainer_mod = pytest.importorskip("cir_arc.neural.training.trainer")
dataset_mod = pytest.importorskip("cir_arc.neural.training.dataset")

PerceptionModel = trainer_mod.PerceptionModel
Trainer = trainer_mod.Trainer
SyntheticArcDataset = dataset_mod.SyntheticArcDataset
collate_variable_grids = dataset_mod.collate_variable_grids


def test_acceptance_parameter_count_range():
    """ACCEPTANCE CRITERIA: Total parameter count of PerceptionModel is strictly in [200K, 500K]."""
    model = PerceptionModel()
    total_params = sum(p.numel() for p in model.parameters())

    assert 200000 <= total_params <= 500000, (
        f"PerceptionModel parameter count ({total_params}) violated acceptance constraint [200K, 500K]"
    )


def test_acceptance_competitive_binding_sum_to_one():
    """ACCEPTANCE CRITERIA: SlotAttention attention maps sum to 1.0 (±1e-5) over slot dimension."""
    slot_attn = SlotAttention(n_slots=24, slot_dim=128, feat_dim=128, n_iter=3)

    for num_tokens in [25, 96, 100, 225, 900]:
        x = torch.randn(2, num_tokens, 128)
        _, _, attn_maps = slot_attn(x)

        # Sum across the slot dimension (K=24)
        sums = attn_maps.sum(dim=1)
        expected = torch.ones_like(sums)

        max_err = (sums - expected).abs().max().item()
        assert max_err < 1e-5, f"Competitive binding invariant violated with max error {max_err}"


def test_acceptance_perception_model_uniform_batch():
    """ACCEPTANCE CRITERIA: PerceptionModel forward pass completes on a batch of 4 random 10x10 grids."""
    model = PerceptionModel()
    model.eval()

    batch_grids = torch.randint(0, 10, (4, 10, 10), dtype=torch.long)
    with torch.no_grad():
        out = model(batch_grids)

    assert "slots" in out
    assert "objectness" in out
    assert "attn_maps" in out
    assert "props" in out
    assert "recon_logits" in out

    assert out["slots"].shape == (4, 24, 128)
    assert out["objectness"].shape == (4, 24)
    assert out["attn_maps"].shape == (4, 24, 100)
    assert out["recon_logits"].shape == (4, 10, 10, 10)


def test_acceptance_perception_model_heterogeneous_batch():
    """ACCEPTANCE CRITERIA: Forward pass on grids of different sizes (5x5, 8x12, 15x15) in same batch."""
    model = PerceptionModel()
    model.eval()

    B = 3
    H_max, W_max = 15, 15
    grids = torch.randint(0, 10, (B, H_max, W_max), dtype=torch.long)
    masks = torch.zeros((B, H_max, W_max), dtype=torch.float32)

    # Grid 0: 5x5
    masks[0, :5, :5] = 1.0
    # Grid 1: 8x12
    masks[1, :8, :12] = 1.0
    # Grid 2: 15x15
    masks[2, :15, :15] = 1.0

    with torch.no_grad():
        out = model(grids, mask=masks)

    assert out["recon_logits"].shape == (3, 15, 15, 10)
    assert torch.isfinite(out["recon_logits"]).all()


def test_acceptance_dataset_real_json_loading():
    """ACCEPTANCE CRITERIA: Dataset class successfully loads task JSONs from data/synthetic/train/."""
    train_dir = "data/synthetic/train"
    if not os.path.exists(train_dir):
        pytest.skip(f"{train_dir} does not exist")

    dataset = SyntheticArcDataset(data_dir=train_dir)
    assert len(dataset) > 0

    first_item = dataset[0]
    assert "input_grid" in first_item
    assert "gt_objects" in first_item
    assert isinstance(first_item["input_grid"], torch.Tensor)
    assert first_item["input_grid"].dtype == torch.long


def test_acceptance_single_training_step_optimization():
    """ACCEPTANCE CRITERIA: One training step (forward+backward+optimizer.step) completes cleanly."""
    model = PerceptionModel()
    trainer = Trainer(model=model, lr=1e-3)

    B, H, W = 2, 8, 8
    sample_batch = {
        "input_grids": torch.randint(0, 10, (B, H, W), dtype=torch.long),
        "input_masks": torch.ones((B, H, W), dtype=torch.float32),
        "gt_objects": [[], []],
        "heights": [H, H],
        "widths": [W, W],
    }

    metrics = trainer.train_step(sample_batch)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert not torch.isnan(torch.tensor(metrics["loss"]))


def test_acceptance_checkpoint_roundtrip_reproducibility():
    """ACCEPTANCE CRITERIA: Model produces identical outputs before save and after load."""
    model1 = PerceptionModel()
    model1.eval()

    test_input = torch.randint(0, 10, (2, 6, 6), dtype=torch.long)
    with torch.no_grad():
        out1 = model1(test_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = os.path.join(tmpdir, "model.pt")
        torch.save(model1.state_dict(), ckpt_file)

        model2 = PerceptionModel()
        model2.load_state_dict(torch.load(ckpt_file, weights_only=True))
        model2.eval()

        with torch.no_grad():
            out2 = model2(test_input)

    assert torch.allclose(out1["recon_logits"], out2["recon_logits"], atol=1e-6)
    assert torch.allclose(out1["slots"], out2["slots"], atol=1e-6)
    assert torch.allclose(out1["objectness"], out2["objectness"], atol=1e-6)


def test_acceptance_config_phase2_yaml_valid():
    """ACCEPTANCE CRITERIA: configs/phase2.yaml exists, is parseable, and has all required keys."""
    config_path = "configs/phase2.yaml"
    assert os.path.exists(config_path), f"{config_path} must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "training" in cfg
    assert "loss_weights" in cfg
    assert "data" in cfg


def test_acceptance_phase1_integration():
    """Verify Phase 1 core functionality (Grid and extract_objects) remains fully operational."""
    grid = Grid.from_list([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0],
    ])
    objects = extract_objects(grid, background_color=0)
    assert len(objects) == 2
    colors = {obj.color for obj in objects}
    assert colors == {1, 2}
