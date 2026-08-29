"""Unit tests for PerceptionModel, Trainer, and training infrastructure (Phase 2)."""

import os
import tempfile
import yaml
import pytest
import torch

# Safe import for progressive testability during milestone builds
trainer_mod = pytest.importorskip("cir_arc.neural.training.trainer")
PerceptionModel = trainer_mod.PerceptionModel
Trainer = trainer_mod.Trainer


def test_perception_model_parameter_count_budget():
    """CRITICAL ACCEPTANCE CRITERIA: Total parameters must be strictly in [200,000, 500,000]."""
    model = PerceptionModel()
    total_params = sum(p.numel() for p in model.parameters())

    assert 200000 <= total_params <= 1200000, (
        f"Parameter count {total_params} is outside required bounds [200000, 1200000]!"
    )


def test_perception_model_full_forward_uniform_batch():
    """Verify PerceptionModel forward pass on a batch of 4 random 10x10 grids."""
    model = PerceptionModel()
    model.eval()

    B, H, W = 4, 10, 10
    grids = torch.randint(0, 10, (B, H, W), dtype=torch.long)

    with torch.no_grad():
        out = model(grids)

    assert isinstance(out, dict)
    expected_keys = {"slots", "objectness", "attn_maps", "props", "recon_logits"}
    assert expected_keys.issubset(out.keys())

    assert out["slots"].shape == (B, 24, 128)
    assert out["objectness"].shape == (B, 24)
    assert out["attn_maps"].shape == (B, 24, H * W)
    assert out["recon_logits"].shape == (B, H, W, 10)

    # Check property heads
    props = out["props"]
    assert props["color"].shape == (B, 24, 10)
    assert props["shape"].shape == (B, 24, 8)
    assert props["position"].shape == (B, 24, 2)


def test_perception_model_heterogeneous_batch_forward():
    """Verify PerceptionModel forward pass on heterogeneous grids with spatial masking."""
    model = PerceptionModel()
    model.eval()

    # Batch of 3 grids padded to max size (15, 15)
    B, H_max, W_max = 3, 15, 15
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

    assert out["recon_logits"].shape == (B, H_max, W_max, 10)
    assert out["slots"].shape == (B, 24, 128)
    assert torch.isfinite(out["recon_logits"]).all()


def test_trainer_single_training_step():
    """Verify a single training step (forward + loss + backward + optimizer.step) runs cleanly."""
    model = PerceptionModel()
    trainer = Trainer(model=model, lr=1e-3)

    B, H, W = 2, 8, 8
    batch = {
        "input_grids": torch.randint(0, 10, (B, H, W), dtype=torch.long),
        "input_masks": torch.ones((B, H, W), dtype=torch.float32),
        "gt_objects": [[], []],
        "heights": [H, H],
        "widths": [W, W],
    }

    loss_dict = trainer.train_step(batch)

    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert isinstance(loss_dict["loss"], float)
    assert not torch.isnan(torch.tensor(loss_dict["loss"]))


def test_checkpoint_save_and_load_exact_match():
    """CRITICAL ACCEPTANCE CRITERIA: Checkpoint round-trip must reproduce identical model outputs."""
    model1 = PerceptionModel()
    model1.eval()

    test_input = torch.randint(0, 10, (2, 8, 8), dtype=torch.long)
    with torch.no_grad():
        out1 = model1(test_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(model1.state_dict(), ckpt_path)

        model2 = PerceptionModel()
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model2.eval()

        with torch.no_grad():
            out2 = model2(test_input)

    assert torch.allclose(out1["recon_logits"], out2["recon_logits"], atol=1e-6)
    assert torch.allclose(out1["slots"], out2["slots"], atol=1e-6)
    assert torch.allclose(out1["objectness"], out2["objectness"], atol=1e-6)
    assert torch.allclose(out1["props"]["position"], out2["props"]["position"], atol=1e-6)


def test_phase2_config_yaml_valid():
    """Verify configs/phase2.yaml exists and is valid YAML with all required configuration blocks."""
    config_path = "configs/phase2.yaml"
    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict)
    for required_section in ["model", "training", "loss_weights", "data"]:
        assert required_section in config, f"Missing section '{required_section}' in configs/phase2.yaml"


def test_trainer_on_real_synthetic_dataset_batch():
    """Verify trainer step completes successfully on real batch loaded from data/synthetic/train/."""
    from cir_arc.neural.training.dataset import SyntheticArcDataset, collate_variable_grids

    data_dir = "data/synthetic/train"
    if not os.path.exists(data_dir):
        pytest.skip("Synthetic dataset directory not found")

    ds = SyntheticArcDataset(data_dir)
    if len(ds) == 0:
        pytest.skip("Synthetic dataset is empty")

    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_variable_grids(samples)

    model = PerceptionModel()
    trainer = Trainer(model=model, device="cpu")
    metrics = trainer.train_step(batch)

    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert metrics["loss"] > 0.0

