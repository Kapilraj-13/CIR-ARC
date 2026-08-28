# Project: CIR-ARC Phase 2 (Object-Centric Neural Perception)

## Architecture

CIR-ARC Phase 2 builds an object-centric neural perception system using learned Slot Attention to decompose ARC grids into discrete, symbolic object representations.

```
Discrete Grid (B, H, W) [values 0-9, mask 10]
          │
          ▼
   ColorEmbedding: nn.Embedding(11, 48)  --> (B, H, W, 48)
          │
          ▼
       CNNStem: 3-layer spatial encoder (Conv2d 48->64, 2x DepthwiseSeparableConv 64->128, 128->128, GroupNorm+GELU)
          │
          ▼
  Spatial Tokens: (B, H*W, 128)
          │
          ▼
   SlotAttention: K=24 slots, dim=128, competitive binding softmax(Q K^T / sqrt(d), dim=slots)
          ├──> Slots: (B, 24, 128)
          ├──> Objectness: (B, 24) [Sigmoid]
          └──> Attention Maps: (B, 24, H*W) [sums to 1.0 along slot dim]
          │
          ├───> PropertyHeads (6 parallel MLPs)
          │     ├── Color: (B, 24, 10) [Logits]
          │     ├── Shape: (B, 24, 8) [Logits]
          │     ├── Size: (B, 24, 1) [Sigmoid]
          │     ├── Position: (B, 24, 2) [Sigmoid (row, col)]
          │     ├── Orientation: (B, 24, 4) [Logits]
          │     └── Symmetry: (B, 24, 4) [Binary Logits]
          │
          └───> ReconstructionDecoder (Cross-attention with learned Row/Col Pos Embeddings)
                └── Color Logits: (B, H, W, 10)
```

## Feature Inventory

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | ColorEmbedding | `nn.Embedding(11, 48)` for ARC colors 0-9 + mask token 10 | M1 | R1 |
| 2 | CNNStem | 3-layer resolution-preserving encoder (48->64->128->128, GroupNorm, GELU) | M1 | R1 |
| 3 | SlotAttention | Competitive binding slot attention (K=24, dim=128, 3 iters, GRUCell, objectness head) | M1 | R1 |
| 4 | PropertyHeads | Parallel MLPs for color, shape, size, position, orientation, symmetry | M1 | R1 |
| 5 | ReconstructionDecoder | Cell-to-slot cross-attention with learned 2D pos embeddings predicting (B,H,W,10) | M1 | R1 |
| 6 | HungarianMatching | Optimal bijective slot-to-object matching using scipy linear_sum_assignment | M2 | R2 |
| 7 | ReconstructionLoss | Masked cell-level cross-entropy loss between logits and target grid | M2 | R2 |
| 8 | PropertyLoss | Supervised loss on matched slot-object pairs (Color CE, Pos MSE, Size MSE) | M2 | R2 |
| 9 | DiversityLoss | Slot cosine similarity penalty + objectness L1 sparsity penalty | M2 | R2 |
| 10 | PerceptionModel | Unified nn.Module chaining embedding->CNN->SlotAttention->Heads->Decoder | M3 | R3 |
| 11 | Trainer | AdamW training loop with gradient clipping (1.0), loss weights, checkpointing | M3 | R3 |
| 12 | SyntheticDataset | PyTorch Dataset loading synthetic task JSONs, extracting GT objects, padding grids | M3 | R3 |
| 13 | Phase2Config | YAML configuration file (`configs/phase2.yaml`) with all hyperparameters | M3 | R3 |
| 14 | PerceptionMetrics | Recon acc, Object detection F1, Color acc, Pos MAE, Size MAE | M4 | R4 |
| 15 | NeuralTestSuite | Comprehensive unit, property, and integration test suite (>=20 tests) | E2E/M5 | R5 |
| 16 | AcceptanceVerification | Verification of all acceptance criteria, parameter bounds [200K, 500K], git commit | M5 | AC |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Neural Perception Pipeline | `src/cir_arc/neural/perception/` (embedding, cnn_stem, slot_attention, property_heads, reconstruction) | None | DONE |
| M2 | Loss Functions & Matching | `src/cir_arc/neural/losses/` (matching, reconstruction, property, diversity) | M1 | DONE |
| M3 | Training Infra & Dataset | `src/cir_arc/neural/training/` (trainer with PerceptionModel, dataset) & `configs/phase2.yaml` | M1, M2 | DONE |
| M4 | Perception Metrics | `src/cir_arc/neural/evaluation/perception_metrics.py` | M1, M2 | DONE |
| M5 | Final Integration & E2E Verification | Pass all Phase 1 & Phase 2 tests, verify parameter count, attention sums, checkpointing, git commit | M1, M2, M3, M4, E2E | DONE |

## Interface Contracts

### ColorEmbedding
- Input: `grid: torch.LongTensor` shape `(B, H, W)` with values in `0..10`
- Output: `torch.FloatTensor` shape `(B, H, W, 48)`

### CNNStem
- Input: `x: torch.FloatTensor` shape `(B, H, W, 48)`
- Output: `torch.FloatTensor` shape `(B, H*W, 128)`

### SlotAttention
- Input: `inputs: torch.FloatTensor` shape `(B, N, 128)` where `N = H*W`, optional `mask: torch.BoolTensor` shape `(B, N)`
- Output:
  - `slots`: `(B, K, 128)` where `K = 24`
  - `objectness`: `(B, K)` in `[0, 1]`
  - `attn_maps`: `(B, K, N)` where `sum(dim=1)` equals 1.0 ± 1e-5

### PropertyHeads
- Input: `slots: torch.FloatTensor` shape `(B, K, 128)`
- Output: `dict` containing:
  - `color`: `(B, K, 10)`
  - `shape`: `(B, K, 8)`
  - `size`: `(B, K, 1)` (in `[0, 1]`)
  - `position`: `(B, K, 2)` (in `[0, 1]`)
  - `orientation`: `(B, K, 4)`
  - `symmetry`: `(B, K, 4)`

### ReconstructionDecoder
- Input: `slots: (B, K, 128)`, `objectness: (B, K)`, `H: int`, `W: int`
- Output: `torch.FloatTensor` shape `(B, H, W, 10)`

### Hungarian Matching (`hungarian_matching`)
- Input: `pred_props: dict`, `gt_objects: List[ArcObject]`, `H: int`, `W: int`
- Output: `List[Tuple[int, int]]` list of `(pred_slot_idx, gt_obj_idx)` pairs

### PerceptionModel
- Input: `grid: torch.LongTensor` shape `(B, H, W)`, optional `mask: torch.BoolTensor` shape `(B, H, W)`
- Output: `dict` containing `slots`, `objectness`, `attn_maps`, `props`, `recon_logits`

## Code Layout

- `src/cir_arc/neural/`:
  - `__init__.py`
  - `perception/`:
    - `__init__.py`
    - `embedding.py`
    - `cnn_stem.py`
    - `slot_attention.py`
    - `property_heads.py`
    - `reconstruction.py`
  - `losses/`:
    - `__init__.py`
    - `matching.py`
    - `reconstruction.py`
    - `property.py`
    - `diversity.py`
  - `training/`:
    - `__init__.py`
    - `trainer.py`
    - `dataset.py`
  - `evaluation/`:
    - `__init__.py`
    - `perception_metrics.py`
- `configs/`:
  - `phase2.yaml`
- `tests/neural/`:
  - `__init__.py`
  - `test_embedding.py`
  - `test_cnn_stem.py`
  - `test_slot_attention.py`
  - `test_property_heads.py`
  - `test_reconstruction.py`
  - `test_losses.py`
  - `test_dataset.py`
  - `test_trainer.py`
  - `test_metrics.py`
  - `test_e2e_acceptance.py`
