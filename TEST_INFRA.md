# E2E Test Infra: CIR-ARC Phase 2 Neural Perception

## Test Philosophy
- Opaque-box, requirement-driven. Derives from ORIGINAL_REQUEST.md.
- Verification methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinations + Real-World Workloads + Adversarial Coverage Hardening.

## Feature Inventory & Test Mapping

| # | Feature | Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|--------|:----------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 | ColorEmbedding | R1 | 5 tests | 5 tests | ✓ | ✓ |
| 2 | CNNStem | R1 | 5 tests | 5 tests | ✓ | ✓ |
| 3 | SlotAttention | R1 | 5 tests | 5 tests | ✓ | ✓ |
| 4 | PropertyHeads | R1 | 5 tests | 5 tests | ✓ | ✓ |
| 5 | ReconstructionDecoder | R1 | 5 tests | 5 tests | ✓ | ✓ |
| 6 | HungarianMatching | R2 | 5 tests | 5 tests | ✓ | ✓ |
| 7 | Losses (Recon, Prop, Div) | R2 | 5 tests | 5 tests | ✓ | ✓ |
| 8 | PerceptionModel & Trainer | R3 | 5 tests | 5 tests | ✓ | ✓ |
| 9 | SyntheticDataset & Padding | R3 | 5 tests | 5 tests | ✓ | ✓ |
| 10 | PerceptionMetrics | R4 | 5 tests | 5 tests | ✓ | ✓ |

## Test Architecture
- Test Runner: `pytest -v tests/` and `pytest -q`
- Individual Smoke Tests: `python -m cir_arc.neural.perception.<module>`
- Total Neural Tests Planned: >= 25 tests
- Existing Regression Suite: 167 tests (must all pass)

## Acceptance Verification Criteria
1. Module correctness: All 167 Phase 1 tests pass + >=20 new neural tests pass.
2. Smoke tests: Every module executes its CLI entrypoint with exit code 0.
3. Batch Forward: Batch of 4 random 10x10 grids and heterogeneous batch (5x5, 8x12, 15x15) pass without error.
4. Parameter Count: Total parameters in `[200000, 500000]` (verified ~401,080).
5. Attention Invariant: Slot attention maps sum to 1.0 (±1e-5) across slots dimension.
6. Dataset & Training Step: Load task JSON from `data/synthetic/train/`, complete one forward+backward+step.
7. Checkpoint Round-Trip: Model produces identical outputs after save and load.
8. Configs: `configs/phase2.yaml` is valid and parseable.
