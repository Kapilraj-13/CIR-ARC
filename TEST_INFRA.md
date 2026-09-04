# E2E Test Infra: CIR-ARC Phase 4

## Test Philosophy
- Opaque-box, requirement-driven, testing all 10 ARC-AGI-3 puzzle mechanics, symbolic state synthesis, causal goal induction, hybrid reasoning, and benchmark scorecard generation.
- Methodology: Category-Partition + Boundary Value Analysis + Cross-Feature Interactions + Real-World Workload Testing.

## Feature Inventory & Test Coverage
| # | Feature | Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Scenario) |
|---|---------|--------|:----------------:|:-----------------:|:----------------------:|:-----------------:|
| F1 | 10-Mechanic Environments | ORIGINAL_REQUEST §R1 | >=5 | >=5 | ✓ | ✓ |
| F2 | 10-Tier Difficulty Progression | ORIGINAL_REQUEST §R1 | >=5 | >=5 | ✓ | ✓ |
| F3 | Symbolic State Synthesizer | ORIGINAL_REQUEST §R2 | >=5 | >=5 | ✓ | ✓ |
| F4 | Coupled Kinematics / Symmetries | ORIGINAL_REQUEST §R2 | >=5 | >=5 | ✓ | ✓ |
| F5 | Empirical Transition Matrix | ORIGINAL_REQUEST §R2 | >=5 | >=5 | ✓ | ✓ |
| F6 | Zero-Reward Goal Induction | ORIGINAL_REQUEST §R2 | >=5 | >=5 | ✓ | ✓ |
| F7 | Deterministic Symbolic Rule Engine | ORIGINAL_REQUEST §R3 | >=5 | >=5 | ✓ | ✓ |
| F8 | LLM Provider Interface | ORIGINAL_REQUEST §R3 | >=5 | >=5 | ✓ | ✓ |
| F9 | Dynamic Sub-goal Replanner | ORIGINAL_REQUEST §R3 | >=5 | >=5 | ✓ | ✓ |
| F10 | 100-Env Benchmark Runner | ORIGINAL_REQUEST §R4 | >=5 | >=5 | ✓ | ✓ |
| F11 | Official Benchmark Integration | ORIGINAL_REQUEST §R4 | >=5 | >=5 | ✓ | ✓ |
| F12 | Comprehensive Test Suite | ORIGINAL_REQUEST §R5 | >=5 | >=5 | ✓ | ✓ |

## Test Architecture
- Test runner: `uv run pytest`
- Test files:
  - `tests/test_interactive_suite.py`: Validates all 10 mechanics, 10 tiers, reset/step/is_terminal, and BaseEnvironment adherence.
  - `tests/test_symbolic_state.py`: Validates entity extraction, centroids, bboxes, symmetries, coupled multi-agent kinematics, and passable terrain.
  - `tests/test_hypothesis_inducer.py`: Validates action-effect matrix $P(\Delta s \mid s, a)$ updates and causal zero-reward goal hypotheses formulation across mechanics.
  - `tests/test_llm_reasoner.py`: Validates deterministic rule engine (<5ms latency), LLM interface mock/local fallback, and dynamic failure replanner.

## Coverage Goals
- Maintain 339 existing passing tests.
- Add >=25 new unit & integration tests (target >=36 new tests -> 375+ total).
- 100% test pass rate with 0 regressions.
