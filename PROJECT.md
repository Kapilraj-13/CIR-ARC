# Project: CIR-ARC Phase 4 (Hybrid Neuro-Symbolic Reasoning, Autonomous Goal Induction, and 100-Environment Procedural Benchmark Suite)

## Architecture
The CIR-ARC Phase 4 architecture extends the neuro-symbolic cognitive architecture with a complete procedural evaluation suite, structured symbolic state synthesis, empirical causal transition induction, an ultra-fast offline deterministic rule engine with optional LLM reasoning, and dynamic replanning.

```
+-----------------------------------------------------------------------------------+
|                        100-Environment Benchmark Suite                             |
|               (10 ARC-AGI-3 Core Mechanics x 10 Difficulty Tiers)                 |
|               src/cir_arc/generators/interactive_suite.py                         |
+----------------------------------------+------------------------------------------+
                                         | Observation / FrameData (MultiLayerGrid)
                                         v
+-----------------------------------------------------------------------------------+
|                        Symbolic State Synthesizer                                 |
|               src/cir_arc/reasoning/symbolic_state.py                             |
|  - Raw Grid & Slot Decomposition -> SymbolicEntity (BBox, Symmetries, Velocity)   |
|  - Coupled Multi-Agent Kinematics & Symmetries Detection                         |
|  - Spatial Relations & Passable Terrain Map                                       |
+----------------------------------------+------------------------------------------+
                                         | Structured SymbolicState
                                         v
+-----------------------------------------------------------------------------------+
|                        Causal Hypothesis Inducer                                  |
|               src/cir_arc/reasoning/hypothesis_inducer.py                         |
|  - Empirical Action-Effect Transition Matrix P(Delta s | s, a)                    |
|  - Zero-Reward Causal Goal Formulation (10 Core Mechanics)                         |
|  - Bayesian Multi-Factor Hypothesis Scoring                                       |
+----------------------------------------+------------------------------------------+
                                         | Ranked Goal Hypotheses & Transition Models
                                         v
+-----------------------------------------------------------------------------------+
|                 Local & LLM-Guided Hybrid Reasoner                                |
|               src/cir_arc/reasoning/llm_reasoner.py                               |
|  - DeterministicRuleEngine (<5ms offline symbolic solver)                         |
|  - LLMProvider Interface (Gemini / OpenAI / Anthropic / Local fallback)           |
|  - SubgoalReplanner (Pre/Post-condition validation, failure replanning)           |
+----------------------------------------+------------------------------------------+
                                         | Action Selection
                                         v
+-----------------------------------------------------------------------------------+
|                        Benchmark Runner & Scorecards                              |
|               scripts/benchmark_100_interactive.py                                |
|  - End-to-End Evaluation across 100 Envs                                          |
|  - Per-mechanic Win Rates, Latencies, Recovery Rates, Actions                     |
+-----------------------------------------------------------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | 10-Mechanic Procedural Environments | 10 distinct interactive mechanics implementing `BaseEnvironment` | M1 | ORIGINAL_REQUEST §R1 |
| F2 | 10-Tier Difficulty Progression | Monotonic scaling across 10 difficulty tiers (100 total envs) with guaranteed solvability | M1 | ORIGINAL_REQUEST §R1 |
| F3 | Symbolic State Synthesizer | Entity decomposition, centroids, bboxes, symmetries, controllability, passable masks | M2 | ORIGINAL_REQUEST §R2 |
| F4 | Coupled Kinematics & Symmetry Detector | Identifies multi-agent coupled symmetries (like m0r0) | M2 | ORIGINAL_REQUEST §R2 |
| F5 | Empirical Transition Matrix | $P(\Delta s \mid s, a)$ mapping from probing transitions | M2 | ORIGINAL_REQUEST §R2 |
| F6 | Zero-Reward Goal Inducer | Autonomous win condition induction across all 10 mechanics | M2 | ORIGINAL_REQUEST §R2 |
| F7 | Deterministic Symbolic Rule Engine | Fast offline deterministic solver (<5ms per step, zero API dependencies) | M3 | ORIGINAL_REQUEST §R3 |
| F8 | Extensible LLM Provider Interface | Optional integration for Gemini, OpenAI, Anthropic with graceful local fallback | M3 | ORIGINAL_REQUEST §R3 |
| F9 | Dynamic Sub-goal Replanner | Collision / dead-end contradiction detection & on-the-fly hypothesis re-induction | M3 | ORIGINAL_REQUEST §R3 |
| F10 | 100-Env Benchmark Runner | Evaluates agent on all 100 envs with formatted scorecards | M4 | ORIGINAL_REQUEST §R4 |
| F11 | Official Benchmark Integration | Compatibility with `scripts/benchmark_arc_agi3.py` | M4 | ORIGINAL_REQUEST §R4 |
| F12 | Comprehensive Test Suite & Zero Regressions | >=25 new passing tests covering R1-R5, maintaining 339 existing tests (total 365+ passing) | M5 | ORIGINAL_REQUEST §R5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Procedural 100-Environment Benchmark Suite | `src/cir_arc/generators/interactive_suite.py` with 10 mechanics x 10 difficulty tiers conforming to `BaseEnvironment` | None | IN_PROGRESS |
| M2 | Symbolic State Synthesizer & Causal Inducer | `src/cir_arc/reasoning/__init__.py`, `src/cir_arc/reasoning/symbolic_state.py`, `src/cir_arc/reasoning/hypothesis_inducer.py` | None | PLANNED |
| M3 | Local Rule Engine & LLM Hybrid Reasoner | `src/cir_arc/reasoning/llm_reasoner.py` (DeterministicRuleEngine, LLMProvider, SubgoalReplanner, HybridReasoner) | M2 | PLANNED |
| M4 | Benchmark Runner & Scorecard Pipeline | `scripts/benchmark_100_interactive.py` | M1, M2, M3 | PLANNED |
| M5 | Comprehensive Test Suite & Coverage Hardening | Test suite in `tests/test_interactive_suite.py`, `tests/test_symbolic_state.py`, `tests/test_hypothesis_inducer.py`, `tests/test_llm_reasoner.py`, E2E test verification, adversarial coverage check | M1, M2, M3, M4 | PLANNED |

## Interface Contracts

### 1. `BaseEnvironment` Protocol (`cir_arc.environment.base`)
- `reset() -> FrameData`
- `step(action: Action) -> FrameData`
- `current_observation() -> Optional[FrameData]`
- `enumerate_actions(frame: Optional[FrameData] = None) -> List[ActionSpec]`
- `is_terminal() -> bool`
- `close() -> None`

### 2. `SymbolicState` & `SymbolicEntity` (`cir_arc.reasoning.symbolic_state`)
- `SymbolicEntity`: `entity_id: int`, `color: int`, `mask: np.ndarray`, `bbox: Tuple[int, int, int, int]`, `centroid: Tuple[float, float]`, `area: int`, `is_controllable: bool`, `velocity: Tuple[float, float]`, `symmetries: List[str]`, `properties: Dict[str, Any]`
- `SymbolicState`: `entities: List[SymbolicEntity]`, `spatial_relations: List[Dict[str, Any]]`, `coupled_pairs: List[Tuple[int, int, str]]`, `passable_mask: np.ndarray`, `raw_grid: np.ndarray`, `step_count: int`
- `SymbolicStateSynthesizer.synthesize(frame: FrameData, prev_state: Optional[SymbolicState] = None) -> SymbolicState`

### 3. `HypothesisInducer` (`cir_arc.reasoning.hypothesis_inducer`)
- `ActionEffectTransitionMatrix.update(state_before: SymbolicState, action: Action, state_after: SymbolicState)`
- `HypothesisInducer.induce_goals(symbolic_state: SymbolicState, transition_matrix: ActionEffectTransitionMatrix) -> List[CausalGoalHypothesis]`

### 4. `HybridReasoner` (`cir_arc.reasoning.llm_reasoner`)
- `DeterministicRuleEngine.solve_step(state: SymbolicState, goal: CausalGoalHypothesis) -> Optional[Action]`
- `LLMProvider.reason(prompt_context: Dict[str, Any]) -> Optional[Dict[str, Any]]`
- `SubgoalReplanner.replan_on_failure(state: SymbolicState, failed_action: Action) -> List[Action]`
- `HybridReasoner.select_action(frame: FrameData) -> Action`

## Code Layout
```
src/cir_arc/
├── environment/
│   ├── base.py
│   ├── actions.py
│   ├── frame.py
│   └── state_delta.py
├── generators/
│   ├── __init__.py
│   ├── single_rule.py
│   ├── composition.py
│   └── interactive_suite.py     # [M1] 10 mechanics x 10 tiers (100 envs)
├── reasoning/
│   ├── __init__.py              # [M2] Package initialization
│   ├── symbolic_state.py        # [M2] Entity decomposition & coupled symmetries
│   ├── hypothesis_inducer.py    # [M2] Transition matrix & zero-reward goal induction
│   └── llm_reasoner.py          # [M3] Deterministic engine, LLM provider, replanner
scripts/
├── benchmark_arc_agi3.py
└── benchmark_100_interactive.py # [M4] 100-Env Benchmark & Scorecard pipeline
tests/
├── test_interactive_suite.py    # [M5] Suite verification across 10 mechanics
├── test_symbolic_state.py       # [M5] Synthesizer & coupled symmetry tests
├── test_hypothesis_inducer.py   # [M5] Causal induction & transition matrix tests
└── test_llm_reasoner.py         # [M5] Deterministic engine, LLM interface, replanner tests
```
