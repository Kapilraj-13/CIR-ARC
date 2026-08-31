# Project: CIR-ARC Phase 3

## Architecture
CIR-ARC Phase 3 is an interactive, object-centric neuro-symbolic framework for ARC-AGI-3 environments, fusing:
1. **Interactive Environment Harness (`src/cir_arc/environment/`)**: Adapts `arcengine` / `rc_agi` APIs, manages multi-layer grids up to 64x64 with 16 discrete colors, 8 action types (discrete 0-5, 7, parameterized click 6), frame hashing, zero false-positive mutation tracking, and session recording.
2. **Temporal Object-Centric Perception (`src/cir_arc/neural/perception/` & `src/cir_arc/neural/temporal/`)**: 16-color Slot Attention adapter with dynamic 64x64 multi-scale patch stem and `TemporalSlotTracker` (Hungarian association, kinematic trajectory history, property mutations, >= 90% slot persistence).
3. **Multi-Agent Active Probing & Inspector (`src/cir_arc/probing/`)**: Non-random exploration extracting Action Effect Matrices (AEM), Dynamic Object Catalogs, Pixel Resource/HUD indicators, and Game Phase State Machines (GSM).
4. **Neuro-Symbolic Hypothesis Inducer & DSL World Model (`src/cir_arc/hypothesis/` & `src/cir_arc/dsl/`)**: Causal rule induction from frame/slot deltas into parameterized transition rules and interactive DSL forward simulation primitives.
5. **Hierarchical Planner & Solving Runtime (`src/cir_arc/solving/`)**: LangGraph cognitive loops (Perception -> Memory -> Hypothesis -> Plan -> Action Dispatch), A*/BFS/CodeAgent solvers, `.recording.jsonl` session logging, scorecard tracking, and AgentOps telemetry.
6. **Verification & Benchmark Harness (`tests/e2e/`, `tests/benchmarks/`)**: 5-tier verification suite, mock session replay (`ls20`), and performance benchmark evaluation.

```
                    ┌────────────────────────┐
                    │  ARC-AGI-3 Environment │
                    │ (arcengine / MockEngine)│
                    └───────────┬────────────┘
                                │ FrameData (MultiLayerGrid, 64x64, 16 colors)
                                ▼
                    ┌────────────────────────┐
                    │ Neural Slot Attention  │
                    │  & Temporal Tracker    │
                    └───────────┬────────────┘
                                │ Tracked Slots, Centroids, Mutations
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
┌─────────────────┐   ┌─────────────────┐      ┌─────────────────┐
│ Active Prober   │   │ Causal Inducer  │      │ DSL World Model │
│ & Inspector     │──▶│ & Rule Grammar  │─────▶│ & Forward Sim   │
└─────────────────┘   └─────────────────┘      └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ LangGraph /     │
                                               │ Algorithmic     │
                                               │ Solver Runtime  │
                                               └────────┬────────┘
                                                        │ ActionInput
                                                        ▼
                                               ┌─────────────────┐
                                               │ Action Dispatch │
                                               │ & Session Log   │
                                               └─────────────────┘
```

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1.1 | 8-Action Space Adapter | Support discrete actions 0-5, 7 and parameterized click 6 (x, y coords + reasoning payload) with parameter validation | M1 | ORIGINAL_REQUEST §R1 |
| F1.2 | Multi-Layer Grid (64x64, 16 colors) | Multi-layer grid container up to 64x64, 16 discrete colors (0-15), layer compositing, and SHA-256 frame hashing | M1 | ORIGINAL_REQUEST §R1 |
| F1.3 | State Delta & Object Extraction | `find_objects()` multi-layer connected component extraction and `compute_state_delta()` with zero false-positive mutations | M1 | ORIGINAL_REQUEST §R1 |
| F1.4 | Environment Engine & Adapters | `BaseEnvironment`, `MockEngine`, `RCEngineAdapter` (`arcengine` integration) with step, reset, and lifecycle tracking | M1 | ORIGINAL_REQUEST §R1 |
| F1.5 | Session Recording & Playback | Session recorder writing `.recording.jsonl` files and `PlaybackAgent` for deterministic replay | M1 | ORIGINAL_REQUEST §R1 |
| F2.1 | 16-Color Palette & Embedding | `ColorEmbedding` upgraded to 17 tokens (16 colors 0-15 + 1 mask token 16), property heads, and loss updates | M2 | ORIGINAL_REQUEST §R2 |
| F2.2 | Dynamic 64x64 Spatial Scaling | `MultiScalePatchStem` with 2x2 patch stem for 64x64 resolution and dynamic coordinate decoders | M2 | ORIGINAL_REQUEST §R2 |
| F2.3 | Multi-Layer Depth Attribution | `layer_head` ($z$-index) in `PropertyHeads` and layer-aware slot decoding | M2 | ORIGINAL_REQUEST §R2 |
| F2.4 | Temporal Slot Tracking Engine | `TemporalSlotTracker` with Hungarian inter-frame track association, cosine similarity, centroid distance, attention IoU | M2 | ORIGINAL_REQUEST §R2 |
| F2.5 | Kinematics & Mutation Detection | Centroid trajectory history, velocity vectors, 8-way motion directions, and property mutation detection ($\ge 90\%$ slot persistence) | M2 | ORIGINAL_REQUEST §R2 |
| F2.6 | Recurrent Slot Warm-Starting | `SlotAttention` accepting optional `prev_slots` query priors across sequential steps $t \to t+1$ | M2 | ORIGINAL_REQUEST §R2 |
| F3.1 | Action Effect Matrix (AEM) | Extraction and maintenance of per-action effect matrices, reversibility, preconditions, and delta distributions | M3 | ORIGINAL_REQUEST §R3 |
| F3.2 | Dynamic Object Catalog | Object archetype classification, controllability detection (player identification), and affordances | M3 | ORIGINAL_REQUEST §R3 |
| F3.3 | Pixel Resource & HUD Inspector | Detection of dynamic counters, resource indicators, score/life displays, and boundary zones | M3 | ORIGINAL_REQUEST §R3 |
| F3.4 | Game Phase State Machine (GSM) | Phase tracking across title screens, gameplay, stage transitions, win/loss states | M3 | ORIGINAL_REQUEST §R3 |
| F3.5 | Non-Random Inspector Agent | Exploratory probing agent utilizing epistemic uncertainty and safe reversion actions | M3 | ORIGINAL_REQUEST §R3 |
| F4.1 | Frame/Slot Delta Mapping | Conversion of raw state deltas and temporal slot mutations into symbolic delta representations | M4 | ORIGINAL_REQUEST §R4 |
| F4.2 | Parameterized Transition Grammar | Rule grammar modeling kinematics ($\Delta x, \Delta y$), collisions, triggers, color mutations, and autonomous dynamics | M4 | ORIGINAL_REQUEST §R4 |
| F4.3 | Causal Hypothesis Induction | `HypothesisInductionEngine` ranking, corroborating, and pruning candidate transition rules with confidence metrics | M4 | ORIGINAL_REQUEST §R4 |
| F4.4 | Interactive DSL Forward Model | DSL simulation engine executing forward state predictions for candidate action sequences | M4 | ORIGINAL_REQUEST §R4 |
| F4.5 | Symbolic Macro-Actions | Macro-action primitives (`macro_move_until_obstacle`, `macro_interact_target`, `macro_collect_all`) | M4 | ORIGINAL_REQUEST §R4 |
| F5.1 | LangGraph Cognitive Loop | Perception -> Memory/Scratchpad -> Hypothesis -> Symbolic Plan -> Action Dispatch state graph | M5 | ORIGINAL_REQUEST §R5 |
| F5.2 | Algorithmic Solvers (A*/BFS) | A* and BFS search solvers planning over the DSL world model to reach goal states | M5 | ORIGINAL_REQUEST §R5 |
| F5.3 | SmolAgents / CodeAgent Solver | Code-generation solver synthesizing executable Python action sequences with sandbox guards | M5 | ORIGINAL_REQUEST §R5 |
| F5.4 | Solving Runtime & Scorecards | Solving execution loop managing `MAX_ACTIONS` limits, scorecard tracking, and session logging | M5 | ORIGINAL_REQUEST §R5 |
| F5.5 | AgentOps Telemetry & Observability | Structured event logging, telemetry traces, and performance metric export | M5 | ORIGINAL_REQUEST §R5 |
| F6.1 | E2E Test Infrastructure | Test runner, pass/fail assertion suite, and multi-tier harness | M6 | ORIGINAL_REQUEST §R6 |
| F6.2 | Mock ARC-AGI-3 Game Sessions | Deterministic replay of recorded ARC-AGI-3 sessions (e.g. `ls20`) | M6 | ORIGINAL_REQUEST §R6 |
| F6.3 | Tier 1-4 Verification Suite | Comprehensive opaque-box test cases covering features, boundaries, pairwise combinations, and real workloads | M6 | ORIGINAL_REQUEST §R6 |
| F6.4 | Benchmark Metrics Suite | Automated evaluation of win rate, action efficiency, perception F1, and rule induction precision | M6 | ORIGINAL_REQUEST §R6 |
| F6.5 | Tier 5 Adversarial Coverage Hardening | White-box adversarial testing, boundary perturbation, and stress evaluation | M-Final | ORIGINAL_REQUEST §R6 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Environment & Multi-Layer Integration (R1) | F1.1, F1.2, F1.3, F1.4, F1.5 (`src/cir_arc/environment/`, `src/cir_arc/recording/`) | none | PLANNED |
| M2 | Temporal Slot Perception & Tracking (R2) | F2.1, F2.2, F2.3, F2.4, F2.5, F2.6 (`src/cir_arc/neural/perception/`, `src/cir_arc/neural/temporal/`) | M1 | PLANNED |
| M3 | Active Probing & Environment Inspector (R3) | F3.1, F3.2, F3.3, F3.4, F3.5 (`src/cir_arc/probing/`) | M1, M2 | PLANNED |
| M4 | Hypothesis Inducer & DSL World Model (R4) | F4.1, F4.2, F4.3, F4.4, F4.5 (`src/cir_arc/hypothesis/`, `src/cir_arc/dsl/`) | M1, M2, M3 | PLANNED |
| M5 | Hierarchical Planner & Solving Runtime (R5) | F5.1, F5.2, F5.3, F5.4, F5.5 (`src/cir_arc/solving/`) | M1, M2, M3, M4 | PLANNED |
| M6 | E2E Testing Track (R6) | F6.1, F6.2, F6.3, F6.4 (`tests/e2e/`, `tests/benchmarks/`, `TEST_READY.md`) | M1 | PLANNED (Parallel Track) |
| M-Final | Integration Acceptance & Tier 5 Hardening | F6.5, 100% pass on Tiers 1-4 + Tier 5 adversarial coverage hardening | M1, M2, M3, M4, M5, M6 | PLANNED |

---

## Interface Contracts

### Environment ↔ Perception
```python
# MultiLayerGrid 
class MultiLayerGrid:
    layers: list[np.ndarray]  # shape: (L, H, W), dtype: uint8 in [0, 15]
    height: int               # <= 64
    width: int                # <= 64
    num_layers: int
    def composite(self) -> np.ndarray: ...
    def hash(self) -> str: ...

# FrameData
class FrameData:
    game_id: str
    grid: MultiLayerGrid
    state: GameState  # NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER
    levels_completed: int
    win_levels: int
    action_input: Optional[Action]
    available_actions: list[int]
    step_count: int
```

### Perception ↔ Probing & Hypothesis
```python
# TrackedSlot
class TrackedSlot:
    track_id: int
    slot_embedding: torch.Tensor  # shape: (slot_dim,)
    predicted_color: int          # 0..15
    predicted_shape: int
    predicted_size: float
    centroid: tuple[float, float]
    velocity: tuple[float, float]
    motion_direction: Optional[str] # N, S, E, W, NE, NW, SE, SW, STILL
    layer_index: int
    attention_mask: np.ndarray    # shape: (H, W), float in [0, 1]
    lifecycle_state: str          # SPAWNED, ACTIVE, DESTROYED

# StateDelta
class StateDelta:
    frame_hash_before: str
    frame_hash_after: str
    action_taken: Action
    has_mutation: bool
    pixel_diff_count: int
    mutations: list[PropertyMutation]
    spawned_tracks: list[TrackedSlot]
    destroyed_tracks: list[TrackedSlot]
    moved_tracks: list[TrackedSlot]
```

### Probing / Hypothesis ↔ Solving Runtime
```python
# EnvironmentProfile (from Inspector Agent)
class EnvironmentProfile:
    action_effect_matrix: dict[int, ActionEffect]
    controllable_tracks: list[int]
    object_catalog: list[ObjectArchetype]
    pixel_resources: list[ResourceIndicator]
    state_machine: GameStateMachine

# TransitionRule (from Hypothesis Inducer)
class TransitionRule:
    rule_id: str
    action_trigger: Optional[int]
    condition: Callable[[WorldState], bool]
    effect: Callable[[WorldState], WorldState]
    confidence: float
```

---

## Code Layout

```
src/cir_arc/
├── core/                   # Existing Phase 1 grid, objects, tasks (preserved)
├── dsl/                    # Phase 1 DSL + Phase 3 interactive world model & macro-actions
│   ├── primitives.py       # Phase 1 static rule registry
│   ├── compose.py          # Rule composition
│   ├── interactive_primitives.py # Macro-actions & step transitions
│   └── world_model.py      # DSL forward simulation world model
├── neural/                 # Neural perception subsystem
│   ├── perception/         # Embeddings, patch stems, slot attention, decoders (16 colors, 64x64)
│   ├── temporal/           # TemporalSlotTracker, Hungarian matcher, kinematic trajectories
│   └── training/           # PerceptionModel & datasets
├── environment/            # ARC-AGI-3 environment harness (R1)
│   ├── actions.py          # ActionType (0-7), Action dataclass
│   ├── frame.py            # MultiLayerGrid, FrameData, frame hashing
│   ├── state_delta.py      # Zero false-positive delta computation, find_objects
│   ├── base.py             # BaseEnvironment abstract interface
│   ├── mock_engine.py      # Simulated ARC-AGI-3 deterministic environment
│   └── rc_adapter.py       # arcengine / rc_agi native integration
├── probing/                # Active probing & environment inspector (R3)
│   ├── action_matrix.py    # Action Effect Matrix (AEM)
│   ├── object_catalog.py   # Dynamic object catalog & affordances
│   ├── resource_inspector.py # Pixel resource & HUD inspector
│   ├── state_machine.py    # Game phase state machine (GSM)
│   └── inspector_agent.py  # Active non-random probing agent
├── hypothesis/             # Neuro-symbolic hypothesis induction (R4)
│   ├── delta_mapper.py     # Frame & slot delta mapping
│   ├── transition_grammar.py # Causal transition rule grammar
│   └── induction_engine.py # Hypothesis ranking & pruning engine
├── solving/                # Hierarchical solving runtime (R5)
│   ├── cognitive_loop.py   # LangGraph cognitive state graph
│   ├── search_solvers.py   # A* and BFS search solvers
│   ├── code_agent.py       # SmolAgents / CodeAgent solver
│   ├── runtime.py          # Solving runtime, scorecard, MAX_ACTIONS
│   └── telemetry.py        # AgentOps observability & traces
└── recording/              # Session recording & playback (R1, R5)
    ├── recorder.py         # .recording.jsonl session logger
    └── playback.py         # PlaybackAgent replay validator

tests/
├── environment/            # R1 unit & integration tests
├── neural/                 # R2 perception & temporal slot tests
├── probing/                # R3 active probing tests
├── hypothesis/             # R4 hypothesis induction tests
├── solving/                # R5 solving runtime tests
├── e2e/                    # R6 E2E Test Suite (Tiers 1-4)
└── benchmarks/             # R6 Benchmark replay & evaluation suite
```
