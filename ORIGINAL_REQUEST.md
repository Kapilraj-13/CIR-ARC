# Original User Request

## 2026-08-28T19:27:16Z

Build CIR-ARC Phase 3: an interactive, object-centric multi-agent perception, hypothesis induction, and solving framework for ARC-AGI-3 environments, fusing Slot Attention perception, active probing exploration, and symbolic DSL macro-actions.

Working directory: d:/AI_Projects/CIR-ARC
Integrity mode: development

## Requirements

### R1. ARC-AGI-3 Environment Harness & Multi-Layer Grid Integration
Implement an interactive environment adapter and state representation system supporting ARC-AGI-3 game semantics (rcengine / rc_agi API), handling multi-layer grids up to 64x64 with 16 discrete colors, action spaces (discrete 0-5, 7 and parameterized click 6), frame hashing, and session recording.

### R2. Temporal Object-Centric Perception & Slot Attention Adapter
Extend CIR-ARC's Slot Attention neural perception pipeline to handle ARC-AGI-3 dimensions (16 colors + mask token, 64x64 resolution via multi-scale/patch stem) and temporal slot tracking across frame transitions  \to t+1$ to track object persistence, centroid trajectories, property mutations, and layer dynamics.

### R3. Multi-Agent Active Probing & Environment Inspector
Implement a modular environment inspection and exploratory probing agent system that passively explores games to extract action effect matrices, object catalogs, dynamic resource indicators, and game phase state machines without guessing random actions.

### R4. Neuro-Symbolic Hypothesis Inducer & DSL World Model
Build a causal hypothesis induction engine that maps observed frame deltas to parameterized transition rules and uses CIR-ARC DSL primitives as forward simulation models and macro-actions for candidate validation.

### R5. Hierarchical Agent Planner & Solving Runtime
Develop a solving runtime supporting LangGraph-based cognitive loops (Perception -> Memory/Scratchpad -> Hypothesis -> Symbolic Plan -> Action Dispatch) and SmolAgents/CodeAgent algorithmic solvers (A*, BFS, rule-based heuristics) with full session replay, scorecard tracking, and AgentOps observability.

### R6. Verification Test Suite & Benchmark Harness
Provide end-to-end integration tests, mock environment replay tests against recorded ARC-AGI-3 sessions (e.g., ls20), active probe verification, and benchmark evaluation metrics (win rate, action efficiency, perception F1, rule induction precision).

## Acceptance Criteria

### Environment & State Handling
- [ ] Environment adapter correctly steps through all 8 ARC-AGI-3 action types and returns structured FrameData with multi-layer 64x64 support.
- [ ] Frame hashing and object extraction (ind_objects) uniquely identify connected components and state deltas with zero false-positive mutations on identical frames.

### Perception & Temporal Slots
- [ ] Perception model accepts 16 colors and dynamic grid shapes (up to 64x64) without out-of-bounds embedding or dimension mismatches.
- [ ] Temporal slot tracking maintains object identity matching across sequential frames with $\ge 90\%$ slot persistence accuracy on moving/transforming objects.

### Probing & World Model
- [ ] Inspector agent produces a structured EnvironmentProfile detailing active actions, detected objects, pixel resources, and phase transitions.
- [ ] Hypothesis engine successfully infers action effects (e.g. movement vectors, toggle triggers) from probe observation sequences.

### Solving & Replay
- [ ] Hierarchical agent solver completes game runs without unhandled exceptions or infinite loops, respecting MAX_ACTIONS limits.
- [ ] Session recorder produces compliant .recording.jsonl files replayable via the Playback agent.
- [ ] All unit and integration test suites pass with \%$ green status.
