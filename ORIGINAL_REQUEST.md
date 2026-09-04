# Original User Request

## 2026-09-01T06:44:14Z

Implement Phase 4 (Hybrid Neuro-Symbolic Reasoning, Autonomous Goal Induction, and 100-Environment Procedural Benchmark Suite) for the CIR-ARC project.

Working directory: D:\AI_Projects\CIR-ARC
Integrity mode: development

## Requirements

### R1. Procedural 100-Environment Benchmark Suite
Create src/cir_arc/generators/interactive_suite.py generating 100 distinct interactive environments spanning 10 core ARC-AGI-3 puzzle mechanics with 10 difficulty tiers each:
1. Mirrored Symmetry & Coupled Kinematics (like m0r0)
2. Gravity & Falling Trajectories (like r25, c33)
3. Pressure-Plate Switches & Sliding Gates (like p35, pf33)
4. Key & Lock Inventory Mazes (like 	n36, su15)
5. Inertia & Frictionless Ice Sliding (like 	r87, sp80)
6. Portal & Teleportation Chambers (like 
a86, sk48)
7. Sokoban & Block Pushing (like cl78, iu86)
8. Trail & Floor Pattern Painting (like wa30, 	u93)
9. Laser Optics & Reflective Mirrors (like pk90, pu71)
10. Reactive Dynamic Mazes with Moving Hazards (like kq74, jn23)
All generated environments must implement the BaseEnvironment protocol (step, eset, current_observation, enumerate_actions, is_terminal).

### R2. Symbolic State Synthesizer & Causal Inducer
Create src/cir_arc/reasoning/symbolic_state.py and src/cir_arc/reasoning/hypothesis_inducer.py that:
- Decompose raw perception grids and temporal slot representations into structured spatial entities (Centroids, Bounding Boxes, Symmetries, Controllability).
- Construct an empirical action-effect transition matrix from active probing deltas.
- Formulate zero-reward goal hypotheses (Convergence/Meeting Point, Gate Clearance, Pattern Matching, Switch Activation).

### R3. Phase 4 Local & LLM-Guided Hybrid Reasoner
Create src/cir_arc/reasoning/llm_reasoner.py:
- Fast, fully offline deterministic symbolic rule engine for instant reasoning (zero external API dependencies, <5ms per step).
- Extensible LLM interface (supporting Gemini / OpenAI / Anthropic if API keys are set).
- Sub-goal decomposition and dynamic failure replanner that re-induces hypotheses upon unexpected collisions or dead-ends.

### R4. Benchmark Runner & Scorecard Pipeline
Create scripts/benchmark_100_interactive.py that:
- Runs the autonomous cognitive agent across all 100 procedural environments.
- Outputs detailed scorecards: Win Rate per mechanic, total action counts, latency, and recovery rates.
- Integrates seamlessly with existing enchmark_arc_agi3.py for official games.

### R5. Comprehensive Test Suite
Create unit and integration tests under 	ests/test_interactive_suite.py, 	ests/test_symbolic_state.py, 	ests/test_hypothesis_inducer.py, and 	ests/test_llm_reasoner.py ensuring at least 25 new tests, all passing with zero regressions on existing 339 tests.

## Acceptance Criteria

### Correctness & Robustness
- [ ] All 100 procedural environment instances generate cleanly, are strictly solvable, and conform to BaseEnvironment.
- [ ] Symbolic state synthesizer correctly identifies entity kinematics and coupled multi-agent symmetries.
- [ ] Goal inducer autonomously identifies win conditions across all 10 puzzle mechanics without hardcoded rules.
- [ ] scripts/benchmark_100_interactive.py executes end-to-end and outputs a formatted markdown/table summary.
- [ ] All existing 339 tests + new Phase 4 tests pass (pytest -q exits code 0 with 360+ passing tests).
