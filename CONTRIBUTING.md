# Contributing to CIR-ARC

## Module Architecture

```
core/       → Grid, ArcTask, ArcObject (shared data types)
dsl/        → Transformation primitives and composition
generators/ → Synthetic task generators
eval/       → Evaluator, benchmark runner
baselines/  → Random and heuristic agents
scripts/    → CLI entry points
tests/      → Unit and integration tests
data/       → Generated and external datasets (NEVER committed to git)
```

## Key Conventions

### 1. Grid Format
All grids are `numpy.int8` arrays of shape `(H, W)` with values `0-9`.
Use `core.grid.Grid` — never create your own grid representation.

### 2. Task Format
All tasks use `core.task.ArcTask`. Synthetic tasks MUST include:
- `source = "synthetic"`
- `rule_type` (matching a key in `RULE_REGISTRY`)
- `rule_params` (the exact params used)
- `difficulty` (1 = single rule, 2 = two-rule composition)

### 3. Agent Interface
All agents must implement:
```python
class MyAgent:
    name: str = "my_agent"
    def predict(self, task: ArcTask) -> List[Grid]: ...
```

### 4. Seed Policy
See `SEEDS.md`. Never use global seeds. Always pass `np.random.default_rng(seed)` explicitly.

### 5. Testing
- Write tests before code.
- Run `make test` before every PR.
- Run `make leakage` after any data regeneration.

### 6. Data
- Never commit generated data to git.
- All generated data lives under `data/synthetic/`.
- Use `make generate` to regenerate.
