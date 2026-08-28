# Seed Policy

All random operations use explicit integer seeds.
Seeds are NEVER set globally — always passed explicitly.

## Allocation

| Seed | Purpose |
|------|-------------------------------------------|
| 42   | Grid content generation (object shapes, colors) |
| 100  | Rule parameter sampling (which axis, which colors) |
| 200  | Dataset split (which tasks go to train vs held_out) |
| 300  | Evaluation sampling (which held_out tasks to use in benchmark) |
| 999  | Adversarial task generation (Phase 2+) |

## Rules

- **Never use**: `random.seed()` or `np.random.seed()` at module level.
- **Always use**: `np.random.default_rng(seed)` passed as argument.
- **Rationale**: Global seeds break when test order changes or parallel jobs run simultaneously.
