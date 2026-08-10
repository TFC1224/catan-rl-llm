---
name: vf-guard-discovery
description: VF-Guard achieves 90% win rate by combining LLM with Value Function scoring at inference time
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-07T10:57:40.631Z
---

# VF-Guard Discovery

**Date:** 2026-08-07

## What we found

VF-Guard: LLM proposes action types, Value Function scores all valid actions (milliseconds), picks the best. This achieves **90% win rate vs WeightedRandom** (9/10 games, 4P) — matching pure VF's upper bound.

## How it works

```
1. LLM generates action proposal (strategic understanding)
2. VF scores ALL valid actions (tactical optimization, CPU milliseconds)
3. If VF prefers a different action → override LLM
4. ~50% of non-trivial decisions get overridden by VF
```

Overrides are mostly **within-type refinement** (which settlement node, which road position) rather than changing action types. LLM picks the right action category; VF optimizes the specific choice.

## Key metrics

| Method | Win Rate | Speed |
|---|---|---|
| SFT (AlphaBeta imitation) | 25% | 2 min/game |
| VF only | 90% (3P) | 0.1 sec/game |
| VF-Guard | 90% (9/10, 4P) | 2 min/game |

## Why this matters

- **GRPO is unnecessary.** VF-Guard already matches VF's ceiling with zero RL training.
- **VF scoring is 1000x faster than simulation.** Makes RL training feasible using VF as reward proxy.
- **Path forward: distillation.** Train model to internalize VF preferences via SFT on VF-Guard decisions.

**See also:** [[multi-option-strategy-plan]]
**Artifacts:** `scripts/eval_vf_guard.py`
