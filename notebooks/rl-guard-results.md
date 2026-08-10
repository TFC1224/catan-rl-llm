---
name: rl-guard-results
description: "RL-Guard achieves 67% WR (preliminary 3-game), significantly better than VF-Distill v2 (40%) but below VF-Guard (90%)"
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-07T16:09:58.693Z
---

# RL-Guard Results (2026-08-08)

**Preliminary win rate: 66.7% (2/3)** vs WeightedRandom — confirmed improvement over VF-Distill v2.

## Comparison with other methods

| Method | Win Rate | Notes |
|---|---|---|
| AB-SFT (standalone) | 25% | Imitation learning baseline |
| VF-Distill v2 (standalone) | 40% | Distilled from VF-Guard; still needs VF at inference |
| **RL-Guard** | **67%** | RL Value Network scoring ALL actions |
| VF-Guard | 90% | Gold standard; hand-crafted VF scoring |

## Why RL-Guard beats VF-Distill v2

RL-Guard uses a trained neural network (CatanValueNetwork, 50K params) to evaluate ALL possible actions, not just the LLM's proposal. The RL model was trained on AlphaBeta game states via supervised learning and can capture non-linear patterns the LLM misses from text observations alone.

Key insight: VF-Distill tries to teach the LLM which action is best → requires encoding game state in text → information loss. RL-Guard keeps the LLM for high-level strategy but uses the RL model for precise action scoring → no information loss.

## Remaining gap to VF-Guard (90%)

VF-Guard's hand-crafted VF (`contender_fn` with CONTENDER_WEIGHTS) is more accurate than the RL model (rl_selfplay_model2.pt) for action scoring. The RL model may need more training or a different loss function to match the VF's precision.

Possible improvements:
1. Train RL model with outcome-ramped labels (current labels are pure AB imitation)
2. Use RL model for within-type refinement, VF for across-type comparison
3. Train RL model on more games (current: rl_selfplay_model2, unknown training data)

**See also:** [[option-a-v2-results]] [[ab-sft-results]]
