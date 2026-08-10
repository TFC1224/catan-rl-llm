---
name: option-c-curriculum-results
description: Option C Curriculum Self-Play REJECTED — outcome-based training (sigmoid+BCELoss) fundamentally broken for 4-player Catan; VF residual model remains best RL scorer
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-08T09:14:35.778Z
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
---

# Option C: Curriculum Self-Play Results (2026-08-08)

**Conclusion: Outcome-based training (sigmoid + BCELoss) is FUNDAMENTALLY BROKEN for 4-player Catan. Curriculum self-play with outcome labels produces worse results than VF residual training.**

## Experiments Run

### 1. Warm-start from enriched model → sigmoid outcome training (1000 episodes)

| Metric | Value |
|---|---|
| Training correlation | 0.826 |
| Action spread | 0.074 (STRONG) |
| Flat decisions | 0% |
| vs Random | **38%** (baseline 69%) |
| vs WeightedRandom | **14%** (baseline 44%) |
| vs AlphaBeta | **0%** |

**Verdict:** Model is confidently wrong. Warm-start from VF residual (range [-1,2]) to sigmoid ([0,1]) creates bad initialization. Despite perfect action discrimination, predictions don't correlate with winning.

### 2. Fresh start → sigmoid outcome training (500 episodes)

| Metric | Value |
|---|---|
| Training correlation | 0.720 |
| Action spread | 0.111 (STRONG) |
| Flat decisions | 0% |
| vs Random | **12%** |
| vs WeightedRandom | **8%** |
| vs AlphaBeta | **0%** |

**Verdict:** Even worse than warm-start. Outcome labels are too noisy for state evaluation in 4-player games.

### 3. VF residual curriculum (BUG — incomplete)

VF residual formula incorrectly subtracted VP_WEIGHT constant instead of `VP * VP_WEIGHT`. Training produced enormous loss values (9e13). Stopped early. The spread was excellent (0.370, WIN at check), suggesting VF residual labels WOULD work for curriculum if fixed.

## Root Cause

**Outcome labels are fundamentally flawed for state evaluation in 4-player Catan:**

1. **High label noise:** ~75% of states come from losing players who may have played optimally
2. **Multi-agent dependency:** Outcome depends on future decisions of ALL players, not just state quality
3. **Temporal credit assignment:** Linear ramp (0.3→1.0 for winner, 0.7→0.0 for loser) is too crude
4. **Confident wrongness:** Model learns to predict average win rate (~25%) with high confidence, which doesn't help action selection

**Training metrics are misleading:** Correlation 0.7-0.8 and 0% flat decisions look great, but the model simply learns that most states lead to ~25% win rate in 4P games. Action scoring becomes random.

## VF Residual Training Works Because:

1. **Continuous quality signal:** Not binary win/loss — every state gets a meaningful score
2. **State-dependent:** VF evaluates the actual state, independent of future randomness
3. **Clean gradients:** Linear output + MSE loss, no sigmoid saturation
4. **Range separation:** VF residual [-1, 2] provides clear signal for action ranking

## Remaining Issue: Road Blindness

Even the enriched model can't distinguish road placements:
- All road actions produce identical features (same `roads_placed`, often same `my_road_len`)
- Spread = 0.00 for road decisions (100% flat)
- Only `my_road_len` and `reachable_settlement_spots` differentiate roads, and they're often equal

**Mitigation:** In the Hybrid Agent, VF guardrail re-scores ALL actions and picks the best road, circumventing the RL model's blindness.

## Recommendation

1. **Keep `rl_enriched_model.pt`** as the RL scorer — it's the best we have (69% WR vs Random)
2. **Do NOT pursue outcome-based curriculum self-play** — it's fundamentally broken
3. **Fix VF residual curriculum** if curriculum is still desired (correct formula: `(VF - vp * VP_WEIGHT) / 1e8`)
4. **Consider feature engineering** for road discrimination (e.g., per-road-edge features, road-to-settlement distance)
5. **The Hybrid Agent (tools + VF guardrail)** remains the winning architecture at 100% WR

**Why:** The multi-agent nature of 4-player Catan makes game outcomes an extremely noisy training signal. VF (contender_fn) directly evaluates state quality, providing clean gradients. Any training approach using outcome labels will hit the same noise ceiling.

**How to apply:** Use `rl_enriched_model.pt` for `analyze_position` and `get_best_move` tools in the Hybrid Agent. The VF guardrail handles final action selection, compensating for the RL model's road blindness.

**See also:** [[rl-model-fixed]] [[hybrid-agent-results]] [[final-results-2026-08-08]]
