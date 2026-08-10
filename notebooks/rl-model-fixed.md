---
name: rl-model-fixed
description: "RL model fixed with enriched 72-feature extraction + VF residual training — 69% WR vs Random, 44% vs WeightedRandom, 3.1% flat (was 47%)"
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-08T07:41:16.202Z
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
---

# RL Model Fix — Enriched Features + VF Residual Training (2026-08-08)

**Conclusion: RL model is FIXED. 69% WR vs Random (was ~25%), 44% vs WeightedRandom (was 0%), 3.1% flat decisions (was 47%).**

## Root Cause

The original `rl_selfplay_model2.pt` (30 features) couldn't discriminate between similar actions because:
1. All BUILD_ROAD actions produced identical feature vectors (same `roads_placed` count)
2. All MOVE_ROBBER actions produced nearly identical feature vectors
3. The model had 47% flat decisions (spread < 0.001) — couldn't tell actions apart

## Fix: Two Changes

### 1. Enriched Feature Extraction (30 → 72 features)

Added features that CHANGE based on which specific action is taken:
- **Production per resource** (5 features): Changes with robber/settlement placement
- **Robber context** (3 features): Robber position effects — which player, tile value
- **Port access** (4 features): Which ports are owned
- **Opponent detail per player** (18 features): VP, knights, cities, production per opponent
- **Robber resource one-hot** (5 features): What resource is being blocked
- **Road/expansion** (4 features): Tiles owned, avg production per tile
- **Build flags** (4 features): can_build_city/settlement/road/dev_card
- **Hand synergy** (1 feature): Distance to city/settlement completion
- **Threat detection** (2 features): Number of opponents with ≥8 VP, ≥6 VP

### 2. VF Residual Training

Instead of predicting raw VF (dominated by VP, 6 orders of magnitude above other terms), train to predict VF residual:
```
label = (VF - VP * 3e14) / 1e8  # range [-1, 2]
```
This captures production quality, building synergy, port access — the features that distinguish good from bad states at the same VP level.

**Training:** 300 AlphaBeta games → ~100K samples → 500 training steps → MSE loss, linear output (no sigmoid)

**Model:** `rl_enriched_model.pt` — 72-dim input, [256, 128, 64] hidden, linear output, `use_sigmoid=False`

## Results

| Metric | Old (30-feat) | New (72-feat) | Improvement |
|--------|--------------|---------------|-------------|
| vs Random | ~25% | **69.0%** | 2.8x |
| vs WeightedRandom | 0-25% | **44.0%** | 1.8x+ |
| Overall WR | ~25% | **56.5%** | 2.3x |
| Action spread | ~0.018 | **0.0644** | 3.6x |
| Flat decisions | 47% | **3.1%** | 15x reduction |

## Files Changed

- `rl_value_network.py`: Expanded `extract_features()` from 30 to 72 features, added `use_sigmoid` param to `CatanValueNetwork`
- `agent_tools.py`: Fixed `analyze_position()` to handle linear output models (applies sigmoid manually), fixed `production` variable bug
- `train_rl_quick.py`: Fast training with RandomPlayer games + outcome labels (rejected — 8% WR)
- `train_rl_enriched.py`: Full training with AlphaBeta games + VF residual labels

## Next Steps

The RL model is now usable for:
1. **Hybrid Agent**: Works with `analyze_position`, `get_best_move`, `check_threats` tools
2. **Curriculum Self-Play (Option C)**: The model can learn and improve through self-play since it can now discriminate between actions

**Why:** The original 30 features aggregated spatial information into counts that were identical for same-type actions. The 72 enriched features encode per-resource production, robber position effects, opponent detail, and build capability — all of which differ between specific actions.

**How to apply:** Use `rl_enriched_model.pt` (72-dim, linear output) for all RL model needs. The `agent_tools.py` functions handle the linear output automatically via sigmoid conversion for win probability display.

**See also:** [[rl-guard-results]] [[hybrid-agent-results]] [[final-results-2026-08-08]]
