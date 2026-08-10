---
name: option-a-v2-results
description: "Option A v2 achieved 40% win rate with three key fixes over v1's 20%"
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-07T15:34:52.833Z
---

# Option A v2 Results (2026-08-07)

**Win rate: 40% (8/20)** vs WeightedRandom — significant improvement over:
- v1: 20% (2/10) — override-only filtering not applied, started from base model
- AB-SFT baseline: 25% — pure imitation learning

## Three fixes applied (Catanatron-inspired)

1. **Override-only filtering**: Only train on 439 decisions where VF corrected LLM (vs 1022 total) — eliminates dilution from LLM's already-correct choices
2. **AB-SFT checkpoint init**: Continue from AB-SFT LoRA adapter instead of base model — preserves basic game knowledge
3. **Lower learning rate**: 1e-4 (Catanatron-style fine-tuning) instead of 2e-4

## Training details

- 439 override examples from 100 VF-Guard games (v1 data)
- 2 epochs, batch size 16 (2×8), max_length 1024
- Loss: 0.073, Accuracy: 97.6%
- Training time: 9.5 min

## Remaining gap

- VF-Guard (LLM + VF at inference): 90% WR
- Best standalone model: 40% WR
- Gap: Text observations cannot encode spatial details for VF's within-type refinements
- Path forward: Agent Tools (Option B) — give LLM access to RL/AlphaBeta for tactical decisions

**See also:** [[vf-guard-discovery]] [[multi-option-strategy-plan]]
