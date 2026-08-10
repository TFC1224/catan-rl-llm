---
name: ab-sft-results
description: AlphaBeta SFT training converges but achieves only random-level play (25% win rate)
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-07T05:51:02.498Z
---

# AlphaBeta SFT Results

**Date:** 2026-08-07

## What we did

1. Generated SFT data from AlphaBetaPlayer (DarekYu's fork) — 300 games, 18,945 strategic decisions, 98% AB win rate
2. Trained Qwen3-8B via QLoRA SFT (r=16, alpha=32) on this data — loss 1.627→0.044, accuracy 68%→98% in 200 steps
3. Evaluated against RandomPlayer (4-player, 10VP, temp=0.1) — 20 games

## Results

| Metric | Value |
|---|---|
| SFT train loss (final) | 0.044 |
| SFT train accuracy | 98.3% |
| Action validity (eval) | 100% |
| Win rate vs Random | 25.0% (5/20) |
| Random baseline (4P) | 25.0% |

## Key finding

SFT on AlphaBeta imitation data teaches the model to output valid actions but does NOT improve strategic play. The model builds settlements, cities, and roads, but at exactly random-level effectiveness.

## Why

Imitation learning (SFT) copies surface patterns from the teacher but cannot generalize strategic reasoning. The model maps observation→action_number patterns but doesn't understand WHY AlphaBeta chose those actions. In new game states (different board layouts, opponents), the mapping fails.

## Implications for next steps

- SFT provides action validity baseline but not skill
- RL (GRPO) with game-outcome rewards is needed for strategic improvement
- Alternatively: incorporate reasoning traces in SFT data, use more diverse training data, or explore reward-model-based approaches

**Artifacts:**
- Checkpoint: `checkpoints/ab_sft/checkpoint-200/`
- Data: `data/ab_sft/main/`
- Eval results: `checkpoints/ab_sft/checkpoint-200/eval_random_2g.json`
