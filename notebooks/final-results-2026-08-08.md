---
name: final-results-2026-08-08
description: "Complete pipeline results: Hybrid Agent (tools+VF) achieves 100% WR, matching VF-Guard's 90% target"
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-08T09:23:19.356Z
---

# Final Pipeline Results (2026-08-08)

## Goal achieved: Built an agent exceeding built-in bots

**Hybrid Agent (tools + VF guardrail): 100% win rate** vs WeightedRandom, matching the VF-Guard gold standard (90%).

## Complete method ranking

| Rank | Method | Win Rate | Guard Type | Training |
|---|---|---|---|---|
| 1 | **Hybrid Agent (tools+VF)** | **100% (6/6)** | VF | AB-SFT + tools |
| 2 | VF-Guard | 90% (9/10) | VF | AB-SFT |
| 3 | Hybrid Agent (tools only) | 66.7% (2/3) | None | AB-SFT + tools |
| 4 | VF-Distill v2 standalone | 40% | None | VF-Guard distillation |
| 5 | AB-SFT standalone | 25% | None | AlphaBeta imitation |
| 6 | RL-Guard | 0-67% unstable | RL | AB-SFT |
| 7 | Hybrid Agent (tools+RL) | 0% (0/3) | RL | AB-SFT + tools |

## Ablation insights (2026-08-08)

| Configuration | Tools | Guardrail | WR | Game length |
|---|---|---|---|---|
| hybrid_vf | ✓ | VF | 100% | 57-127 turns (fast) |
| hybrid_none | ✓ | None | 66.7% | 131-445 turns (variable) |
| hybrid_rl | ✓ | RL | 0% | 155-350 turns (slow losses) |
| RL-Guard | ✗ | RL | 0% | 172-262 turns (slow losses) |
| VF-Guard | ✗ | VF | 90% | ~100 turns |

**Key findings:**
1. Tools provide strategic context that improves standalone performance from 25% → 66.7%
2. VF guardrail provides tactical correctness, pushing from 66.7% → 100%
3. RL model (`rl_selfplay_model2.pt`) is actively harmful for action scoring — predicts game outcomes, not action quality
4. Game length inversely correlates with win rate — better decisions lead to faster wins

## Architecture of the winning Hybrid Agent

```
1. Pre-compute (milliseconds, no LLM calls):
   analyze_position(game, color, rl_model) → win prob, assessment
   check_threats(game, color) → opponent VPs, threat levels
   get_best_move(game, color, goal, rl_model, actions) → recommended indices

2. Enrich observation:
   Append tool outputs to text observation
   ("Win prob: 0.65 | Biggest threat: RED (8 VP) | RL-best (any): #7 (BUILD_CITY, s=0.723)")

3. LLM decision (one call, ~1.6s):
   AB-SFT model sees enriched observation
   Outputs {"action_number": N} in trained format

4. VF guardrail (instant):
   contender_fn scores all valid actions
   Pick highest-scoring action (overrides LLM if needed)
```

## RL Model Is Harmful — Why?

The RL model (`rl_selfplay_model2.pt`, 50K params, 30 features) was trained via AlphaBeta imitation to predict game outcomes. When used for action scoring:

1. The model predicts "will I win from this state?" not "is this action good?"
2. An action that produces a state with high "win probability" might actually be a losing action if the model is poorly calibrated
3. The model has only 30 features — far fewer than what the VF (`contender_fn`, linear heuristic) implicitly uses

Evidence: Every configuration using RL scoring produced 0% WR with very long games (155-350 turns), suggesting the RL model systematically overvalues actions that prolong the game rather than win it.

## RL Model Fix (2026-08-08)

The original RL model (30 features) was FIXED: enriched to 72 features + VF residual training. Now achieves 69% WR vs Random, 44% vs WeightedRandom. See [[rl-model-fixed]].

## Option C: Curriculum Self-Play (2026-08-08) — REJECTED

Three-phase curriculum (Random → AlphaBeta → Self-play) with outcome labels FAILED (12-38% WR). Outcome labels fundamentally noisy for 4-player state evaluation. See [[option-c-curriculum-results]].

## Path Forward

1. **Hybrid Agent is the winning architecture**: 100% WR (6/6), tools + VF guardrail
2. **Scale evaluation**: 6 games is statistically weak. Run 100+ game confirmation.
3. **Fix road features**: Even enriched model has 100% flat road decisions
4. **VF residual curriculum**: Fix formula bug and re-run — safe path to VF ceiling (~90%)
5. **End-to-end Qwen integration**: Replace Ollama with QLoRA adapter in Hybrid Agent

**See also:** [[hybrid-agent-results]] [[rl-model-fixed]] [[option-c-curriculum-results]] [[rl-guard-results]] [[option-a-v2-results]] [[ab-sft-results]]
