---
name: grpo-results
description: "GRPO experiment complete: VF-scored rollout data is harmful for SFT training (0-20% WR vs 25% baseline)"
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-08T02:19:34.447Z
---

# GRPO Comparison Results (2026-08-08)

**Conclusion: GRPO and VF-scored rollout data are NOT effective for training Catan-playing LLMs.**

## Final Results

| Method | Win Rate | Data Source | Examples |
|---|---|---|---|
| **AB-SFT (baseline)** | **25%** | AlphaBeta games | ~2,000 |
| **VF-Distill v2** | **40%** | VF-Guard overrides only | 439 |
| GRPO-SFT-All | 20% (1/5) | VF-best from all rollouts | 1,821 |
| GRPO-SFT-Filtered | 0% (0/5) | VF-best, high-discrimination only | 925 |
| GRPO-SFT-Balanced | 0% (0/5) | VF-best, phase-balanced | 725 |

## Why GRPO Failed

1. **VF-guard rollout data is noisy**: VF scores from arbitrary game states don't form a coherent policy. The "best" action in one context contradicts the "best" action in a similar context.

2. **More data ≠ better**: 1,821 examples (4x VF-Distill v2) produced worse results (20% vs 40%). Data quality matters more than quantity.

3. **Filtering made it worse**: High-discrimination groups (925) scored 0% — these are harder examples where VF scores differ meaningfully, but the model can't learn a consistent pattern from them.

4. **VF-Distill v2 worked because it used VF-Guard override patterns**: LLM proposes → VF corrects → train on correction. This consistent correction pattern is learnable. Raw VF-best actions from arbitrary states are not.

5. **Full GRPO (weighted SFT on all actions) failed** — loss plateau at ~1.0, unable to converge.

## Key Insight

The VF (`contender_fn`) is good for **correcting at inference time** (VF-Guard, Hybrid Agent) but its action scores are NOT suitable for **training data generation**. The VF is a linear heuristic that:
- Works as a guardrail (90-100% WR with VF guard)
- Fails as a teacher (0-20% WR when model imitates VF selections)

The winning architecture remains: **Hybrid Agent = LLM with tool-enriched observations + VF guardrail at inference**.

**See also:** [[final-results-2026-08-08]] [[hybrid-agent-results]] [[vf-guard-discovery]]
