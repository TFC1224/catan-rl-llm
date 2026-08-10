---
name: aesl-experiment-results
description: AESL diversity early-stopping hypothesis REJECTED for Catan SFT — entropy-peak checkpoint (0% WR) worse than best-loss checkpoint (20% WR)
metadata: 
  node_type: memory
  type: project
  originSessionId: ce8089de-6570-46ef-a0a3-754ef13e0fa7
  modified: 2026-08-08T05:26:39.058Z
---

# AESL Diversity Early-Stopping Experiment Results (2026-08-08)

**Conclusion: AESL hypothesis is REJECTED for Catan cold-start SFT. The entropy-peak checkpoint performs WORSE than the best-loss checkpoint.**

## Final Results

| Method | Win Rate | Avg Turns | Games |
|--------|----------|-----------|-------|
| WeightedRandom (baseline) | ~25% | ~200 | — |
| AB-SFT (existing, 17k×3epochs) | 25% | ~200 | 5 |
| VF-Distill v2 | 40% | ~200 | 5 |
| Hybrid Agent (tools+VF) | 100% | ~90 | 6 |
| **AESL Best-Loss (step=500)** | **20.0%** | 259 | 10 |
| **AESL Entropy-Peak (step=150)** | **0.0%** | 304 | 10 |

## AESL Hypothesis Test: REJECTED

Entropy-peak WR (0.0%) ≤ Best-loss WR (20.0%)

## Entropy Trajectory

| Step | Entropy | PPL | Train Loss |
|------|---------|-----|-----------|
| 50   | 0.2013  | 1.37 | 1.7822 |
| 100  | 0.2421  | 1.29 | 0.1747 |
| 150  | **0.2514** | 1.27 | 0.1167 | ← PEAK ENTROPY
| 200  | 0.2427  | 1.27 | 0.1092 |
| 250  | 0.2358  | 1.27 | 0.1112 |
| 300  | 0.2441  | 1.26 | 0.1096 |
| 350  | 0.2298  | 1.26 | 0.1026 |
| 400  | 0.2455  | 1.27 | 0.1068 |
| 450  | 0.2386  | 1.27 | 0.1016 |
| 500  | 0.2367  | **1.26** | 0.0978 | ← BEST LOSS
| 550  | 0.2363  | 1.26 | 0.0976 |
| 600  | 0.2360  | 1.26 | 0.0864 |

## Why AESL Didn't Work for Catan

1. **Domain mismatch**: AESL targets math reasoning with long-CoT outputs (thousands of tokens of reasoning). Catan outputs are short JSON (`{"action_number": 5}`) with ~6 tokens per response. Entropy dynamics are fundamentally different.

2. **Insufficient training at peak**: At step 150, the model had only seen ~1,200 examples. The model was still confused, not "diversely capable" — the high entropy reflects uncertainty from lack of learning, not useful strategic diversity.

3. **No RL phase**: The AESL paper's diversity peak predicts post-RL performance. Without RL (GRPO doesn't work for Catan), the diversity metric loses its predictive power.

4. **Different data regime**: 5k examples × 1 epoch vs the original AB-SFT (17k × 3 epochs). The best-loss checkpoint at 20% is close to the 25% baseline but still slightly below, suggesting more data/epochs are needed.

## Training Config
- Base: Qwen3-8B + fresh LoRA (r=16, alpha=32)
- Data: 5,000 AB-SFT examples (sampled from 17,050)
- Epochs: 1 (625 steps at batch_size=2, grad_accum=4)
- LR: 1e-4 (cosine schedule, 10% warmup)
- Max length: 2048 tokens
- Training time: 79.8 min
- Evaluation: 10 games per checkpoint vs 3×WeightedRandom opponents

## Key Insight

For Catan SFT, **more training is better** — the model needs sufficient exposure to game states before it can make competent decisions. The entropy peak occurs too early (before competence is achieved) to be useful as an early-stopping criterion.

**See also:** [[ab-sft-results]] [[final-results-2026-08-08]] [[hybrid-agent-results]]

**Why:** The AESL experiment tested whether diversity-based early stopping could improve Catan gameplay, but the hypothesis was rejected.
**How to apply:** Do NOT use entropy-based early stopping for Catan SFT. Use best validation loss or fixed number of epochs. Focus on the Hybrid Agent architecture (tools+VF) which remains the winning approach at 100% WR.
