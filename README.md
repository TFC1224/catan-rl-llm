# Catan RL + LLM: Training Qwen3-8B to Play Settlers of Catan

This project uses **LlamaGym** agent patterns and **GRPO** (Group Relative Policy Optimization) from **TRL** to fine-tune **Qwen3-8B-Instruct** as a competitive Settlers of Catan AI player.

> **Current status (2026-08):** SFT cold-start and SimSFT have been validated. Full GRPO reinforcement learning hit a structural obstacle (entropy collapse after SFT) and is being reattempted via AESL-style lightweight cold-start.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Current Status & Results](#current-status--results)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [GPU Requirements](#gpu-requirements)
- [Key Findings](#key-findings)
- [Roadmap](#roadmap)
- [Experiment Documentation](#experiment-documentation)
- [References](#references)

---

## Architecture

```
LLM Agent (Qwen3-8B)  ←→  Catanatron Gym Environment
        │                        │
        │  format_observation()   │  env.get_valid_actions()
        │  extract_action()       │  env.step(action_index)
        │                        │
        v                        v
   GRPOTrainer (TRL)    ←  Reward via game simulation
```

- **Agent Pattern**: LlamaGym-style (get_system_prompt, format_observation, extract_action)
- **Training**: SFT cold-start → GRPO (TRL) with group-relative advantage (no value head needed)
- **Efficiency**: QLoRA (4-bit) — fits in 24 GB VRAM
- **Model**: Qwen3-8B-Instruct
- **Engine**: Catanatron v3.2.1 + catanatron-gym v4.0.0
- **TRL version**: 1.9.2

---

## Project Structure

```
├── configs/              # YAML configuration files
├── src/catan_rl/
│   ├── agent/            # Agent implementation (LlamaGym pattern)
│   │   ├── base.py       # Abstract CatanAgent
│   │   ├── qwen_agent.py # Qwen3-8B concrete agent
│   │   ├── observation.py# Game state → text formatter
│   │   ├── action_parser.py # Robust action parsing (5 strategies)
│   │   └── prompts.py    # System prompts (3 variants)
│   ├── env/              # Catanatron wrappers
│   │   ├── catan_env.py  # Environment factory
│   │   ├── game_state.py # Serialization
│   │   ├── simulator.py  # Parallel game simulation
│   │   └── reward.py     # Reward functions
│   ├── data/             # Data pipeline
│   │   ├── rollout.py    # Game data collection
│   │   ├── sft_dataset.py# SFT data generation
│   │   ├── grpo_dataset.py # GRPO dataset construction
│   │   └── preprocessing.py # Chat templating
│   ├── training/         # Training orchestration
│   │   ├── train_sft.py  # SFT training
│   │   ├── train_grpo.py # GRPO training
│   │   └── utils.py      # Model loading, LoRA setup
│   └── eval/             # Evaluation
│       ├── arena.py      # Tournament runner
│       ├── metrics.py    # Win rate, ELO, stats
│       └── visualize.py  # Matplotlib plots
├── scripts/              # CLI entry points
├── experiments/          # Experiment documentation
├── notebooks/            # Jupyter notebooks
└── data/                 # Generated datasets (gitignored)
```

---

## Current Status & Results

### Phase 1 — Environment Setup ✅
All 18/18 environment checks passed (torch 2.1.2, transformers 5.14.1, trl 1.9.2, catanatron 3.2.1). `flash_attention_2` is unavailable on the target hardware; we use `sdpa` instead. `wandb` is not configured (`report_to=none`).

### Phase 2 — Agent Implementation ✅
Five core modules: `base.py`, `prompts.py` (3 prompt variants), `observation.py` (7-section structured observation), `action_parser.py` (5-stage fallback parsing), `qwen_agent.py`.

Key API adaptations discovered:
- `state.current_color` is a **method**, not a property → call `state.current_color()`
- Catanatron's `Game` has **no `.players` attribute**
- P0 must be passed as `Color.BLUE` enum, not a string
- `env.get_valid_actions()` returns **action-space indices** (0–289), but the model emits **sequential indices** — a mapping is required

### Phase 3 — SFT Cold-Start ✅

| Metric | Value |
|---|---|
| Expert data | 100 games VictoryPointPlayer vs WeightedRandom, MINI 6VP |
| Records | 20,558 (train 18,502 / val 2,056) |
| Training time | ~1h 45m (3 epochs, 564 steps) |
| Final train loss | 0.0887 |
| Final eval loss | 0.02186 |
| Token accuracy | 99.08% |
| **Action legality** | **100% (1018/1018)** |
| Win rate vs WeightedRandom | 2/6 = 33.3% (2-player) |

**Verdict:** SFT teaches format perfectly, but the resulting model is too deterministic for downstream RL.

### Phase 4 — GRPO Reinforcement Learning ❌

Full GRPO training revealed a structural obstacle: post-SFT entropy collapses to **0.01–0.06**, so within each group of K=4 generations the responses are nearly identical → advantage ≈ 0 → no learning signal. This is not a hyperparameter issue; it is a cold-start design problem.

| Metric | Observed | Problem |
|---|---|---|
| Entropy | 0.01–0.06 | Model near-deterministic |
| Advantage | ≈ 0 | No group-relative signal |
| Reward | -0.94 ~ -0.44 | Dominated by invalid actions |
| KL | 1e-9 ~ 1e-4 | Policy barely moves |

### Phase 3.5 — SimSFT (Simulation-Guided Refinement) ✅

As a workaround, we developed **SimSFT**: enumerate all valid actions for each state, simulate each with VictoryPointPlayer (5 rollouts), and use the simulated-best action as the new SFT target.

| Metric | Value |
|---|---|
| Input records | 1,000 (from GRPO iter1 `rollout.jsonl`) |
| Refinable states | 212 (states with >1 valid action) |
| States with better action found | **44.3%** |
| Average reward improvement | **+0.146** |
| Training time | ~4 minutes |
| Final loss | 0.036 / eval 0.034 |

SimSFT validates the pipeline and the data-quality assumption, but 212 records are too few to materially change model behavior.

---

## Quick Start

### 1. Environment Setup

```bash
bash scripts/setup_env.sh            # Install dependencies
python scripts/download_model.py      # Download Qwen3-8B-Instruct
python scripts/test_imports.py        # Verify environment (18/18 checks)
```

### 2. SFT Pretraining

```bash
python scripts/generate_sft_data.py --num_games 500 --output data/sft/
python scripts/train_sft.py --data data/sft/ --output checkpoints/sft/
```

### 3. GRPO Reinforcement Learning

```bash
# Iteration 1: MINI map, WeightedRandom opponents
python scripts/rollout.py --model checkpoints/sft/ --output data/grpo/iter1/ --num_games 100
python scripts/train_grpo.py --lora checkpoints/sft/ --data data/grpo/iter1/ --output checkpoints/grpo/iter1/

# Iteration 2: MINI map, mixed opponents
python scripts/rollout.py --model checkpoints/grpo/iter1/ --opponents WeightedRandomPlayer VictoryPointPlayer --output data/grpo/iter2/
python scripts/train_grpo.py --lora checkpoints/grpo/iter1/ --data data/grpo/iter2/ --output checkpoints/grpo/iter2/

# Iteration 3: BASE map, WeightedRandom opponents
python scripts/rollout.py --model checkpoints/grpo/iter2/ --map BASE --vps 10 --output data/grpo/iter3/
python scripts/train_grpo.py --lora checkpoints/grpo/iter2/ --data data/grpo/iter3/ --output checkpoints/grpo/iter3/

# Iteration 4: BASE map, VictoryPointPlayer opponents
python scripts/rollout.py --model checkpoints/grpo/iter3/ --map BASE --vps 10 --opponents VictoryPointPlayer VictoryPointPlayer --output data/grpo/iter4/
python scripts/train_grpo.py --lora checkpoints/grpo/iter3/ --data data/grpo/iter4/ --output checkpoints/grpo/iter4/
```

### 4. Evaluation

```bash
python scripts/evaluate.py --model checkpoints/grpo/iter4/ --games 100 --output results/
```

---

## Configuration

All hyperparameters are in `configs/`:
- `default.yaml` — Model, LoRA, environment, and generation settings
- `sft_config.yaml` — SFT data generation and training parameters
- `grpo_config.yaml` — GRPO training parameters and curriculum
- `eval_config.yaml` — Evaluation matchups and settings

---

## GPU Requirements

| Component | VRAM |
|---|---|
| Qwen3-8B-Instruct (4-bit) | ~6 GB |
| LoRA adapters (r=16) | ~0.1 GB |
| KV cache (generation) | ~2 GB |
| **Recommended** | **12+ GB** |

Tested on: NVIDIA RTX 4090 D (24 GB)

---

## Key Findings

1. **Action legality is solvable.** SFT alone is enough to push JSON action validity to 100%, but this is "format learning," not "strategy learning."

2. **GRPO requires a diverse cold-start.** A cold-start that over-fits to evaluation accuracy collapses the policy distribution. This matches the *distribution forgetting / diversity forgetting* phenomenon documented in the AESL paper (ICLR 2026).

3. **Catan's reward signal is sparse and noisy.** Under MINI 6VP, even the expert bot struggles to win — Dice variance dominates action choice. Standard 10VP on the BASE map is a more reliable evaluation setting.

4. **SimSFT is a useful alternative target.** It avoids GRPO's diversity requirement by curating better actions via simulation, but it needs significantly more data than 212 records to influence behavior.

5. **Engineering pitfalls** (recorded for future iterations):
   - TRL v1.9.2 reward-fn signature is `List[str]`, not `List[List[Dict]]`.
   - Qwen3 thinking mode must be disabled (`enable_thinking=False`) for stable generation.
   - TRL v1.9.2 has no `max_prompt_length` argument.
   - `per_device_train_batch_size` must be divisible by `num_generations`.
   - `state.current_color` is a method; `Color.BLUE` must be an enum, not a string.

---

## Roadmap

### Phase 5 — AESL-Style Lightweight Cold-Start (next)
The current SFT trains 18.5k records to convergence, which over-collapses the policy.
- Implement AESL adaptive-weighted CE loss:
  ```
  L = -Σ_t  w_t · log πθ(s*_t | q, s_<t)
  w_t = 1 - sigmoid( logit_t / t_scaling · prefix_avg_log_prob_t )
  ```
  with `t_scaling ∈ [3, 5]`.
- Use 3k lightweight samples instead of the full 18.5k.
- Monitor entropy / self-BLEU and stop at the **diversity peak**, not the eval-loss minimum.
- Target post-SFT entropy > 0.3.

### Phase 6 — GRPO Re-Attempt
- Replace the current SFT checkpoint with the AESL checkpoint.
- Use the **BASE map + 10VP** for evaluation to avoid the MINI-board reward sparsity issue.
- Flush stdout in evaluation scripts to fix the output-buffering bug that wasted runs.

### Phase 7 — SimSFT at Scale
- Generate a larger SimSFT dataset (212 → thousands) using the AESL model as the rollout policy.
- Compare against the Phase-6 GRPO checkpoint.

### Optional — Tool-Use Hybrid Agent
- Layer the trained model behind a tool-calling scaffold (analysis + threat detection + best-move + AlphaBeta verification), the same architecture used in the sibling `Catanatron-main` Hybrid Agent that reached 100% win rate in preliminary trials.

---

## Experiment Documentation

Detailed per-phase reports live in `experiments/`:
- `01_phase1_setup.md` — Environment setup log
- `02_phase2_agent.md` — Agent implementation details
- `03_phase3_sft.md` — SFT training results
- `04_phase4_rl.md` — GRPO training iterations
- `05_phase5_eval.md` — Final evaluation report (in progress)

---

## References

- [Catanatron](https://github.com/bcollazo/catanatron) — Catan game engine
- [LlamaGym](https://github.com/KhoomeiK/LlamaGym) — LLM + Gym RL framework
- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning
- *Agents of Change: Self-Evolving LLM Agents for Strategic Planning* (Belle et al., 2025)
- *Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT* (AESL, ICLR 2026)
