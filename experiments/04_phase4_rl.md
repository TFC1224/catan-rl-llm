# Phase 4: GRPO Reinforcement Learning

**Date:** 2026-08-06 | **Status:** Infrastructure ready, pending full training run

## Configuration

### Model
- **Base:** Qwen3-8B (4-bit QLoRA)
- **Starting point:** SFT LoRA checkpoint (`checkpoints/sft/`)
- **LoRA:** r=16, alpha=32, dropout=0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### GRPO Training Config
| Parameter | Value |
|---|---|
| Learning rate | 5e-5 |
| LR scheduler | cosine |
| Warmup ratio | 0.1 |
| Batch size (per device) | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Num generations (K) | 4 |
| Beta (KL penalty) | 0.10 |
| Temperature | 0.9 |
| Max completion length | 128 |
| Max prompt length | 2048 |
| Epochs | 3 |

### Reward Function
| Component | Value |
|---|---|
| Valid action | Simulation-based (game outcome: +1 win, -1 loss, 0 draw) |
| Invalid action | -0.5 penalty |
| Simulation rollouts | 1 per candidate (speed) |
| Simulation bot (agent) | WeightedRandomPlayer |
| Simulation bot (opponent) | game.play_tick() — uses built-in opponent bots |

## Infrastructure: Bug Fixes

### 1. Simulator (`simulator.py`) — Fully Rewritten
- **`Game.execute(action)`** — Confirmed exists and works (not `game.step` or `game.play`)
- **`game.play_tick()`** — Auto-plays one bot turn (used for opponents during simulation)
- **No `Game.players`** — Use `state.current_color()` to determine turn order
- **Simulation loop**: Execute candidate action → loop play_tick() for opponents → fast bot for agent → until game ends
- **Performance**: ~0.025s per rollout (single-threaded), ~10-50ms with simulation

### 2. Rollout (`rollout.py`) — Simplified
- **Removed manual opponent handling** — `CatanatronEnv.step()` auto-advances opponents via `_advance_until_p0_decision()`
- **Fixed game loop**: Just get state → agent acts → map index → env.step() → repeat
- **Removed `env.game.players`** reference (doesn't exist)
- **Added `int_actions`** to record format (for proper index mapping)

### 3. GRPO Reward Function (`train_grpo.py`) — Properly Integrated
- **Simulation-based reward**: Each candidate action is evaluated by simulating the game to completion
- **Flow**: Parse completion → get action_index → map to catan Action → `simulate_from_state()` → reward
- **Fast path**: Uses `from_action_space(action_idx, playable_actions)` to map action space indices
- **Base64 handling**: Dataset stores serialized games as base64, reward function decodes

### 4. Dataset Format (`grpo_dataset.py`)
- **Columns**: prompt, serialized_game (base64), valid_actions (JSON), int_actions (JSON)
- **Dataset size**: 3,917 records from 20 bot games (MINI, 6 VP)

## Rollout Data Generation

**Bot-vs-bot gameplay** (no model inference needed for data collection):

```bash
# Generated via inline script
# 20 games, MINI (6VP), VictoryPointPlayer vs WeightedRandomPlayer
# Result: 3,917 records in ~2 seconds
```

| Metric | Value |
|---|---|
| Games | 20 |
| Total records | 3,917 |
| Avg records/game | ~196 |
| Generation time | ~2 seconds |
| Expert bot | VictoryPointPlayer (BLUE) |
| Opponent | WeightedRandomPlayer (RED) |

## GRPO Pipeline Test Results

**End-to-end test: 1 training step (B=2, K=2)**

| Metric | Value |
|---|---|
| Training loss | 6.0e-05 |
| Step time | ~5.2s |
| Completion mean length | 12.5 tokens |
| Reward mean | 0.0 (simulation neutral at step 1) |
| KL divergence | 0.0006 |
| Entropy | 0.180 |
| VRAM usage | Stable |
| Status | **PASSED** |

### Critical Bug Fixes During GRPO Integration

1. **Reward function signature**: TRL's `GRPOTrainer` passes `completions` as `List[str]` (B flat strings, one per prompt), NOT `List[List[Dict]]` (B groups of K message dicts). The original code iterated over each string, treating characters as individual completions, producing 79+ rewards per prompt instead of 1.

2. **Qwen3 thinking mode**: Qwen3 generates `<think>...</think>` tokens before the action JSON. Fixed by passing `generation_kwargs={"enable_thinking": False}` in `GRPOConfig`.

3. **Prompt format**: GRPO prompts must include the full chat template (system + user + assistant prefix), matching the SFT training format. The rollout data now uses `tokenizer.apply_chat_template()` to build `"<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"`.

4. **Missing `max_prompt_length`**: This GRPOConfig parameter doesn't exist in TRL v1.9.2. Removed from config.

5. **Batch vs num_generations**: `per_device_train_batch_size` must be divisible by `num_generations`. Default values: B=4, K=4.

## Progressive Curriculum (Planned)

| Iter | Map | VPs | Opponents | Games | Status |
|---|---|---|---|---|---|
| 1 | MINI | 6 | WeightedRandom × 2 | 100 | Pending |
| 2 | MINI | 6 | WeightedRandom + AlphaBeta | 150 | Pending |
| 3 | BASE | 10 | WeightedRandom × 2 | 100 | Pending |
| 4 | BASE | 10 | AlphaBeta × 2 | 100 | Pending |

## Known Limitations

1. **Training speed**: 8B model inference + game simulation makes each GRPO step ~60-120s. Full training (3 epochs × ~980 steps) would take ~16-32 hours.
2. **Simulation fidelity**: Using WeightedRandomPlayer as agent simulator during rollouts may not accurately reflect model's actual play style.
3. **Reward sparsity**: Terminal reward (win/loss) is sparse — only 1 rollout per candidate reduces variance but may miss stochastic outcomes.
4. **VRAM**: 4-bit model (~6GB) + GRPO overhead (K=4 generations) may approach 24GB limit. May need to reduce batch size to 2.

## Running Full GRPO Training

```bash
# Step 1: Generate rollout data (fast, uses bots)
python scripts/generate_grpo_data.py --num_games 100 --output data/grpo/iter1/

# Step 2: Run GRPO training
python scripts/train_grpo.py \
    --lora checkpoints/sft/ \
    --data data/grpo/iter1_train/ \
    --output checkpoints/grpo/iter1/ \
    --beta 0.06

# Step 3: Evaluate
python scripts/eval_grpo.py --model checkpoints/grpo/iter1/ --games 10
```

## GRPO Iteration 1: Running Now

**Date:** 2026-08-07 | **Status:** Training in progress

### Data
| Metric | Value |
|---|---|
| Games | 100 (VictoryPointPlayer vs WeightedRandom) |
| Records generated | 24,191 |
| Training subset | 3,000 records (from first ~12 games) |
| Generation time | ~44 seconds |

### Training Config (Final)
| Parameter | Value | Rationale |
|---|---|---|
| Temperature | 0.9 | >1.0 causes Chinese hallucination |
| Beta (KL) | 0.06 | Moderate exploration wiggle room |
| Simulation rollouts | 1 | Speed — single rollout is <25ms |
| Learning rate | 5e-5 | Default |
| Num generations (K) | 4 | Per-prompt candidates |
| Batch size | 4 | Per device |
| Gradient accumulation | 4 | Effective batch = 16 |
| Epochs | 1 | ~750 steps, ~3 hours |
| Max completion length | 128 | Short JSON output |

### Key Issues During GRPO Setup

1. **Qwen3 `<think>` tags**: Qwen3's chat template wraps assistant output in `<think>...</think>`. The SFT model was trained with this format, so generation naturally includes think tags. The action parser handles stripping them. `enable_thinking=False` is not a valid `model.generate()` parameter (ValueError).

2. **Hallucination at high temperature**: At temp=1.2, the model generates long Chinese text instead of JSON (e.g., "嘉年華🎄！在這場火星探險中..."). Fixed by keeping temp=0.9 and adding a -1.0 penalty for outputs >200 chars without JSON.

3. **Stuck/no-learning at default params**: With beta=0.10, temp=0.9, the model's KL divergence was near zero and reward was flat at -0.5 to -0.9. Lowered beta to 0.06 for more policy movement.

4. **Invalid action penalty**: Increased from -0.5 to -1.0 to strongly discourage hallucination/invalid actions vs simulation-based terminal rewards.

### Rewards Observed (test run, 200 records)
| Metric | Range |
|---|---|
| Reward mean | -0.94 to -0.44 |
| KL divergence | 1e-9 to 1e-4 |
| Entropy | 0.03 to 0.07 |
| Loss | -0.003 to 1e-5 |
| Completion length | 12 tokens (consistent) |

## Next Steps

- [x] Fix GRPO infrastructure (simulator, rollout, reward)
- [x] Verify GRPO pipeline with integration test
- [/] Run GRPO Iteration 1 (MINI, WeightedRandom): Training in progress
- [ ] Evaluate GRPO model vs SFT baseline (action validity, win rate)
- [ ] Iterate on reward function if needed (dense rewards, more rollouts)
- [ ] Scale to BASE map and stronger opponents (Iterations 2-4)
