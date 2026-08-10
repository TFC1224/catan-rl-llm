# Phase 2: Agent Implementation

**Date:** 2026-08-06 | **Duration:** ~4 hours | **GPU:** NVIDIA GeForce RTX 4090 D 24GB

## 1. Configuration

### Agent Architecture
- **Pattern:** LlamaGym Agent (get_system_prompt, format_observation, extract_action)
- **Base class:** `CatanAgent` (abstract)
- **Concrete class:** `QwenCatanAgent`
- **Model:** Qwen/Qwen3-8B-Instruct (4-bit QLoRA)
- **Prompt version:** V1 (basic rules + core strategy)

### Files Implemented
| File | Lines | Purpose |
|---|---|---|
| `src/catan_rl/agent/base.py` | ~250 | Abstract CatanAgent with LlamaGym pattern |
| `src/catan_rl/agent/prompts.py` | ~180 | 3 system prompt variants (V1 basic, V2 advanced, concise) |
| `src/catan_rl/agent/observation.py` | ~240 | Structured observation formatter (7 sections) |
| `src/catan_rl/agent/action_parser.py` | ~230 | 5-strategy robust action parser |
| `src/catan_rl/agent/qwen_agent.py` | ~220 | Qwen3-8B concrete implementation |
| `src/catan_rl/env/catan_env.py` | ~115 | Environment factory with opponent registry |
| `src/catan_rl/env/game_state.py` | ~110 | Serialization/deserialization utilities |
| `src/catan_rl/env/reward.py` | ~150 | Terminal, dense, and composite reward functions |

## 2. Design Decisions

### Why decouple agent from trainer?
LlamaGym's Agent pattern provides a clean interface (get_system_prompt, format_observation, extract_action). By decoupling the agent from the training algorithm, we can:
1. Use the same agent for data collection AND evaluation
2. Swap training algorithms (PPO, GRPO, DPO) without changing agent code
3. Test agent behavior without a loaded model (random fallback for debugging)

### Action Parsing Strategy
The parser uses 5 fallback strategies (in order):
1. **JSON number:** `{"action_number": <N>}` — preferred format
2. **JSON type:** `{"action": "<TYPE>", "params": {...}}` — flexible format
3. **Regex match:** Match action type names in free text
4. **Fuzzy match:** String similarity against valid action descriptions
5. **Random fallback:** Last resort (logged as invalid)

### Observation Formatting
The observation formatter produces 7 structured sections:
1. Game Phase — current game state (BUILD_INITIAL_SETTLEMENT, PLAY_TURN, etc.)
2. Player Resources — 5 resource counts
3. Development Cards — hand + played knights
4. Buildings Owned — roads/settlements/cities + longest road/army status
5. Victory Points — all players
6. Board Summary — robber position, key info
7. Available Actions — numbered list with descriptions

### Available Opponent Bots
- WeightedRandomPlayer: Random-but-biased baseline
- VictoryPointPlayer: VP-maximizing strategy (strongest built-in)
- Aliases for compatibility: AlphaBetaPlayer, ValueFunctionPlayer (map to VictoryPointPlayer)

## 3. Results

### Import Verification
All agent modules import successfully:
```
All imports: OK
Available opponents: ['WeightedRandomPlayer', 'VictoryPointPlayer', 'AlphaBetaPlayer', 'ValueFunctionPlayer']
```

### Observation Formatting Test
Sample output from MINI map, initial settlement phase:
```
## Game Phase
Build Initial Settlement

## Your Resources (Player 0)
Wood: 0 | Brick: 0 | Sheep: 0 | Wheat: 0 | Ore: 0

## Available Actions
0: BLUE BUILD_SETTLEMENT (node=0)
1: BLUE BUILD_SETTLEMENT (node=1)
...
20: BLUE BUILD_SETTLEMENT (node=23)

Reply with ONLY: {"action_number": <integer>}
```

Observation length: ~760 chars — fits comfortably within 2048 token prompt budget.

### System Prompt Quality
- V1: 950 chars — includes rules, building costs, VP table, 6 strategy heuristics
- V2: 1700 chars — adds resource valuation by phase, probability math, opponent analysis
- Concise: 380 chars — bare essentials for limited context

## 4. Key Observations

1. **Action representation:** `env.get_valid_actions()` returns integer indices (for env.step()), while `state.playable_actions` returns rich `Action(color, action_type, value)` namedtuples (for display). Both always have the same length and index mapping.

2. **Catanatron version:** v3.2.1 has 2 built-in bots (WeightedRandomPlayer, VictoryPointPlayer). The AlphaBetaPlayer referenced in project docs is from a newer experimental version — we use VictoryPointPlayer as the strongest available opponent.

3. **Step return format:** 5-tuple (obs, reward, terminated, truncated, info) — Gymnasium convention.

4. **Action space:** Discrete(290) — much larger than expected. Many action indices correspond to the same action type with different parameter values (e.g., different node IDs for BUILD_SETTLEMENT).

## 5. Artifacts

- Agent module: `src/catan_rl/agent/` (5 files)
- Environment module: `src/catan_rl/env/` (3 files)
- Configuration: `configs/default.yaml`, `configs/sft_config.yaml`, `configs/grpo_config.yaml`, `configs/eval_config.yaml`

## 6. Next Steps

Proceed to Phase 3: SFT Pretraining
- Generate training data from VictoryPointPlayer gameplay
- Train Qwen3-8B-Instruct with LoRA on formatted observations → actions
- Target: >95% action validity rate, baseline win rate vs WeightedRandom on MINI
