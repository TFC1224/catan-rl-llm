#!/usr/bin/env python3
"""
QwenAgentPlayer: LLM orchestrates tool calls each turn (Catanatron AgentPlayer adapted for Qwen3-8B).

Key differences from original AgentPlayer (which uses Ollama):
1. Uses QwenCatanAgent instead of ollama.Client
2. Uses base Qwen3-8B (no LoRA) for flexible JSON tool calling
3. Same tool set: analyze_position, check_threats, get_best_move, simulate_outcome, execute_action
4. Max 6 tool calls per turn

Architecture:
    1. LLM receives: available actions + tool definitions + turn context
    2. LLM calls analyze_position → gets win probability
    3. LLM calls check_threats → identifies dangerous opponents
    4. LLM calls get_best_move → RL model finds best action for goal
    5. LLM calls execute_action → commits to action index

Usage:
    python scripts/eval_qwen_agent.py --games 10 --opponent weighted_random
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from typing import List

import numpy as np

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CATANATRON_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CATANATRON_ROOT, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import get_value_fn
from catan_rl.rl.value import CONTENDER_WEIGHTS
from catan_rl.agent.qwen_agent import QwenCatanAgent
from catanatron_experimental.agent_tools import (
    analyze_position, check_threats, get_best_move, simulate_outcome,
)
from catanatron_experimental.rl_value_network import CatanValueNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RL_MODEL_PATH = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/rl_selfplay_model2.pt'

TOOL_SYSTEM_PROMPT = """You are an expert Settlers of Catan player with access to analysis tools.
Each response must be a single JSON object calling ONE tool at a time.

Available tools:

  analyze_position — assess your board position using the RL value network
  check_threats — identify dangerous opponents and their VP counts
  get_best_move — find best action for a strategic goal. Args: {"goal": "<goal>"}
                  Goals: build_city, build_settlement, expand_roads,
                         buy_dev_card, maximize_production, block_opponent, trade, any
                  Returns recommended_index — the EXACT action index to use.
  simulate_outcome — verify an action with AlphaBeta lookahead. Args: {"action_index": N}
  execute_action — commit to action and end turn. REQUIRED as final call.
                   Args: {"action_index": N, "reasoning": "why"}

Response format (ONE tool per response, NO other text):
{"tool": "tool_name", "args": {}, "reasoning": "why you are calling this tool"}

CRITICAL RULES:
- Always start with analyze_position
- Always end with execute_action
- Maximum 6 tool calls total per turn
- When get_best_move returns recommended_index, use that EXACT index in execute_action
- Do NOT choose END_TURN unless every other option scores worse
- Respond with ONLY the JSON object, no markdown code blocks, no other text"""


class QwenAgentPlayer(Player):
    """LLM orchestrates Catanatron tools through JSON tool calling."""

    def __init__(self, color, agent, rl_model, vf=None, use_vf_guard=True,
                 max_tool_calls=6):
        super().__init__(color)
        self.agent = agent
        self.rl_model = rl_model
        self.vf = vf
        self.use_vf_guard = use_vf_guard
        self.max_tool_calls = max_tool_calls
        self.total_decisions = 0
        self.overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            self.total_decisions += 1
            return actions[0]

        action_names = {a.action_type.name for a in actions}
        if action_names == {"ROLL"}:
            return actions[0]

        self.total_decisions += 1

        if all(n == "DISCARD_RESOURCE" for n in action_names):
            return self._heuristic_discard(game, actions)
        if all(n == "MOVE_ROBBER" for n in action_names):
            return self._heuristic_robber(game, actions)

        # Main tool-calling loop
        state = game.state
        idx = state.colors.index(self.color)
        my_vp = state.player_state.get(f"P{idx}_VICTORY_POINTS", 0)

        # Build action index for tool reference
        action_index = [
            {"index": i, "action": a.action_type.name, "value": str(a.value)}
            for i, a in enumerate(actions[:30])
        ]
        actions_text = "\n".join(
            f"[{e['index']}] {e['action']}: {e['value']}"
            for e in action_index
        )

        # Build user message
        user_msg = (
            f"Turn {state.num_turns} | You have {my_vp}/10 VP\n\n"
            f"Available actions ({len(actions)} total):\n{actions_text}\n\n"
            f"Start by calling analyze_position to assess your situation."
        )

        messages = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        chosen_action = actions[0]

        for _ in range(self.max_tool_calls):
            # Call LLM
            try:
                response = self._call_llm(messages)
            except Exception as e:
                logger.warning(f"LLM error: {e}")
                break

            if response is None:
                break

            # Parse JSON tool call
            tool_req = self._parse_tool_call(response)
            if tool_req is None:
                # Try to extract action index as fallback
                idx_match = re.search(r'\b(\d+)\b', response)
                if idx_match:
                    ai = max(0, min(int(idx_match.group()), len(actions) - 1))
                    chosen_action = actions[ai]
                break

            tool_name = tool_req.get("tool", "")
            args = tool_req.get("args", {})
            reasoning = tool_req.get("reasoning", "")

            # Execute tool
            result = self._execute_tool(tool_name, args, game, actions)

            if tool_name == "execute_action":
                ai = max(0, min(args.get("action_index", 0), len(actions) - 1))
                chosen_action = actions[ai]

                # VF-Guard: double-check with VF scoring
                if self.use_vf_guard and self.vf:
                    best_idx = self._vf_score_all(game, actions)
                    if best_idx != ai:
                        self.overrides += 1
                        chosen_action = actions[best_idx]

                return chosen_action

            # Append exchange to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for {tool_name}:\n"
                    f"{json.dumps(result, default=str)}\n\n"
                    f"Now call your next tool. "
                    f"If get_best_move gave you a recommended_index, "
                    f"use that exact index in execute_action."
                ),
            })

        # Fallback: VF-Guard scoring
        if self.use_vf_guard and self.vf:
            best_idx = self._vf_score_all(game, actions)
            if best_idx != 0:
                self.overrides += 1
            return actions[best_idx]
        return chosen_action

    def _call_llm(self, messages):
        """Call Qwen model with conversation messages."""
        # Build a single prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

        # Use agent's generate_response
        raw = self.agent.generate_response(prompt)

        # Clean response
        raw = raw.strip()
        # Remove any trailing assistant markers
        for marker in ["<|im_end|>", "<|im_start|>"]:
            if marker in raw:
                raw = raw.split(marker)[0].strip()
        return raw

    def _parse_tool_call(self, response):
        """Parse JSON tool call from LLM response."""
        # Clean markdown code blocks
        clean = re.sub(r'```(?:json)?\s*|\s*```', '', response).strip()

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', clean)
        if json_match:
            clean = json_match.group()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.debug(f"Could not parse: {clean[:100]}")
            return None

    def _execute_tool(self, tool_name, args, game, actions):
        """Execute a tool and return result."""
        if tool_name == "analyze_position":
            return analyze_position(game, self.color, self.rl_model)
        elif tool_name == "check_threats":
            return check_threats(game, self.color)
        elif tool_name == "get_best_move":
            goal = args.get("goal", "any")
            return get_best_move(game, self.color, goal, self.rl_model, actions)
        elif tool_name == "simulate_outcome":
            ai = max(0, min(args.get("action_index", 0), len(actions) - 1))
            return simulate_outcome(game, self.color, actions[ai])
        elif tool_name == "execute_action":
            return {"status": "executing"}
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _vf_score_all(self, game, actions):
        """VF-Guard: score all actions, return best index."""
        best_idx, best_score = 0, float('-inf')
        for i, action in enumerate(actions):
            try:
                gc = game.copy()
                gc.execute(action)
                score = self.vf(gc, self.color)
                if score > best_score:
                    best_score = score
                    best_idx = i
            except Exception:
                pass
        return best_idx

    def _heuristic_discard(self, game, actions):
        state = game.state
        idx = state.colors.index(self.color)
        resources = {
            r: state.player_state.get(f"P{idx}_{r}_IN_HAND", 0)
            for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        }
        return max(actions, key=lambda a: resources.get(str(a.value), 0))

    def _heuristic_robber(self, game, actions):
        state = game.state
        best_vp, best_color = -1, None
        for i, c in enumerate(state.colors):
            if c == self.color:
                continue
            vp = state.player_state.get(f"P{i}_VICTORY_POINTS", 0)
            if vp > best_vp:
                best_vp, best_color = vp, c
        for a in actions:
            if a.value and len(a.value) > 1 and a.value[1] == best_color:
                return a
        return actions[0]

    @property
    def override_rate(self):
        return self.overrides / max(self.total_decisions, 1)


def load_agent(device="cuda"):
    """Load base Qwen3-8B (no LoRA) for flexible tool calling."""
    logger.info("Loading base Qwen3-8B (no LoRA) for tool calling...")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        device=device,
        load_in_4bit=True,
        lora_path=None,  # No LoRA — base model for flexible JSON
        prompt_version="v1",
    )
    agent.temperature = 0.1
    return agent


def load_rl_model():
    if not os.path.exists(RL_MODEL_PATH):
        logger.error(f"RL model not found: {RL_MODEL_PATH}")
        return None
    model = CatanValueNetwork.load(RL_MODEL_PATH)
    model.eval()
    logger.info(f"RL model loaded from {RL_MODEL_PATH}")
    return model


def run_evaluation(num_games=10, opponent_type="weighted_random",
                   use_vf_guard=True, device="cuda", seed=42):
    random.seed(seed)
    np.random.seed(seed)

    agent = load_agent(device)
    rl_model = load_rl_model()
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS) if use_vf_guard else None

    opponent_class = WeightedRandomPlayer if opponent_type == "weighted_random" else RandomPlayer
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    results = []
    t_start = time.time()

    for i in range(num_games):
        game_seed = seed + i * 100
        shuffled = list(colors)
        random.seed(game_seed)
        random.shuffle(shuffled)

        agent_color = shuffled[0]
        player = QwenAgentPlayer(agent_color, agent, rl_model, vf, use_vf_guard)
        opponents = [opponent_class(c) for c in shuffled[1:]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        try:
            game_obj = Game(all_players, vps_to_win=10)
            winner = game_obj.play()
        except Exception as e:
            logger.warning(f"Game error (seed={game_seed}): {e}")
            results.append({"outcome": "ERROR"})
            continue

        outcome = "WIN" if winner == agent_color else "LOSS"
        results.append({
            "outcome": outcome,
            "turns": game_obj.state.num_turns,
            "overrides": player.overrides,
            "total_decisions": player.total_decisions,
        })

        if (i + 1) % 3 == 0:
            elapsed = time.time() - t_start
            wins = sum(1 for r in results if r["outcome"] == "WIN")
            logger.info(
                f"Game {i+1}/{num_games} | Wins: {wins}/{i+1} "
                f"({wins/(i+1)*100:.0f}%) | Elapsed: {elapsed:.0f}s"
            )

    total_time = time.time() - t_start
    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")

    logger.info("=" * 60)
    logger.info("  QwenAgentPlayer Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"  VF-Guard: {use_vf_guard}")
    logger.info(f"  Games: {num_games} | Win rate: {wins}/{completed} ({wins/max(completed,1)*100:.1f}%)")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--opponent", type=str, default="weighted_random")
    parser.add_argument("--no_vf_guard", action="store_true",
                        help="Disable VF guardrail")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        num_games=args.games,
        opponent_type=args.opponent,
        use_vf_guard=not args.no_vf_guard,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
