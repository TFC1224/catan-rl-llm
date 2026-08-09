#!/usr/bin/env python3
"""
RL-Guard: LLM proposes actions, RL Value Network scores all candidates, picks best.

Same architecture as VF-Guard but using the trained RL Value Network instead of
the hand-crafted Value Function. The RL network was trained via AlphaBeta imitation
learning and may capture patterns the linear VF misses.

Usage:
    python scripts/eval_rl_guard.py --games 20 --opponent weighted_random
"""

import argparse
import json
import logging
import os
import random
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
from catanatron.state_functions import player_key
from catan_rl.agent.qwen_agent import QwenCatanAgent
from catanatron_experimental.rl_value_network import CatanValueNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RL_MODEL_PATH = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/rl_selfplay_model2.pt'


class RLGuardPlayer(Player):
    """LLM proposes → RL Value Network scores ALL → picks best."""

    def __init__(self, color, agent, rl_model):
        super().__init__(color)
        self.agent = agent
        self.rl_model = rl_model
        self.total_decisions = 0
        self.rl_overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            self.total_decisions += 1
            return actions[0]

        self.total_decisions += 1

        # 1. LLM proposes action
        try:
            agent_action = self.agent.act(
                observation=game.state,
                valid_actions=actions,
                player_index=0,
            )
            llm_idx = agent_action.action_index
            if not (0 <= llm_idx < len(actions)):
                llm_idx = 0
        except Exception:
            llm_idx = 0

        # 2. RL model scores ALL actions
        best_idx, best_score = 0, float('-inf')
        llm_score = None
        for i, action in enumerate(actions):
            try:
                gc = game.copy()
                gc.execute(action)
                score = self.rl_model.predict(gc, self.color)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_idx = i
            if i == llm_idx:
                llm_score = score

        # 3. Use RL best if it beats LLM choice
        if best_idx != llm_idx and llm_score is not None and best_score > llm_score:
            self.rl_overrides += 1
            return actions[best_idx]
        return actions[llm_idx]

    @property
    def override_rate(self):
        return self.rl_overrides / max(self.total_decisions, 1)


def load_agent(checkpoint_path: str, temperature: float = 0.1, device: str = "cuda"):
    logger.info(f"Loading agent from: {checkpoint_path}")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        device=device,
        load_in_4bit=True,
        lora_path=checkpoint_path,
        prompt_version="v1",
    )
    agent.temperature = temperature
    return agent


def load_rl_model():
    if not os.path.exists(RL_MODEL_PATH):
        logger.error(f"RL model not found: {RL_MODEL_PATH}")
        sys.exit(1)
    model = CatanValueNetwork.load(RL_MODEL_PATH)
    model.eval()
    logger.info(f"RL model loaded from {RL_MODEL_PATH}")
    return model


def run_evaluation(checkpoint_path, num_games=20, opponent_type="weighted_random",
                   num_players=4, vps_to_win=10, temperature=0.1, device="cuda", seed=42):
    random.seed(seed)
    np.random.seed(seed)

    agent = load_agent(checkpoint_path, temperature, device)
    rl_model = load_rl_model()

    opponent_class = WeightedRandomPlayer if opponent_type == "weighted_random" else RandomPlayer
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    results = []
    overrides_list = []
    t_start = time.time()

    for i in range(num_games):
        game_seed = seed + i * 100
        shuffled = list(colors)
        random.seed(game_seed)
        random.shuffle(shuffled)

        agent_color = shuffled[0]
        player = RLGuardPlayer(agent_color, agent, rl_model)
        opponents = [opponent_class(c) for c in shuffled[1:num_players]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        try:
            game_obj = Game(all_players, vps_to_win=vps_to_win)
            winner = game_obj.play()
        except Exception as e:
            logger.warning(f"Game error (seed={game_seed}): {e}")
            results.append({"outcome": "ERROR", "error": str(e)})
            continue

        outcome = "WIN" if winner == agent_color else "LOSS"
        results.append({
            "outcome": outcome,
            "turns": game_obj.state.num_turns,
            "overrides": player.rl_overrides,
            "total_decisions": player.total_decisions,
            "override_rate": player.override_rate,
        })
        overrides_list.append(player.rl_overrides)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            wins = sum(1 for r in results if r["outcome"] == "WIN")
            logger.info(
                f"Game {i+1}/{num_games} | Wins: {wins}/{i+1} "
                f"({wins/(i+1)*100:.0f}%) | Avg overrides: {np.mean(overrides_list):.0f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

    total_time = time.time() - t_start
    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")

    logger.info("=" * 60)
    logger.info("  RL-Guard Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  RL Model: {RL_MODEL_PATH}")
    logger.info(f"  Games: {num_games} | Win rate: {wins}/{completed} ({wins/max(completed,1)*100:.1f}%)")
    logger.info(f"  Avg overrides/game: {np.mean(overrides_list):.0f}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/ab_sft/checkpoint-200/")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", type=str, default="weighted_random")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.model,
        num_games=args.games,
        opponent_type=args.opponent,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
