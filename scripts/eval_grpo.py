#!/usr/bin/env python3
"""
Evaluate GRPO-trained models standalone (no VF guardrail at inference).

Usage:
    python scripts/eval_grpo.py --model checkpoints/grpo/checkpoint-3/ --games 5
"""

import argparse, json, logging, os, random, sys, time
import numpy as np
import torch

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path: sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catan_rl.agent.qwen_agent import QwenCatanAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StandalonePlayer(Player):
    """Pure LLM — no guardrail. Tests GRPO model quality."""

    def __init__(self, color, agent):
        super().__init__(color); self.agent = agent; self.total = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1: self.total += 1; return actions[0] if actions else None
        self.total += 1
        try:
            r = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            idx = r.action_index
            if not (0 <= idx < len(actions)): idx = 0
        except Exception: idx = 0
        return actions[idx]


def run_eval(checkpoint_path, num_games=5, seed=42):
    random.seed(seed); np.random.seed(seed)

    logger.info(f"Loading GRPO model: {checkpoint_path}")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/", device="cuda", load_in_4bit=True,
        lora_path=checkpoint_path, prompt_version="v1")
    agent.max_new_tokens = 16; agent.temperature = 0.1; agent.do_sample = True

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    results = []; t_start = time.time()

    for i in range(num_games):
        gs = seed + i * 100
        random.seed(gs); shuffled = list(colors); random.shuffle(shuffled)
        ac = shuffled[0]
        player = StandalonePlayer(ac, agent)
        opponents = [WeightedRandomPlayer(c) for c in shuffled[1:]]
        all_players = [player] + opponents; random.shuffle(all_players)

        logger.info(f"Game {i+1}/{num_games} (seed={gs})...")
        gt = time.time()
        try:
            game = Game(all_players, vps_to_win=10); winner = game.play()
            outcome = "WIN" if winner == ac else "LOSS"
        except Exception as e:
            logger.warning(f"Error: {e}"); outcome = "ERROR"

        turns = game.state.num_turns if hasattr(game, 'state') else 0
        game_time = time.time() - gt
        torch.cuda.empty_cache()
        results.append({"outcome": outcome, "turns": turns, "game_time_s": game_time})
        wins = sum(1 for r in results if r["outcome"] == "WIN")
        elapsed = time.time() - t_start
        logger.info(f"  Game {i+1}/{num_games} | {wins}W/{i+1-wins}L | "
                   f"{turns}t/{game_time:.0f}s | {elapsed:.0f}s total")

    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")
    wr = wins / max(completed, 1)
    total_time = time.time() - t_start
    logger.info(f"GRPO Result: {wins}/{completed} ({wr:.1%}) | {total_time:.0f}s ({total_time/60:.1f}min)")
    return {"win_rate": wr, "wins": wins, "games": completed, "time_s": total_time}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="checkpoints/grpo/checkpoint-3/")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_eval(args.model, args.games, args.seed)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f: json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
