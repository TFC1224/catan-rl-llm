#!/usr/bin/env python3
"""Evaluate GRPO SFT models (standalone, no guardrail) vs WeightedRandom."""

import argparse, json, logging, os, random, sys, time
import numpy as np
import torch

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CAT_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CAT_ROOT, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catan_rl.agent.qwen_agent import QwenCatanAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StandalonePlayer(Player):
    """Pure LLM decision — tests model quality without guardrails."""

    def __init__(self, color, agent):
        super().__init__(color)
        self.agent = agent
        self.total = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total += 1
            return actions[0] if actions else None
        self.total += 1
        try:
            r = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            idx = r.action_index
            if not (0 <= idx < len(actions)):
                idx = 0
        except Exception:
            idx = 0
        return actions[idx]


def evaluate(checkpoint_path, name, num_games=5, seed=42):
    logger.info(f"\n{'='*50}\n  {name}\n{'='*50}")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/", device="cuda", load_in_4bit=True,
        lora_path=checkpoint_path, prompt_version="v1",
    )
    agent.max_new_tokens = 16
    agent.temperature = 0.1
    agent.do_sample = True

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    results = []
    t_start = time.time()

    for i in range(num_games):
        gs = seed + i * 100
        random.seed(gs)
        shuffled = list(colors)
        random.shuffle(shuffled)
        ac = shuffled[0]
        player = StandalonePlayer(ac, agent)
        opponents = [WeightedRandomPlayer(c) for c in shuffled[1:]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        logger.info(f"[{name}] Game {i+1}/{num_games} (seed={gs})...")
        gt = time.time()
        try:
            game = Game(all_players, vps_to_win=10)
            winner = game.play()
            outcome = "WIN" if winner == ac else "LOSS"
        except Exception as e:
            logger.warning(f"Error: {e}")
            outcome = "ERROR"

        turns = game.state.num_turns if hasattr(game, 'state') else 0
        game_time = time.time() - gt
        torch.cuda.empty_cache()
        results.append({"outcome": outcome, "turns": turns, "game_time_s": game_time})
        wins = sum(1 for r in results if r["outcome"] == "WIN")
        elapsed = time.time() - t_start
        logger.info(f"  Game {i+1} | {wins}W/{i+1-wins}L | {turns}t/{game_time:.0f}s | {elapsed:.0f}s")

    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")
    wr = wins / max(completed, 1)
    total_time = time.time() - t_start
    logger.info(f"[{name}] RESULT: {wins}/{completed} ({wr:.1%}) | {total_time:.0f}s ({total_time/60:.1f}min)")
    return {"name": name, "win_rate": wr, "wins": wins, "games": completed, "time_s": total_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="all",
                       help="Comma-separated: all,filtered,balanced or 'all'")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    configs = {
        "all": ("checkpoints/grpo_sft_all/best", "GRPO-SFT-All"),
        "filtered": ("checkpoints/grpo_sft_filtered/best", "GRPO-SFT-Filtered"),
        "balanced": ("checkpoints/grpo_sft_balanced/best", "GRPO-SFT-Balanced"),
    }

    if args.models == "all":
        models = list(configs.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]

    all_results = []
    for model_key in models:
        ckpt, name = configs[model_key]
        if not os.path.exists(ckpt):
            logger.warning(f"Checkpoint not found: {ckpt}, skipping")
            continue
        result = evaluate(ckpt, name, args.games, args.seed)
        all_results.append(result)
        torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("  GRPO SFT COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  {'Method':25s} {'Win Rate':>8s} {'Games':>8s}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8}")
    logger.info(f"  {'AB-SFT (baseline)':25s} {'25.0%':>8s} {'5':>8s}")
    logger.info(f"  {'VF-Distill v2 (prev)':25s} {'40.0%':>8s} {'5':>8s}")
    for r in all_results:
        logger.info(f"  {r['name']:25s} {r['win_rate']:>7.1%} {r['games']:>8d}")
    logger.info("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
