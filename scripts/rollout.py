#!/usr/bin/env python3
"""
Collect game rollout data for GRPO training.

Plays games with the current (possibly trained) agent model and
records (state, valid_actions, model_response, game_outcome) tuples.

Usage:
    python scripts/rollout.py --model checkpoints/sft/ --output data/grpo/iter1/ --num_games 100
    python scripts/rollout.py --model checkpoints/grpo/iter1/ --map BASE --vps 10 --output data/grpo/iter3/
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.catan_rl.agent.qwen_agent import QwenCatanAgent
from src.catan_rl.env.catan_env import make_catan_env
from src.catan_rl.data.rollout import play_game_with_agent
from src.catan_rl.data.grpo_dataset import save_rollout_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Collect game rollout data for GRPO training"
    )
    parser.add_argument(
        "--model", type=str, default="checkpoints/sft/",
        help="Path to model/LoRA checkpoint (default: checkpoints/sft/)"
    )
    parser.add_argument(
        "--map", type=str, default="MINI",
        choices=["MINI", "BASE"],
        help="Map type (default: MINI)"
    )
    parser.add_argument(
        "--vps", type=int, default=6,
        help="Victory points to win (default: 6)"
    )
    parser.add_argument(
        "--opponents", type=str, nargs="+",
        default=["WeightedRandomPlayer"],
        help="Opponent bot types (default: WeightedRandomPlayer)"
    )
    parser.add_argument(
        "--num_games", type=int, default=100,
        help="Number of games to play (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default="data/grpo/rollout/",
        help="Output directory (default: data/grpo/rollout/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Game Rollout Collection")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Map: {args.map}, VPs: {args.vps}")
    logger.info(f"  Opponents: {args.opponents}")
    logger.info(f"  Games: {args.num_games}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    # Load agent
    logger.info("Loading agent...")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        lora_path=args.model,
        prompt_version="v1",
    )

    all_records = []
    outcomes = {"WIN": 0, "LOSS": 0, "DRAW": 0}

    for game_idx in range(args.num_games):
        env = make_catan_env(
            map_type=args.map,
            vps_to_win=args.vps,
            opponents=args.opponents,
        )

        game_result = play_game_with_agent(
            agent=agent,
            env=env,
            player_index=0,
            record_trajectory=True,
        )

        outcome = game_result["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        all_records.extend(game_result.get("records", []))

        env.close()

        if (game_idx + 1) % 10 == 0:
            logger.info(
                f"  Game {game_idx + 1}/{args.num_games}: "
                f"W:{outcomes['WIN']} L:{outcomes['LOSS']} D:{outcomes['DRAW']}"
            )

    logger.info(f"Rollout complete: {len(all_records)} state records from {args.num_games} games")
    logger.info(f"Outcomes: {outcomes}")

    save_rollout_data(all_records, args.output)
    logger.info(f"Data saved to: {args.output}")


if __name__ == "__main__":
    main()
