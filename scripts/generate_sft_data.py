#!/usr/bin/env python3
"""
Generate SFT training data from expert bot gameplay.

Plays games with VictoryPointPlayer as the "expert" against
WeightedRandomPlayer opponents, recording (state, action) pairs
for supervised fine-tuning.

Usage:
    python scripts/generate_sft_data.py --num_games 500 --output data/sft/
    python scripts/generate_sft_data.py --num_games 100 --map BASE --vps 10 --output data/sft_base/
"""

import argparse
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer

from src.catan_rl.data.sft_dataset import generate_sft_data_from_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT training data from expert bot gameplay"
    )
    parser.add_argument(
        "--num_games", type=int, default=700,
        help="Number of games to play (default: 700)"
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
        "--expert", type=str, default="VictoryPointPlayer",
        choices=["VictoryPointPlayer", "WeightedRandomPlayer"],
        help="Expert bot type (default: VictoryPointPlayer)"
    )
    parser.add_argument(
        "--output", type=str, default="data/sft/",
        help="Output directory (default: data/sft/)"
    )
    parser.add_argument(
        "--train_split", type=float, default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Select expert bot
    expert_map = {
        "VictoryPointPlayer": VictoryPointPlayer,
        "WeightedRandomPlayer": WeightedRandomPlayer,
    }
    expert_bot = expert_map[args.expert]

    logger.info("=" * 60)
    logger.info("  SFT Data Generation")
    logger.info(f"  Expert: {args.expert}")
    logger.info(f"  Map: {args.map}, VPs: {args.vps}")
    logger.info(f"  Games: {args.num_games}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    train_records, val_records = generate_sft_data_from_bot(
        bot_class=expert_bot,
        num_games=args.num_games,
        map_type=args.map,
        vps_to_win=args.vps,
        output_dir=args.output,
        train_split=args.train_split,
        seed=args.seed,
    )

    logger.info(f"Done! Generated {len(train_records)} train + {len(val_records)} val records")
    logger.info(f"Data saved to: {args.output}")


if __name__ == "__main__":
    main()
