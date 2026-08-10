#!/usr/bin/env python3
"""
Run comprehensive evaluation of the trained Catan agent.

Plays the agent against multiple opponent types on different maps
and generates a detailed performance report with plots.

Usage:
    python scripts/evaluate.py --model checkpoints/grpo/iter4/ --output results/
    python scripts/evaluate.py --model checkpoints/sft/ --games 50
"""

import argparse
import logging
import os
import sys
import json
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.catan_rl.agent.qwen_agent import QwenCatanAgent
from src.catan_rl.eval.arena import run_tournament
from src.catan_rl.eval.metrics import compute_metrics, format_metrics_table
from src.catan_rl.eval.visualize import plot_win_rates, plot_learning_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Catan agent against opponents"
    )
    parser.add_argument(
        "--model", type=str, default="checkpoints/sft/",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval_config.yaml",
        help="Path to evaluation config"
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games per matchup (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        eval_config = config.get("evaluation", {})
    else:
        eval_config = {}

    logger.info("=" * 60)
    logger.info("  Catan Agent Evaluation")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Games per matchup: {args.games}")
    logger.info("=" * 60)

    # Load agent
    logger.info("\nLoading agent...")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        lora_path=args.model,
        prompt_version="v1",
    )

    # Define matchups
    matchups = eval_config.get("matchups", [
        {"map_type": "MINI", "vps_to_win": 6, "opponents": ["WeightedRandomPlayer"]},
        {"map_type": "BASE", "vps_to_win": 10, "opponents": ["WeightedRandomPlayer"]},
        {"map_type": "MINI", "vps_to_win": 6, "opponents": ["VictoryPointPlayer"]},
        {"map_type": "BASE", "vps_to_win": 10, "opponents": ["VictoryPointPlayer"]},
    ])

    all_results = {}
    all_metrics = {}

    for matchup in matchups:
        map_type = matchup["map_type"]
        vps = matchup["vps_to_win"]
        opponents = matchup["opponents"]
        opponent_label = opponents[0] if opponents else "Unknown"

        label = f"{opponent_label} ({map_type}, {vps}VP)"
        logger.info(f"\nEvaluating: {label}")

        results = run_tournament(
            agent=agent,
            opponents=opponents,
            num_games=args.games,
            map_type=map_type,
            vps_to_win=vps,
            seed=args.seed,
        )

        metrics = compute_metrics(results)
        all_results[label] = metrics
        all_metrics[label] = metrics

        logger.info(f"\n{format_metrics_table(metrics)}")

    # Generate plots
    logger.info("\nGenerating plots...")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "plots"), exist_ok=True)

    # Win rate plot
    plot_win_rates(
        all_metrics,
        output_path=os.path.join(args.output, "plots", "win_rates.png"),
        title=f"Catan Agent Win Rates ({args.games} games each)",
    )

    # Save metrics JSON
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("  Evaluation Complete")
    logger.info("=" * 60)
    for label, metrics in all_metrics.items():
        logger.info(f"  {label}: {metrics['win_rate']:.1%} win rate")


if __name__ == "__main__":
    main()
