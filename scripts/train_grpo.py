#!/usr/bin/env python3
"""
Run GRPO (Group Relative Policy Optimization) training for Catan agent.

Usage:
    python scripts/train_grpo.py --data data/grpo/iter1/ --output checkpoints/grpo/iter1/
    python scripts/train_grpo.py --config configs/grpo_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.catan_rl.training.train_grpo import train_grpo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Catan agent"
    )
    parser.add_argument(
        "--config", type=str, default="configs/grpo_config.yaml",
        help="Path to GRPO config YAML"
    )
    parser.add_argument(
        "--model", type=str, default="/root/autodl-tmp/Qwen/Qwen3-8B/",
        help="Base model name"
    )
    parser.add_argument(
        "--lora", type=str, default="checkpoints/sft/",
        help="Path to SFT LoRA checkpoint (starting point for RL)"
    )
    parser.add_argument(
        "--data", type=str, default="data/grpo/rollout/",
        help="Path to rollout dataset directory"
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/grpo/",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Override KL penalty coefficient"
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")

    # Build overrides
    overrides = {}
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.beta is not None:
        overrides["beta"] = args.beta

    logger.info("=" * 60)
    logger.info("  GRPO Training")
    logger.info(f"  Base model: {args.model}")
    logger.info(f"  LoRA checkpoint: {args.lora}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    checkpoint_path = train_grpo(
        model_name=args.model,
        lora_path=args.lora if os.path.exists(args.lora) else None,
        dataset_path=args.data,
        output_dir=args.output,
        config=config,
        **overrides,
    )

    logger.info(f"GRPO training complete! Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
