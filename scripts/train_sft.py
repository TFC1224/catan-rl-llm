#!/usr/bin/env python3
"""
Run SFT (Supervised Fine-Tuning) training for the Catan agent.

Usage:
    python scripts/train_sft.py --data data/sft/ --output checkpoints/sft/
    python scripts/train_sft.py --config configs/sft_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.catan_rl.training.train_sft import train_sft

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run SFT training for Catan agent"
    )
    parser.add_argument(
        "--config", type=str, default="configs/sft_config.yaml",
        help="Path to SFT config YAML (default: configs/sft_config.yaml)"
    )
    parser.add_argument(
        "--model", type=str, default="/root/autodl-tmp/Qwen/Qwen3-8B/",
        help="Base model name or path (default: /root/autodl-tmp/Qwen/Qwen3-8B/)"
    )
    parser.add_argument(
        "--data", type=str, default="data/sft/",
        help="Path to SFT data directory (default: data/sft/)"
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/sft/",
        help="Output directory for checkpoints (default: checkpoints/sft/)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override per-device batch size"
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

    # Build override kwargs
    overrides = {}
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size

    logger.info("=" * 60)
    logger.info("  SFT Training")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output: {args.output}")
    if overrides:
        logger.info(f"  Overrides: {overrides}")
    logger.info("=" * 60)

    checkpoint_path = train_sft(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        config=config,
        **overrides,
    )

    logger.info(f"Training complete! Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
