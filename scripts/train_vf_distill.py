#!/usr/bin/env python3
"""
Option A: VF Distillation — Train SFT on VF-Guard corrected decisions.

This script trains the model to internalize VF preferences by learning from
VF-Guard's corrected decisions. After training, the model should approach
VF-Guard performance without needing the VF at inference time.

Usage:
    python scripts/train_vf_distill.py --data data/vf_distill/ --output checkpoints/vf_distill/
"""

import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets import Dataset
from src.catan_rl.data.preprocessing import format_sft_example
from src.catan_rl.training.train_sft import train_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    parser = argparse.ArgumentParser(description="Train VF Distillation model")
    parser.add_argument("--data", type=str, default="data/vf_distill/")
    parser.add_argument("--output", type=str, default="checkpoints/vf_distill/")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--override_only", action="store_true",
                        help="Only train on decisions where VF overrode LLM")
    args = parser.parse_args()

    # Load data
    train_path = os.path.join(args.data, "train.jsonl")
    val_path = os.path.join(args.data, "val.jsonl")

    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run: python scripts/generate_vf_distill_data.py first")
        sys.exit(1)

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path) if os.path.exists(val_path) else []

    logger.info(f"Loaded: {len(train_data)} train, {len(val_data)} val")

    # Optional: filter to only VF override decisions (stronger learning signal)
    if args.override_only:
        train_data = [r for r in train_data if r.get("was_override", False)]
        val_data = [r for r in val_data if r.get("was_override", False)]
        logger.info(f"After override filter: {len(train_data)} train, {len(val_data)} val")

    if len(train_data) == 0:
        logger.error("No training data after filtering!")
        sys.exit(1)

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    # Train
    config = {
        "sft": {
            "training": {
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "learning_rate": args.lr,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "logging_steps": 20,
                "save_steps": 999999,
                "eval_steps": 999999,
                "max_seq_length": 2048,
                "report_to": "none",
            }
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    }

    logger.info(f"Starting training: {len(train_data)} examples, {args.epochs} epochs")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"Learning rate: {args.lr}")

    output = train_sft(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        train_data=train_dataset,
        eval_data=val_dataset,
        output_dir=args.output,
        config=config,
        num_train_epochs=args.epochs,
        save_steps=999999,
        eval_steps=999999,
        save_strategy="no",
        eval_strategy="no",
    )

    logger.info(f"Training complete! Model saved to: {output}")
    logger.info(f"Evaluate with: python scripts/eval_ab_sft.py --model {output} --games 20")


if __name__ == "__main__":
    main()
