#!/usr/bin/env python3
"""
Train Simulation-Guided SFT: fine-tune on improved action data.

Loads the existing SFT LoRA checkpoint and does additional fine-tuning
on the Simulation-Guided data (best actions found by exhaustive search).

Usage:
    python scripts/train_simsft.py \
        --lora checkpoints/sft/ \
        --data data/simsft/iter1/ \
        --output checkpoints/simsft/iter1/
"""

import argparse
import json
import logging
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trl import SFTTrainer, SFTConfig
from src.catan_rl.training.utils import (
    load_model_and_tokenizer,
    load_lora_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_simsft(
    lora_path: str = "checkpoints/sft/",
    data_path: str = "data/simsft/iter1/",
    output_dir: str = "checkpoints/simsft/iter1/",
    model_name: str = "/root/autodl-tmp/Qwen/Qwen3-8B/",
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    grad_accum: int = 4,
    save_steps: int = 50,
    seed: int = 42,
):
    """
    Fine-tune an existing SFT model on Simulation-Guided data.

    Args:
        lora_path: Path to existing SFT LoRA checkpoint
        data_path: Path to simsft data directory (train.jsonl + val.jsonl)
        output_dir: Where to save the fine-tuned checkpoint
        model_name: Base model name (for tokenizer)
        num_epochs: Number of training epochs
        learning_rate: Learning rate (lower than initial SFT since we're fine-tuning)
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        save_steps: Checkpoint save frequency
        seed: Random seed
    """
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("  Simulation-Guided SFT Training")
    logger.info("=" * 60)

    # Load base model + LoRA checkpoint
    logger.info("[1/4] Loading base model and SFT LoRA checkpoint...")
    base_model, tokenizer = load_model_and_tokenizer(
        model_name, load_in_4bit=True,
    )
    model = load_lora_checkpoint(base_model, lora_path)
    logger.info(f"  LoRA loaded from: {lora_path}")

    # Load data
    logger.info("[2/4] Loading SimSFT data...")
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_path, "train.jsonl"),
        split="train",
    )
    val_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_path, "val.jsonl"),
        split="train",
    )
    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Format data: system_prompt + observation + action → chat template
    logger.info("[3/4] Formatting data with chat template...")

    def format_fn(example):
        system_prompt = example.get("system_prompt", "")
        observation = example.get("observation", "")
        action = example.get("action", '{"action_number": 0}')

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
            {"role": "assistant", "content": action},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(format_fn)
    val_dataset = val_dataset.map(format_fn)

    # Configure SFT
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=save_steps,
        save_strategy="steps",
        eval_steps=save_steps,
        eval_strategy="steps",
        max_length=2048,
        dataset_text_field="text",
        packing=False,
        bf16=True,
        report_to="none",
        run_name="catan-simsft",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("[4/4] Training...")
    logger.info(f"  Epochs: {num_epochs}, LR: {learning_rate}")
    logger.info(f"  Batch size: {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    logger.info(f"  Train steps: ~{num_epochs * len(train_dataset) // (batch_size * grad_accum)}")

    train_result = trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")
    logger.info(f"Checkpoint: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train Simulation-Guided SFT")
    parser.add_argument("--lora", type=str, default="checkpoints/sft/")
    parser.add_argument("--data", type=str, default="data/simsft/iter1/")
    parser.add_argument("--output", type=str, default="checkpoints/simsft/iter1/")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/Qwen/Qwen3-8B/")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_simsft(
        lora_path=args.lora,
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        save_steps=args.save_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
