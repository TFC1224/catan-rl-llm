"""
SFT (Supervised Fine-Tuning) training for Catan agent.

Trains Qwen3-8B-Instruct to output valid Catan game actions by
imitating expert bot (VictoryPointPlayer) gameplay.

Uses TRL's SFTTrainer with LoRA for memory efficiency.
"""

import logging
import os
import yaml
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig

from .utils import load_model_and_tokenizer, setup_lora
from ..data.preprocessing import format_sft_example

logger = logging.getLogger(__name__)


def train_sft(
    model_name: str = "/root/autodl-tmp/Qwen/Qwen3-8B/",
    data_path: Optional[str] = None,
    train_data: Optional[Dataset] = None,
    eval_data: Optional[Dataset] = None,
    output_dir: str = "./checkpoints/sft/",
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """
    Run SFT training on Catan gameplay data.

    Args:
        model_name: HuggingFace model ID
        data_path: Path to JSONL data directory (with train.jsonl and val.jsonl)
        train_data: Pre-loaded training Dataset (alternative to data_path)
        eval_data: Pre-loaded validation Dataset (alternative to data_path)
        output_dir: Directory to save checkpoints
        config: Optional config dict (overrides defaults)
        **kwargs: Additional keyword arguments for SFTConfig

    Returns:
        Path to the saved LoRA checkpoint
    """
    if config is None:
        config = {}

    # Load config
    sft_config = config.get("sft", {}).get("training", {})
    sft_config.update(kwargs)

    logger.info("=" * 60)
    logger.info("  Starting SFT Training")
    logger.info("=" * 60)

    # Load model
    logger.info("[1/4] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        load_in_4bit=True,
    )

    # Apply LoRA
    logger.info("[2/4] Setting up LoRA...")
    model = setup_lora(
        model,
        r=config.get("lora", {}).get("r", 16),
        alpha=config.get("lora", {}).get("alpha", 32),
        dropout=config.get("lora", {}).get("dropout", 0.05),
    )

    # Load data
    logger.info("[3/4] Loading training data...")
    if data_path:
        train_dataset = load_dataset("json", data_files=os.path.join(data_path, "train.jsonl"), split="train")
        eval_dataset = load_dataset("json", data_files=os.path.join(data_path, "val.jsonl"), split="train")
        logger.info(f"  Train: {len(train_dataset)} examples | Val: {len(eval_dataset)} examples")
    elif train_data is not None:
        train_dataset = train_data
        eval_dataset = eval_data
    else:
        raise ValueError("Either data_path or train_data must be provided")

    # Format data with chat template
    def format_fn(example):
        return format_sft_example(
            system_prompt=example.get("system_prompt", ""),
            observation_text=example.get("observation", ""),
            action_json=example.get("action", '{"action_number": 0}'),
            tokenizer=tokenizer,
        )

    train_dataset = train_dataset.map(format_fn)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(format_fn)

    # Configure SFT
    sft_training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_config.get("num_train_epochs", 3),
        per_device_train_batch_size=sft_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=sft_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 4),
        learning_rate=sft_config.get("learning_rate", 2.0e-4),
        lr_scheduler_type=sft_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_config.get("warmup_ratio", 0.1),
        logging_steps=sft_config.get("logging_steps", 10),
        save_steps=sft_config.get("save_steps", 200),
        save_strategy="steps",
        eval_steps=sft_config.get("eval_steps", 200),
        eval_strategy="steps",
        max_length=sft_config.get("max_seq_length", 2048),
        dataset_text_field="text",
        packing=False,
        bf16=True,
        report_to=sft_config.get("report_to", "none"),
        run_name=sft_config.get("run_name", "catan-sft"),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("[4/4] Training...")
    train_result = trainer.train()

    # Save final checkpoint
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log results
    logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")
    logger.info(f"Checkpoint saved to: {output_dir}")

    return output_dir
