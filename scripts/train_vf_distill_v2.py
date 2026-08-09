#!/usr/bin/env python3
"""
Option A v2: VF Distillation — Catanatron AlphaBeta-supervised methodology.

Key fixes over v1 (which got 20% WR):
1. OVERRIDE-ONLY: Only train on decisions where VF corrected LLM (strongest signal)
2. AB-SFT INIT: Start from AB-SFT checkpoint, not base model
3. VF-ENRICHED OBS: Observations include VF score comparison (why VF chose X over Y)
4. OUTCOME-AWARE: Records include WIN/LOSS outcome (Catanatron-style blending)
5. LOWER LR: 1e-4 for fine-tuning (matching Catanatron's approach)

Catanatron reference: train_r1_alphabeta_supervised.py
  - Records (features, ab_score) at EVERY decision
  - Min-max normalizes per episode
  - Blends 80% AB score + 20% outcome ramp
  - Fine-tunes with LR=1e-4

Usage:
    # Step 1: Generate enriched data
    python scripts/generate_vf_distill_data_v2.py --games 150 --output data/vf_distill_v2/

    # Step 2: Train
    python scripts/train_vf_distill_v2.py --data data/vf_distill_v2/ --output checkpoints/vf_distill_v2/
"""

import argparse
import json
import logging
import os
import sys

import torch

_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, 'src'))

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from src.catan_rl.data.preprocessing import format_sft_example

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def load_base_model(model_name="/root/autodl-tmp/Qwen/Qwen3-8B/"):
    """Load base Qwen3-8B in 4-bit using BitsAndBytesConfig."""
    logger.info(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_ab_sft_lora(base_model, checkpoint_path):
    """Load AB-SFT LoRA adapter — continue from existing game knowledge."""
    logger.info(f"Loading AB-SFT LoRA: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model


def setup_fresh_lora(model):
    """Set up fresh LoRA (baseline mode)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/vf_distill_v2/")
    parser.add_argument("--output", type=str, default="checkpoints/vf_distill_v2/")
    parser.add_argument("--ab_sft_ckpt", type=str,
                        default="checkpoints/ab_sft/checkpoint-200/")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--fresh_lora", action="store_true",
                        help="Use fresh LoRA instead of AB-SFT checkpoint (baseline)")
    parser.add_argument("--all_data", action="store_true",
                        help="Train on ALL data, not just overrides (baseline)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # === Load data ===
    train_path = os.path.join(args.data, "train.jsonl")
    val_path = os.path.join(args.data, "val.jsonl")
    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run: python scripts/generate_vf_distill_data_v2.py --games 150 first")
        sys.exit(1)

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path) if os.path.exists(val_path) else []
    logger.info(f"Loaded: {len(train_data)} train, {len(val_data)} val")

    # === Filter to override-only (FIX #1) ===
    if not args.all_data:
        train_orig = len(train_data)
        train_data = [r for r in train_data if r.get("was_override", False)]
        val_data = [r for r in val_data if r.get("was_override", False)]
        logger.info(f"Override filter: {len(train_data)}/{train_orig} train, {len(val_data)} val")
    else:
        logger.info("Using ALL data (not just overrides)")

    wins = sum(1 for r in train_data if r.get("outcome") == "WIN")
    logger.info(f"Train outcomes: {wins} WIN, {len(train_data) - wins} LOSS")

    if len(train_data) < 50:
        logger.error(f"Too few training records: {len(train_data)}")
        sys.exit(1)

    # === Load model (FIX #2: start from AB-SFT checkpoint) ===
    model, tokenizer = load_base_model()

    if args.fresh_lora:
        logger.info("FRESH LoRA mode — baseline comparison")
        model = setup_fresh_lora(model)
    else:
        if not os.path.exists(args.ab_sft_ckpt):
            logger.error(f"AB-SFT checkpoint not found: {args.ab_sft_ckpt}")
            sys.exit(1)
        model = load_ab_sft_lora(model, args.ab_sft_ckpt)
        logger.info("AB-SFT LoRA loaded — continuing from game-knowledge checkpoint")

    # === Format data ===
    def format_fn(example):
        return format_sft_example(
            system_prompt=example.get("system_prompt", ""),
            observation_text=example.get("observation", ""),
            action_json=example.get("action", '{"action_number": 0}'),
            tokenizer=tokenizer,
        )

    train_dataset = Dataset.from_list(train_data).map(format_fn)
    val_dataset = Dataset.from_list(val_data).map(format_fn) if val_data else None

    # === Train (FIX #5: Catanatron-style low LR) ===
    sft_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=999999, save_strategy="no",
        eval_steps=999999, eval_strategy="no",
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False, bf16=True,
        report_to="none", run_name="catan-vf-distill-v2",
    )

    eff_batch = args.batch_size * args.grad_accum
    logger.info(f"Training: {len(train_data)} examples × {args.epochs} epochs")
    logger.info(f"  Batch: {args.batch_size}×{args.grad_accum}={eff_batch}, LR: {args.lr}")
    logger.info(f"  Init: {'Fresh LoRA' if args.fresh_lora else 'AB-SFT LoRA'}")
    logger.info(f"  Data: {'All' if args.all_data else 'Override-only'}")

    trainer = SFTTrainer(
        model=model, args=sft_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    logger.info(f"Training complete. Loss: {train_result.training_loss:.4f}")
    logger.info(f"Model saved to: {args.output}")
    logger.info(f"Evaluate: python scripts/eval_ab_sft.py --model {args.output} --games 20")


if __name__ == "__main__":
    main()
