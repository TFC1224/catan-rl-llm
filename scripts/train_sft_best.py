#!/usr/bin/env python3
"""
Fast SFT training on best-action data (VF-Distill style, with richer GRPO data).

Trains only on the VF-best action per decision group — pure SFT, no weighting.
Fast convergence, stable training, proven approach (VF-Distill v2 baseline).

Usage:
    python scripts/train_sft_best.py --data data/grpo/grpo_sft_all.jsonl \
        --epochs 3 --output checkpoints/grpo_sft_all/
"""

import argparse, json, logging, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel, prepare_model_for_kbit_training

_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
if os.path.join(_PROJ, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ, 'src'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an expert Catan player.\nReply with: {\"action_number\": <integer>}"


class BestActionDataset(Dataset):
    """Each item: (observation, best_action_index) — standard SFT format."""

    def __init__(self, data_path):
        self.items = []
        with open(data_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                prompt = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{rec['observation']}\nChoose:<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = json.dumps({"action_number": rec["action_index"]})
                self.items.append({"prompt": prompt, "target": target})
        logger.info(f"Loaded {len(self.items)} examples from {data_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, max_length=768):
    """Tokenize with proper label masking (prompt tokens get -100)."""
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]

    full_texts = [p + t for p, t in zip(prompts, targets)]
    enc = tokenizer(full_texts, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt")

    # Mask prompt tokens
    prompt_enc = tokenizer(prompts, truncation=True, max_length=max_length, padding=False)
    prompt_lens = [len(ids) for ids in prompt_enc.input_ids]

    labels = enc.input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Qwen/Qwen3-8B/", quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/Qwen/Qwen3-8B/", trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_path = "/root/autodl-tmp/catan-rl-llm/catan-rl-llm/checkpoints/ab_sft/checkpoint-200/"
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def train(data_path, epochs=3, batch_size=2, grad_accum=4, lr=5e-5,
          max_length=768, output_dir="checkpoints/sft_best"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model...")
    model, tokenizer = load_model()
    model.train()

    dataset = BestActionDataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(10, total_steps // 10),
        num_training_steps=total_steps,
    )

    logger.info(f"Training: epochs={epochs}, batch={batch_size}×{grad_accum}, "
                f"lr={lr}, examples={len(dataset)}, total_steps={total_steps}")
    logger.info(f"Output: {output_dir}")

    global_step = 0
    best_loss = float('inf')
    t_start = time.time()
    accumulated_losses = []

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

            if torch.isnan(loss):
                continue

            loss.backward()
            accumulated_losses.append(loss.item() * grad_accum)

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_loss = np.mean(accumulated_losses)
                epoch_losses.append(step_loss)
                accumulated_losses = []
                global_step += 1

                if global_step % 50 == 0:
                    elapsed = time.time() - t_start
                    avg_loss = np.mean(epoch_losses[-50:])
                    logger.info(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | "
                               f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # Remaining gradients
        if accumulated_losses:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_losses.append(np.mean(accumulated_losses))
            accumulated_losses = []

        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        elapsed = time.time() - t_start
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | {elapsed:.0f}s")

        # Save
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))

        # Early stop: if loss doesn't improve much, break
        if epoch > 0 and best_loss / avg_epoch_loss > 0.95:
            logger.info(f"Early stopping: loss plateau (best={best_loss:.4f}, current={avg_epoch_loss:.4f})")
            break

    elapsed = time.time() - t_start
    logger.info(f"Done: best_loss={best_loss:.4f}. Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/grpo/grpo_sft_all.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--output", type=str, default="checkpoints/sft_best")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size, args.grad_accum,
          args.lr, args.max_length, args.output)


if __name__ == "__main__":
    main()
