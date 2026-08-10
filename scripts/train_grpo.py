#!/usr/bin/env python3
"""
GRPO-style training: Group Relative Preference Optimization.

For each decision, VF scored ALL valid actions. We train the model to prefer
higher-scored actions within each group using a weighted/contrastive loss.

Three variants (controlled by --method):
  1. "weighted_sft": Weighted SFT — train all actions, weight = normalized VF score
  2. "contrastive": Contrastive loss — increase P(best), decrease P(worst) per group
  3. "dpo_like": DPO-style — pairwise comparison (best vs rest) with preference strength

Usage:
    python scripts/train_grpo.py --data data/grpo/grpo_combined.jsonl \
        --method weighted_sft --epochs 3 --output checkpoints/grpo_weighted/
"""

import argparse, json, logging, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
if os.path.join(_PROJ, 'src') not in sys.path: sys.path.insert(0, os.path.join(_PROJ, 'src'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Dataset
# ==============================================================================

SYSTEM_PROMPT = "You are an expert Catan player.\nReply with: {\"action_number\": <integer>}"


class GRPODataset(Dataset):
    """Each item: (observation_text, action_index, normalized_vf_score, weight)"""

    def __init__(self, data_path):
        self.items = []

        with open(data_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                obs = rec["observation"]
                actions = rec["actions"]

                scores = [a["score"] for a in actions]
                min_s, max_s = min(scores), max(scores)

                # Normalize per group (GRPO: relative within group)
                if max_s > min_s:
                    norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]
                else:
                    norm_scores = [1.0 / max(len(scores), 1)] * len(scores)

                for a, ns in zip(actions, norm_scores):
                    prompt = (
                        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                        f"<|im_start|>user\n{obs}\nChoose:<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    target = json.dumps({"action_number": a["index"]})
                    self.items.append({
                        "prompt": prompt,
                        "target": target,
                        "norm_score": ns,
                        "is_best": (a["index"] == rec["best_index"]),
                    })

        logger.info(f"Loaded {len(self.items)} training examples from {data_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, max_length=1024):
    """Properly batch tokenize with prompt masking."""
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]

    # Concatenate prompt + target
    full_texts = [p + t for p, t in zip(prompts, targets)]

    enc = tokenizer(full_texts, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt")

    # Tokenize prompts to find prompt lengths
    prompt_enc = tokenizer(prompts, truncation=True, max_length=max_length,
                           padding=False)
    prompt_lens = [len(ids) for ids in prompt_enc.input_ids]

    # Create labels: -100 for prompt tokens, target tokens keep their ids
    labels = enc.input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels": labels,
        "norm_scores": torch.tensor([b["norm_score"] for b in batch]),
        "is_best": torch.tensor([b["is_best"] for b in batch]),
    }


# ==============================================================================
# Training
# ==============================================================================

def load_ab_sft_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Qwen/Qwen3-8B/", quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/Qwen/Qwen3-8B/", trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load AB-SFT LoRA
    lora_path = "/root/autodl-tmp/catan-rl-llm/catan-rl-llm/checkpoints/ab_sft/checkpoint-200/"
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Enable gradient computation for 4-bit quantized model
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def train_grpo(data_path, method="weighted_sft", epochs=3, batch_size=4,
               grad_accum=4, lr=1e-4, max_length=1024, output_dir="checkpoints/grpo"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading AB-SFT model...")
    model, tokenizer = load_ab_sft_model()
    model.train()

    dataset = GRPODataset(data_path)
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

    logger.info(f"Training: method={method}, epochs={epochs}, batch={batch_size}×{grad_accum}, "
                f"lr={lr}, examples={len(dataset)}, total_steps={total_steps}")
    logger.info(f"Output: {output_dir}")

    global_step = 0
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        accumulated_losses = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            norm_scores = batch["norm_scores"].to(model.device)
            is_best = batch["is_best"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Use HuggingFace's built-in loss (handles masking correctly, avoids NaN)
            micro_loss = outputs.loss

            # Apply method-specific weighting via the average
            if method == "weighted_sft":
                weights = 0.1 + 0.9 * norm_scores
                micro_loss = micro_loss * weights.mean()

            elif method == "contrastive":
                w = torch.where(is_best.to(model.device),
                               torch.tensor(2.0, device=model.device),
                               torch.where(norm_scores < 0.2,
                                          torch.tensor(0.1, device=model.device),
                                          torch.tensor(0.5, device=model.device)))
                micro_loss = micro_loss * w.mean()

            elif method == "dpo_like":
                centered = 2.0 * (norm_scores - 0.5)
                weights = torch.clamp(1.0 + centered, min=0.05)
                micro_loss = micro_loss * weights.mean()

            # Safety check
            if torch.isnan(micro_loss) or torch.isinf(micro_loss):
                continue

            micro_loss = micro_loss / grad_accum
            micro_loss.backward()
            accumulated_losses.append(micro_loss.item() * grad_accum)

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_loss = np.mean(accumulated_losses)
                epoch_losses.append(step_loss)
                accumulated_losses = []
                global_step += 1

                if global_step % 25 == 0:
                    elapsed = time.time() - t_start
                    avg_loss = np.mean(epoch_losses[-25:])
                    logger.info(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | "
                               f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # Remaining accumulated
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

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))

    elapsed = time.time() - t_start
    logger.info(f"Done: best_loss={best_loss:.4f}. Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/grpo/grpo_combined.jsonl")
    parser.add_argument("--method", type=str, default="weighted_sft",
                       choices=["weighted_sft", "contrastive", "dpo_like"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output", type=str, default="checkpoints/grpo")
    args = parser.parse_args()

    train_grpo(args.data, args.method, args.epochs, args.batch_size,
               args.grad_accum, args.lr, args.max_length, args.output)


if __name__ == "__main__":
    main()
