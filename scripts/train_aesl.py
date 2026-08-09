#!/usr/bin/env python3
"""
AESL (ICLR 2026) Diversity Early-Stopping Experiment for Catan SFT.

Key hypothesis: The checkpoint with highest output diversity (entropy peak)
produces better Catan gameplay than the checkpoint with lowest validation loss.

Pipeline:
1. Train SFT from base Qwen3-8B + fresh LoRA on AB-SFT data
2. Save checkpoint every N steps
3. At each checkpoint, compute entropy on validation prompts
4. Find entropy-peak vs best-loss checkpoints
5. Evaluate both on Catan games

Reference: "Getting Your LLMs Ready for Reinforcement Learning with
Lightweight SFT" (Li et al., ICLR 2026)

Usage:
    python scripts/train_aesl.py --train_size 5000 --val_size 200 \
        --entropy_val_size 50 --save_every 50 --output checkpoints/aesl/
"""

import argparse, json, logging, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig, get_peft_model, PeftModel,
    prepare_model_for_kbit_training,
)

_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
if os.path.join(_PROJ, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ, 'src'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Catan player.\n"
    "Reply with: {\"action_number\": <integer>}"
)


# ==============================================================================
# Dataset
# ==============================================================================

class SFTDataset(Dataset):
    """Load AB-SFT JSONL data: each line has system_prompt, observation, action."""

    def __init__(self, data_path):
        self.items = []
        with open(data_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                # Build chat-format prompt + target
                prompt = (
                    f"<|im_start|>system\n{rec['system_prompt']}<|im_end|>\n"
                    f"<|im_start|>user\n{rec['observation']}\nChoose:<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                target = rec['action']  # e.g. '{"action_number": 5}'
                self.items.append({"prompt": prompt, "target": target})
        logger.info(f"Loaded {len(self.items)} examples from {data_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, max_length=2048):
    """Tokenize with label masking: prompt tokens get -100."""
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


# ==============================================================================
# Model Loading
# ==============================================================================

def load_fresh_lora_model(model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
                          lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Load base model with fresh LoRA adapter (no existing checkpoint)."""
    logger.info(f"Loading base model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient computation for 4-bit layers (MUST come before get_peft_model)
    base_model = prepare_model_for_kbit_training(base_model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total params "
                f"({100*trainable/total:.2f}%)")

    return model, tokenizer


# ==============================================================================
# Entropy Computation (AESL-style diversity monitoring)
# ==============================================================================

@torch.no_grad()
def compute_entropy_metrics(model, tokenizer, eval_items, device, max_length=512):
    """
    Compute diversity metrics on validation prompts.

    Two metrics:
    1. token_entropy: Average entropy of model's predictive distribution
       over ground-truth target tokens. Higher = more diverse/uncertain.
    2. perplexity: exp(average NLL). Lower = better fit to data.

    The AESL paper uses token_entropy for diversity monitoring.
    """
    model.eval()
    total_entropy = 0.0
    total_nll = 0.0
    total_tokens = 0

    for item in eval_items:
        full_text = item["prompt"] + item["target"]
        enc = tokenizer(full_text, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)

        prompt_enc = tokenizer(item["prompt"], truncation=True, max_length=max_length, padding=False)
        prompt_len = len(prompt_enc.input_ids)

        # Forward pass
        outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        logits = outputs.logits  # [1, L, V]

        # For each target token at position j (prompt_len .. seq_len-1):
        #   prediction comes from logits at position j-1
        seq_len = enc.input_ids.shape[1]
        for j in range(prompt_len, seq_len):
            if j - 1 < 0 or j - 1 >= logits.shape[1]:
                continue
            tid = enc.input_ids[0, j].item()
            logits_at_pos = logits[0, j - 1, :]  # [V] — predicts token at position j
            lp = F.log_softmax(logits_at_pos, dim=-1)
            p = F.softmax(logits_at_pos, dim=-1)
            ent = -(p * torch.log(p + 1e-12)).sum().item()
            total_entropy += ent
            total_nll += -lp[tid].item()
            total_tokens += 1

    model.train()
    if total_tokens == 0:
        return {"token_entropy": 0.0, "perplexity": float('inf')}

    avg_entropy = total_entropy / total_tokens
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "token_entropy": avg_entropy,
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
    }


# ==============================================================================
# Training
# ==============================================================================

def train_aesl(
    train_data_path="/root/autodl-tmp/catan-rl-llm/catan-rl-llm/data/ab_sft/main/train.jsonl",
    val_data_path="/root/autodl-tmp/catan-rl-llm/catan-rl-llm/data/ab_sft/main/val.jsonl",
    train_size=5000,
    val_loss_size=200,
    val_entropy_size=50,
    epochs=1,
    batch_size=2,
    grad_accum=4,
    lr=2e-4,
    max_length=1024,
    save_every=50,
    output_dir="checkpoints/aesl",
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Load data ----
    full_train = SFTDataset(train_data_path)
    full_val = SFTDataset(val_data_path)

    # Subset for cold-start scale (paper uses 1k-6k)
    if train_size < len(full_train):
        indices = random.sample(range(len(full_train)), train_size)
        train_dataset = Subset(full_train, indices)
    else:
        train_dataset = full_train

    # Validation subsets
    val_indices = list(range(min(val_loss_size, len(full_val))))
    val_loss_items = [full_val[i] for i in val_indices]

    # Smaller set for entropy computation (faster)
    entropy_val_size = min(val_entropy_size, len(full_val))
    if entropy_val_size < len(full_val):
        ent_indices = random.sample(range(len(full_val)), entropy_val_size)
        val_entropy_items = [full_val[i] for i in ent_indices]
    else:
        val_entropy_items = [full_val[i] for i in range(len(full_val))]

    logger.info(f"Train: {len(train_dataset)} | Val loss: {len(val_loss_items)} "
                f"| Val entropy: {len(val_entropy_items)}")

    # ---- Load model ----
    model, tokenizer = load_fresh_lora_model()
    device = model.device

    # ---- DataLoader ----
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, total_steps // 10),
        num_training_steps=total_steps,
    )

    logger.info(f"Training: epochs={epochs}, batch={batch_size}×{grad_accum}, "
                f"lr={lr}, examples={len(train_dataset)}, "
                f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    logger.info(f"Checkpoint every {save_every} steps → ~{total_steps // save_every} checkpoints")
    logger.info(f"Output: {output_dir}")

    # ---- Training loop ----
    global_step = 0
    best_val_loss = float('inf')
    best_val_loss_step = 0
    peak_entropy = -float('inf')
    peak_entropy_step = 0
    accumulated_losses = []
    t_start = time.time()
    metrics_log = []

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at step {global_step}, skipping")
                continue

            # Also check logits for NaN (catches issues before backprop)
            if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                logger.warning(f"NaN/Inf in logits at step {global_step}, skipping")
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

                # ---- Checkpoint + Entropy Evaluation ----
                if global_step % save_every == 0:
                    elapsed = time.time() - t_start
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    # Compute diversity metrics
                    ent_start = time.time()
                    metrics = compute_entropy_metrics(
                        model, tokenizer, val_entropy_items, device, max_length,
                    )
                    ent_time = time.time() - ent_start

                    metrics["step"] = global_step
                    metrics["train_loss"] = np.mean(epoch_losses[-50:])
                    metrics["lr"] = scheduler.get_last_lr()[0]
                    metrics["elapsed_s"] = elapsed
                    metrics["entropy_time_s"] = ent_time
                    metrics_log.append(metrics)

                    # Track bests
                    if metrics["perplexity"] < best_val_loss:
                        best_val_loss = metrics["perplexity"]
                        best_val_loss_step = global_step
                        # Save best-loss checkpoint
                        best_loss_dir = os.path.join(output_dir, "best_loss")
                        model.save_pretrained(best_loss_dir)
                        tokenizer.save_pretrained(best_loss_dir)

                    if metrics["token_entropy"] > peak_entropy:
                        peak_entropy = metrics["token_entropy"]
                        peak_entropy_step = global_step
                        # Save entropy-peak checkpoint
                        best_ent_dir = os.path.join(output_dir, "best_entropy")
                        model.save_pretrained(best_ent_dir)
                        tokenizer.save_pretrained(best_ent_dir)

                    logger.info(
                        f"[Step {global_step}/{total_steps}] "
                        f"Loss: {metrics['train_loss']:.4f} | "
                        f"Entropy: {metrics['token_entropy']:.4f} | "
                        f"Perplexity: {metrics['perplexity']:.2f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"{elapsed:.0f}s (+{ent_time:.0f}s entropy)"
                    )

                elif global_step % (save_every // 5) == 0:
                    elapsed = time.time() - t_start
                    avg_loss = np.mean(epoch_losses[-save_every//5:])
                    logger.info(
                        f"[Step {global_step}/{total_steps}] "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"{elapsed:.0f}s"
                    )

        # End of epoch
        if accumulated_losses:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_losses.append(np.mean(accumulated_losses))
            accumulated_losses = []

        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        elapsed = time.time() - t_start
        logger.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f} | {elapsed:.0f}s")

    # ---- Save final results ----
    # Final checkpoint
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    # Save metrics log
    metrics_path = os.path.join(output_dir, "aesl_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Summary
    summary = {
        "best_loss_step": best_val_loss_step,
        "best_loss_perplexity": best_val_loss,
        "peak_entropy_step": peak_entropy_step,
        "peak_entropy": peak_entropy,
        "total_steps": global_step,
        "total_time_s": time.time() - t_start,
        "output_dir": output_dir,
    }

    summary_path = os.path.join(output_dir, "aesl_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("  AESL Experiment Complete")
    logger.info("=" * 60)
    logger.info(f"  Best Loss:      step={best_val_loss_step}, ppl={best_val_loss:.2f}")
    logger.info(f"  Peak Entropy:   step={peak_entropy_step}, entropy={peak_entropy:.4f}")
    logger.info(f"  Total steps:    {global_step}")
    logger.info(f"  Total time:     {(time.time()-t_start)/60:.1f} min")
    logger.info(f"  Checkpoints:    {len(metrics_log)}")
    logger.info(f"  Metrics:        {metrics_path}")
    logger.info(f"  Summary:        {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="AESL Diversity Early-Stopping Experiment")
    parser.add_argument("--train_data", type=str,
                        default="/root/autodl-tmp/catan-rl-llm/catan-rl-llm/data/ab_sft/main/train.jsonl")
    parser.add_argument("--val_data", type=str,
                        default="/root/autodl-tmp/catan-rl-llm/catan-rl-llm/data/ab_sft/main/val.jsonl")
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_loss_size", type=int, default=200)
    parser.add_argument("--val_entropy_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--output", type=str, default="checkpoints/aesl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_aesl(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        train_size=args.train_size,
        val_loss_size=args.val_loss_size,
        val_entropy_size=args.val_entropy_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        save_every=args.save_every,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
