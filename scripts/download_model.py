#!/usr/bin/env python3
"""
Download Qwen3-8B-Instruct model from HuggingFace.

This script downloads the base model and tokenizer to the local HF cache.
Uses the HF_TOKEN environment variable if set, otherwise downloads publicly.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen3-8B-Instruct
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download Qwen model from HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B-Instruct",
        help="Model name on HuggingFace Hub (default: Qwen/Qwen3-8B-Instruct)",
    )
    args = parser.parse_args()

    model_name = args.model
    print(f"Downloading model: {model_name}")
    print("=" * 60)

    # Check HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print("HF_TOKEN found. Using authenticated access.")
    else:
        print("No HF_TOKEN set. Using public access (may require model to be public).")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n[1/2] Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        print(f"  Chat template: {tokenizer.chat_template is not None}")

        print(f"\n[2/2] Downloading model {model_name}...")
        print("  This may take 10-30 minutes depending on network speed.")
        print("  Model size: ~16GB (bf16) or ~6GB (4-bit)")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            torch_dtype="auto",
            device_map="auto",
        )

        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Model loaded successfully. Parameters: {param_count:.2f}B")

        print("\n" + "=" * 60)
        print("Model download complete!")
        print(f"Model: {model_name}")
        print(f"Cache location: ~/.cache/huggingface/hub/")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print("  1. Check network connection", file=sys.stderr)
        print("  2. Set HF_TOKEN environment variable for gated models", file=sys.stderr)
        print("  3. Try a smaller model: python scripts/download_model.py --model Qwen/Qwen3-0.6B", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
