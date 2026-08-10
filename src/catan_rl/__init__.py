"""
Catan RL + LLM: Train Qwen3-8B to play Settlers of Catan.

This package implements a full pipeline for training language models
to play the Settlers of Catan board game using:
- LlamaGym Agent pattern for the agent-environment interface
- GRPO (Group Relative Policy Optimization) from TRL for RL training
- QLoRA for memory-efficient fine-tuning on 24GB VRAM GPUs
"""

__version__ = "0.1.0"
