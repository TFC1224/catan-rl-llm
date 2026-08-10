"""
Training utilities: model loading, LoRA setup, configuration helpers.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str = "/root/autodl-tmp/Qwen/Qwen3-8B/",
    load_in_4bit: bool = True,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model ID
        load_in_4bit: Use 4-bit quantization (recommended for 24GB GPUs)
        device_map: Device mapping strategy
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  Set pad_token = eos_token")

    logger.info(f"Loading model: {model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
    }

    if load_in_4bit:
        logger.info("  Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model_kwargs.update(kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"  Model loaded: {param_count:.2f}B parameters")

    return model, tokenizer


def setup_lora(
    model: PreTrainedModel,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> PeftModel:
    """
    Apply LoRA adapters to a model.

    Args:
        model: Base model
        r: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to.
                       Defaults to all linear layers for Qwen models.

    Returns:
        PeftModel with LoRA adapters
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model for k-bit training if quantized
    if model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model


def load_lora_checkpoint(
    base_model: PreTrainedModel,
    lora_path: str,
) -> PeftModel:
    """
    Load a LoRA adapter from a checkpoint.

    Args:
        base_model: The base model (already loaded)
        lora_path: Path to the LoRA adapter checkpoint

    Returns:
        PeftModel with loaded adapters
    """
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model


def get_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract generation parameters from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Dict of generation parameters
    """
    gen_config = config.get("generation", {})
    return {
        "max_new_tokens": gen_config.get("max_new_tokens", 128),
        "temperature": gen_config.get("temperature", 0.8),
        "top_p": gen_config.get("top_p", 0.9),
        "top_k": gen_config.get("top_k", 50),
        "do_sample": gen_config.get("do_sample", True),
    }
