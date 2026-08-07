"""
GRPO (Group Relative Policy Optimization) training for Catan agent.

Uses TRL's GRPOTrainer with a custom reward function that evaluates
candidate actions by simulating game outcomes.

Key design:
- Each game state is a "prompt" for GRPO
- Model generates K=4 candidate actions per state
- Reward function: parse action → validate → simulate game to end → score
- Group-relative advantage: compare K candidates for the same state
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

from .utils import load_model_and_tokenizer, setup_lora, load_lora_checkpoint
from ..agent.action_parser import parse_action
from ..env.simulator import simulate_from_state

logger = logging.getLogger(__name__)


def catan_grpo_reward(
    prompts: List[str],
    completions: List[str],
    serialized_game: List[bytes],
    valid_actions: List[str],
    int_actions: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Custom reward function for GRPO training on Catan.

    TRL's GRPOTrainer calls the reward function with:
    - prompts: List of B prompt strings
    - completions: List of B completion strings (one per prompt)
    - Plus dataset columns (serialized_game, valid_actions, int_actions)

    For each (prompt, completion) pair:
    1. Parse the action from the completion text
    2. Validate against the valid_actions for that state
    3. If valid: deserialize game, map to catan Action, simulate to completion
    4. If invalid: apply penalty (-0.5)
    5. Return list of B rewards

    Args:
        prompts: List of B prompt strings
        completions: List of B completion strings
        serialized_game: List of B pickled game states (bytes)
        valid_actions: List of B JSON strings of valid action descriptions
        int_actions: Optional list of B JSON strings of action space indices
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of B float rewards in [-1, 1]
    """
    from catanatron_gym.envs.catanatron_env import from_action_space

    # Filter out None completions (can happen with truncation)
    B = len(prompts)
    rewards = []
    has_int_actions = int_actions is not None and len(int_actions) > 0

    import re

    for i in range(B):
        prompt = prompts[i]
        completion_text = completions[i] if i < len(completions) else ""

        # Handle None or empty completions
        if completion_text is None or not isinstance(completion_text, str):
            rewards.append(-1.0)
            continue

        completion_text = completion_text.strip()

        # HALLUCINATION GUARD: long responses (>200 chars) without JSON are
        # almost certainly Chinese hallucinations. Valid think-tag responses
        # with JSON are typically 40-80 chars including the think tags.
        if len(completion_text) > 200:
            if not re.search(r'\{[^}]+\}', completion_text):
                rewards.append(-1.0)
                continue

        # Parse valid action descriptions for this state
        try:
            v_actions_str = valid_actions[i] if i < len(valid_actions) else "[]"
            v_actions = json.loads(v_actions_str)
        except (json.JSONDecodeError, TypeError):
            v_actions = []

        # Parse int actions if available
        int_acts = None
        if has_int_actions and i < len(int_actions):
            try:
                int_acts = json.loads(int_actions[i])
            except (json.JSONDecodeError, TypeError):
                pass

        # Deserialize game
        try:
            game_bytes = serialized_game[i] if i < len(serialized_game) else b""
            game = pickle.loads(game_bytes)
            playable = list(game.state.playable_actions)
        except Exception:
            game = None
            playable = []

        # Parse action from model output
        agent_action = parse_action(completion_text, playable)

        if not agent_action.is_valid:
            # Strong penalty for invalid actions to discourage hallucination
            rewards.append(-1.0)
            continue

        # Try simulation-based reward
        seq_idx = agent_action.action_index
        if game is None or not playable or seq_idx < 0 or seq_idx >= len(playable):
            rewards.append(-0.5)
            continue

        # Get the catan Action object
        if int_acts is not None and seq_idx < len(int_acts):
            action_idx = int_acts[seq_idx]
            try:
                catan_action = from_action_space(action_idx, playable)
            except Exception:
                rewards.append(0.0)
                continue
        else:
            catan_action = playable[seq_idx]

        # Simulate from this state with this action
        try:
            sim_reward = simulate_from_state(
                serialized_game=game_bytes,
                action=catan_action,
                player_index=0,
                num_rollouts=5,
            )
            rewards.append(sim_reward)
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            rewards.append(0.0)

    return rewards


def train_grpo(
    model_name: str = "/root/autodl-tmp/Qwen/Qwen3-8B/",
    lora_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    dataset: Optional[Dataset] = None,
    output_dir: str = "./checkpoints/grpo/",
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """
    Run GRPO training for the Catan agent.

    Args:
        model_name: Base model name (for loading tokenizer if lora_path provided)
        lora_path: Path to SFT LoRA checkpoint (starting point for RL)
        dataset_path: Path to rollout dataset directory
        dataset: Pre-loaded Dataset (alternative to dataset_path)
        output_dir: Directory to save checkpoints
        config: Optional config dict
        **kwargs: Additional training overrides

    Returns:
        Path to the saved GRPO checkpoint
    """
    if config is None:
        config = {}

    grpo_config = config.get("grpo", {}).get("training", {})
    grpo_config.update(kwargs)

    logger.info("=" * 60)
    logger.info("  Starting GRPO Training")
    logger.info("=" * 60)

    # Load model
    logger.info("[1/4] Loading model...")
    if lora_path:
        # Load base model, then apply LoRA adapter
        logger.info(f"  Loading base model: {model_name}")
        base_model, tokenizer = load_model_and_tokenizer(model_name, load_in_4bit=True)
        model = load_lora_checkpoint(base_model, lora_path)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, load_in_4bit=True)
        model = setup_lora(model)

    # Load data
    logger.info("[2/4] Loading rollout data...")
    if dataset is not None:
        train_dataset = dataset
    elif dataset_path:
        from datasets import load_dataset
        train_dataset = load_dataset(
            "json",
            data_files=os.path.join(dataset_path, "rollout.jsonl"),
            split="train",
        )
        # Handle bytes columns — they get loaded as strings from JSON
        import base64
        def decode_bytes(example):
            if "serialized_game" in example and isinstance(example["serialized_game"], str):
                example["serialized_game"] = base64.b64decode(example["serialized_game"])
            return example
        train_dataset = train_dataset.map(decode_bytes)
    else:
        raise ValueError("Either dataset or dataset_path must be provided")

    logger.info(f"  Dataset: {len(train_dataset)} examples")

    # Configure GRPO
    logger.info("[3/4] Configuring GRPO trainer...")

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=grpo_config.get("num_train_epochs", 3),
        per_device_train_batch_size=grpo_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=grpo_config.get("gradient_accumulation_steps", 4),
        learning_rate=grpo_config.get("learning_rate", 5.0e-5),
        lr_scheduler_type=grpo_config.get("lr_scheduler_type", "cosine"),
        warmup_steps=grpo_config.get("warmup_steps", 0),
        logging_steps=grpo_config.get("logging_steps", 10),
        save_steps=grpo_config.get("save_steps", 200),
        num_generations=grpo_config.get("num_generations", 4),
        beta=grpo_config.get("beta", 0.10),
        temperature=grpo_config.get("temperature", 0.9),
        max_completion_length=grpo_config.get("max_completion_length", 128),
        bf16=True,
        report_to=grpo_config.get("report_to", "none"),
        run_name=grpo_config.get("run_name", "catan-grpo"),
        # Note: Qwen3 generates <think> tags due to chat template — this is OK,
        # the action parser handles stripping them. No need to suppress them.
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[catan_grpo_reward],
        processing_class=tokenizer,
    )

    # Train
    logger.info("[4/4] Training...")
    train_result = trainer.train()

    # Save
    logger.info(f"Saving GRPO checkpoint to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"GRPO training complete. Checkpoint: {output_dir}")
    return output_dir
