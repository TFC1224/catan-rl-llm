"""
GRPO dataset construction from game rollouts.

Converts rollout records into a HuggingFace Dataset format compatible
with TRL's GRPOTrainer.

Dataset columns:
- "prompt": formatted observation text (system + game state)
- "serialized_game": base64-encoded pickled Game (for simulation reward)
- "valid_actions": JSON string of valid action descriptions
- "int_actions": JSON string of action space indices (for mapping)
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


def build_grpo_dataset(
    rollout_records: List[Dict[str, Any]],
) -> Dataset:
    """
    Build a GRPO-compatible dataset from rollout records.

    Each row in the dataset corresponds to one game state (decision point).
    The GRPOTrainer uses "prompt" for generation and passes other columns
    to the reward function.

    Args:
        rollout_records: List of rollout record dicts from play_game_with_agent()

    Returns:
        HuggingFace Dataset with columns:
        - "prompt": system + observation text
        - "serialized_game": base64-encoded bytes (pickled Game)
        - "valid_actions": JSON string of valid action descriptions
        - "int_actions": JSON string of action space indices
    """
    if not rollout_records:
        logger.warning("No rollout records provided")
        return Dataset.from_dict({})

    data = {
        "prompt": [],
        "serialized_game": [],
        "valid_actions": [],
        "int_actions": [],
    }

    for record in rollout_records:
        data["prompt"].append(record.get("prompt", ""))
        data["valid_actions"].append(record.get("valid_actions", "[]"))
        data["int_actions"].append(record.get("int_actions", "[]"))

        # Encode bytes as base64 for JSON/parquet compatibility
        game_bytes = record.get("serialized_game", b"")
        if isinstance(game_bytes, bytes):
            game_bytes = base64.b64encode(game_bytes).decode("utf-8")
        data["serialized_game"].append(game_bytes)

    dataset = Dataset.from_dict(data)

    logger.info(f"Built GRPO dataset: {len(dataset)} examples")
    return dataset


def load_rollout_data(data_dir: str) -> List[Dict]:
    """
    Load rollout records from a JSONL file.

    Args:
        data_dir: Directory containing rollout.jsonl

    Returns:
        List of record dicts
    """
    filepath = os.path.join(data_dir, "rollout.jsonl")
    if not os.path.exists(filepath):
        logger.error(f"Rollout file not found: {filepath}")
        return []

    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    # Decode base64 game bytes if present
                    if "serialized_game" in record and isinstance(record["serialized_game"], str):
                        record["serialized_game"] = base64.b64decode(
                            record["serialized_game"]
                        )
                    records.append(record)
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(records)} rollout records from {filepath}")
    return records


def save_rollout_data(records: List[Dict], output_dir: str):
    """
    Save rollout records to a JSONL file.

    Args:
        records: List of record dicts
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "rollout.jsonl")

    with open(filepath, "w") as f:
        for record in records:
            # Convert bytes to base64 for JSON serialization
            serialized = dict(record)  # shallow copy
            game_bytes = serialized.get("serialized_game")
            if isinstance(game_bytes, bytes):
                serialized["serialized_game"] = base64.b64encode(
                    game_bytes
                ).decode("utf-8")

            f.write(json.dumps(serialized) + "\n")

    logger.info(f"Saved {len(records)} records to {filepath}")
