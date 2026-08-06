"""
Tokenization and chat template preprocessing for Catan datasets.

Prepares data for both SFT (chat template format) and GRPO (prompt format).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def format_sft_example(
    system_prompt: str,
    observation_text: str,
    action_json: str,
    tokenizer: Any = None,
) -> Dict[str, str]:
    """
    Format a single (observation, action) pair as a chat example for SFT.

    Args:
        system_prompt: System prompt with game rules and strategy
        observation_text: Formatted game state observation
        action_json: The chosen action as a JSON string (e.g., '{"action_number": 3}')
        tokenizer: Optional tokenizer for applying chat template

    Returns:
        Dict with "text" key containing the full formatted string
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation_text},
        {"role": "assistant", "content": action_json},
    ]

    if tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")

    # Fallback: simple concatenation
    text = (
        f"System: {system_prompt}\n\n"
        f"User: {observation_text}\n\n"
        f"Assistant: {action_json}"
    )
    return {"text": text}


def format_grpo_prompt(
    system_prompt: str,
    observation_text: str,
    tokenizer: Any = None,
) -> str:
    """
    Format a game state as a prompt for GRPO (generation-only, no assistant response).

    Args:
        system_prompt: System prompt with game rules and strategy
        observation_text: Formatted game state observation
        tokenizer: Optional tokenizer for applying chat template

    Returns:
        str: Complete prompt string ready for tokenization
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation_text},
    ]

    if tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")

    # Fallback
    return f"System: {system_prompt}\n\nUser: {observation_text}\n\nAssistant:"
