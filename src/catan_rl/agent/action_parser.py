"""
Robust action parser for Catanatron game actions.

Parses LLM text outputs into validated game actions. Uses multiple
fallback strategies for robustness against model output variability.

Parsing strategies (tried in order):
1. JSON: {"action_number": <N>}  — preferred format
2. JSON: {"action": "<TYPE>", "params": {...}}  — fallback JSON
3. Regex: match action type name in text
4. Fuzzy: find closest valid action by string similarity
5. Random: pick a random valid action (last resort)
"""

import json
import re
import random
import logging
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

from .base import AgentAction

logger = logging.getLogger(__name__)


def parse_action(
    response: str,
    valid_actions: List[Any],
    action_descriptions: Optional[List[str]] = None,
) -> AgentAction:
    """
    Parse model output into a validated AgentAction.

    Args:
        response: Raw text output from the LLM
        valid_actions: List of catanatron Action objects
        action_descriptions: Optional pre-computed descriptions for fuzzy matching.
                             If None, they are generated from valid_actions.

    Returns:
        AgentAction with is_valid=True if parsing succeeded, False otherwise
    """
    if not valid_actions:
        return AgentAction(
            action_index=-1,
            action_type="NO_ACTIONS",
            raw_text=response,
            is_valid=False,
        )

    # Pre-compute descriptions for fuzzy matching
    if action_descriptions is None:
        action_descriptions = [str(a) for a in valid_actions]

    # —— Strategy 1: JSON with action_number ——
    result = _try_json_number(response, len(valid_actions))
    if result is not None:
        action_index = result
        return AgentAction(
            action_index=action_index,
            action_type=str(valid_actions[action_index]),
            raw_text=response,
            is_valid=True,
        )

    # —— Strategy 2: JSON with action type + params ——
    result = _try_json_action_type(response, valid_actions)
    if result is not None:
        action_index, action_type, params = result
        return AgentAction(
            action_index=action_index,
            action_type=action_type,
            params=params,
            raw_text=response,
            is_valid=True,
        )

    # —— Strategy 3: Regex match for action type ——
    result = _try_regex_match(response, valid_actions, action_descriptions)
    if result is not None:
        action_index = result
        return AgentAction(
            action_index=action_index,
            action_type=str(valid_actions[action_index]),
            raw_text=response,
            is_valid=True,
        )

    # —— Strategy 4: Fuzzy string matching ——
    result = _try_fuzzy_match(response, action_descriptions)
    if result is not None:
        action_index = result
        return AgentAction(
            action_index=action_index,
            action_type=str(valid_actions[action_index]),
            raw_text=response,
            is_valid=True,
        )

    # —— Strategy 5: Random fallback ——
    logger.warning(
        f"All parsing strategies failed for response: {response[:200]}... "
        f"Selecting random valid action."
    )
    action_index = random.randint(0, len(valid_actions) - 1)
    return AgentAction(
        action_index=action_index,
        action_type=str(valid_actions[action_index]),
        raw_text=response,
        is_valid=False,  # Mark as invalid since parsing failed
    )


def _try_json_number(response: str, num_actions: int) -> Optional[int]:
    """
    Try to parse {"action_number": <N>} from the response.

    Returns action_index if successful, None otherwise.
    """
    # Try to find a JSON object in the response
    json_pattern = r'\{[^}]+\}'
    matches = re.findall(json_pattern, response)

    for match in matches:
        try:
            data = json.loads(match)
            if "action_number" in data:
                idx = int(data["action_number"])
                if 0 <= idx < num_actions:
                    return idx
                # Out of range: try modulo mapping (the model learned sequential
                # indices from training data with variable action counts)
                elif idx >= num_actions and num_actions > 0:
                    mapped = idx % num_actions
                    logger.info(
                        f"action_number {idx} out of range [0, {num_actions}), "
                        f"mapped to {mapped} via modulo"
                    )
                    return mapped
        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    return None


def _try_json_action_type(
    response: str,
    valid_actions: List[Any],
) -> Optional[Tuple[int, str, Dict]]:
    """
    Try to parse {"action": "<TYPE>", "params": {...}} from the response.

    Returns (action_index, action_type, params) if successful, None otherwise.
    """
    json_pattern = r'\{[^}]+\}'
    matches = re.findall(json_pattern, response)

    for match in matches:
        try:
            data = json.loads(match)
            if "action" in data:
                action_name = str(data["action"]).upper().replace(" ", "_")
                params = data.get("params", {})

                # Find matching action
                for i, action in enumerate(valid_actions):
                    action_str = str(action).upper()
                    if action_name in action_str:
                        return (i, action_name, params)

        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _try_regex_match(
    response: str,
    valid_actions: List[Any],
    descriptions: List[str],
) -> Optional[int]:
    """
    Try to match action keywords using regex patterns.

    Common patterns:
    - "BUILD_SETTLEMENT <node_id>"
    - "BUILD_ROAD <edge>"
    - "END_TURN"
    - integer at start of line

    Returns action_index if successful, None otherwise.
    """
    # Pattern 1: Action type names
    action_types = [
        "BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD",
        "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT", "PLAY_MONOPOLY",
        "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING",
        "END_TURN", "ROLL_DICE", "MOVE_ROBBER",
        "MARITIME_TRADE", "DISCARD",
    ]

    response_upper = response.upper()

    for atype in action_types:
        if atype in response_upper:
            # Find first valid action matching this type
            for i, desc in enumerate(descriptions):
                if atype in desc.upper():
                    return i

    # Pattern 2: Integer at the start of a line (e.g., "3" or "3." or "Action 3")
    int_patterns = [
        r'^\s*(\d+)\s*$',          # Just a number on its own line
        r'^\s*(\d+)[\.\)]\s*',      # Number followed by . or )
        r'action\s+(\d+)',          # "action 5" or "Action 5"
        r'choose\s+(\d+)',          # "choose 3"
        r'number\s+(\d+)',          # "number 7"
    ]

    for pattern in int_patterns:
        match = re.search(pattern, response_upper, re.MULTILINE)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(valid_actions):
                return idx

    return None


def _try_fuzzy_match(
    response: str,
    descriptions: List[str],
    threshold: float = 0.5,
) -> Optional[int]:
    """
    Use fuzzy string matching to find the closest valid action.

    Returns action_index if a match exceeds threshold, None otherwise.
    """
    best_score = 0.0
    best_idx = None

    for i, desc in enumerate(descriptions):
        score = SequenceMatcher(None, response.lower(), desc.lower()).ratio()
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= threshold and best_idx is not None:
        return best_idx

    return None


def validate_action(
    action_index: int,
    valid_actions: List[Any],
) -> bool:
    """
    Check if an action index is within the valid range.

    Args:
        action_index: The index to validate
        valid_actions: List of valid Action objects

    Returns:
        True if action_index is valid
    """
    return 0 <= action_index < len(valid_actions)


def action_to_dict(agent_action: AgentAction) -> Dict[str, Any]:
    """
    Convert an AgentAction to a serializable dict.

    Args:
        agent_action: The AgentAction to convert

    Returns:
        Dict with action_index, action_type, params, is_valid, raw_text
    """
    return {
        "action_index": agent_action.action_index,
        "action_type": agent_action.action_type,
        "params": agent_action.params,
        "is_valid": agent_action.is_valid,
        "raw_text": agent_action.raw_text[:200],  # Truncate for storage
    }
