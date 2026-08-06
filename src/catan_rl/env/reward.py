"""
Reward functions for Catanatron game outcomes.

Provides both terminal rewards (win/loss/draw) and dense incremental
rewards (VP changes, resource efficiency) for training.

The primary GRPO reward is terminal (simulated game outcome).
Dense rewards are available as auxiliary signals during early training.
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def terminal_reward(game_outcome: str) -> float:
    """
    Terminal reward based on game outcome.

    Args:
        game_outcome: "WIN", "LOSS", or "DRAW"

    Returns:
        float: +1.0 for win, -1.0 for loss, 0.0 for draw
    """
    rewards = {
        "WIN": 1.0,
        "LOSS": -1.0,
        "DRAW": 0.0,
    }
    return rewards.get(game_outcome.upper(), 0.0)


def victory_point_reward(
    current_vp: int,
    prev_vp: int,
    vps_to_win: int = 10,
) -> float:
    """
    Incremental reward for VP changes.

    Normalized to [-0.1, 0.1] range per VP to avoid dominating
    the terminal reward signal.

    Args:
        current_vp: Current victory points
        prev_vp: Previous turn's victory points
        vps_to_win: Total VPs needed to win

    Returns:
        float: Normalized VP delta reward
    """
    delta = current_vp - prev_vp
    return 0.05 * delta  # Small reward per VP


def resource_efficiency_reward(
    resources_spent: int,
    vp_gained: float,
) -> float:
    """
    Reward for efficient resource-to-VP conversion.

    Args:
        resources_spent: Total resources spent this turn
        vp_gained: VPs gained this turn

    Returns:
        float: Efficiency ratio normalized
    """
    if resources_spent == 0:
        return 0.0
    efficiency = vp_gained / resources_spent
    return min(0.1, efficiency * 0.02)  # Cap at 0.1


def validity_reward(is_valid: bool) -> float:
    """
    Penalty for invalid/illegal actions.

    Strong negative reward to discourage the model from outputting
    actions that don't match any valid game action.

    Args:
        is_valid: Whether the parsed action is valid

    Returns:
        float: 0.0 if valid, -0.5 if invalid
    """
    return 0.0 if is_valid else -0.5


def composite_reward(
    is_valid: bool,
    game_outcome: str,
    vp_delta: float = 0.0,
    weights: Optional[dict] = None,
) -> float:
    """
    Weighted composite reward combining multiple signals.

    Default weights:
    - validity: 0.1
    - terminal: 0.7
    - vp_delta: 0.2

    Args:
        is_valid: Whether action is valid
        game_outcome: "WIN", "LOSS", "DRAW", or None (if game not over)
        vp_delta: VP change this turn
        weights: Optional custom weight dict

    Returns:
        float: Weighted composite reward
    """
    if weights is None:
        weights = {"validity": 0.1, "terminal": 0.7, "vp_delta": 0.2}

    reward = 0.0

    # Validity component
    reward += weights["validity"] * validity_reward(is_valid)

    # Terminal component
    if game_outcome is not None:
        reward += weights["terminal"] * terminal_reward(game_outcome)

    # VP delta component
    reward += weights["vp_delta"] * min(vp_delta * 0.05, 0.1)

    return reward


def determine_game_outcome(
    game: Any,
    agent_player_index: int = 0,
) -> str:
    """
    Determine the game outcome for the agent.

    Args:
        game: catanatron.game.Game instance
        agent_player_index: Index of the agent's player

    Returns:
        str: "WIN", "LOSS", or "DRAW"
    """
    try:
        if not game.is_done():
            return None  # Game not finished

        winner = getattr(game, 'winner', None)
        if winner is None:
            return "DRAW"

        winner_color = str(winner)
        agent_color = ["BLUE", "RED", "WHITE", "ORANGE"][agent_player_index]

        if winner_color.upper() == agent_color:
            return "WIN"
        else:
            return "LOSS"
    except Exception as e:
        logger.warning(f"Failed to determine game outcome: {e}")
        return None
