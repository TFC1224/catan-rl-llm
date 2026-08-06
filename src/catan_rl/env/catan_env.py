"""
Environment wrapper for Catanatron Gym.

Provides a clean factory function for creating configured Catan environments
with various opponent bots, map types, and victory conditions.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import logging

from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer

logger = logging.getLogger(__name__)


# Map of opponent names to their constructor functions
# VictoryPointPlayer is the strongest built-in bot (focuses on VP maximization)
# WeightedRandomPlayer provides a random-but-weighted baseline
OPPONENT_REGISTRY = {
    "WeightedRandomPlayer": lambda color: WeightedRandomPlayer(color),
    "VictoryPointPlayer": lambda color: VictoryPointPlayer(color),
    # Aliases for compatibility with config naming
    "AlphaBetaPlayer": lambda color: VictoryPointPlayer(color),
    "ValueFunctionPlayer": lambda color: VictoryPointPlayer(color),
}


def make_catan_env(
    map_type: str = "MINI",
    vps_to_win: int = 6,
    opponents: Optional[List[Union[str, Any]]] = None,
    reward_function: Optional[Callable] = None,
    representation: str = "mixed",
    max_turns: int = 200,
    **kwargs,
) -> CatanatronEnv:
    """
    Create a configured Catanatron environment.

    Args:
        map_type: "MINI" (7 tiles) or "BASE" (19 tiles)
        vps_to_win: Victory points needed to win (6 for MINI, 10 for BASE)
        opponents: List of opponents. Each can be:
            - A string name from OPPONENT_REGISTRY (e.g., "WeightedRandomPlayer")
            - An already-constructed player object
            Default: [WeightedRandomPlayer("RED")]
        reward_function: Optional custom reward function callback
        representation: Observation representation mode ("mixed" default)
        max_turns: Maximum turns before game is truncated

    Returns:
        CatanatronEnv instance
    """
    if opponents is None:
        opponents = [WeightedRandomPlayer("RED")]

    # Resolve string opponent names to player instances
    resolved_opponents = []
    for i, opponent in enumerate(opponents):
        if isinstance(opponent, str):
            color = ["RED", "WHITE", "ORANGE"][i]
            if opponent in OPPONENT_REGISTRY:
                resolved_opponents.append(OPPONENT_REGISTRY[opponent](color))
            else:
                logger.warning(
                    f"Unknown opponent '{opponent}', using WeightedRandomPlayer"
                )
                resolved_opponents.append(WeightedRandomPlayer(color))
        else:
            resolved_opponents.append(opponent)

    config = {
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "enemies": resolved_opponents,
        "representation": representation,
    }

    if reward_function is not None:
        config["reward_function"] = reward_function

    config.update(kwargs)

    logger.info(
        f"Creating Catan env: map={map_type}, vps={vps_to_win}, "
        f"opponents={[type(o).__name__ for o in resolved_opponents]}"
    )

    env = CatanatronEnv(config=config)

    return env


def make_env_for_curriculum(
    curriculum_step: Dict[str, Any],
) -> CatanatronEnv:
    """
    Create an environment from a curriculum step dict.

    Args:
        curriculum_step: Dict with keys:
            - map_type: str
            - vps_to_win: int
            - opponents: list of str

    Returns:
        CatanatronEnv instance
    """
    return make_catan_env(
        map_type=curriculum_step.get("map_type", "MINI"),
        vps_to_win=curriculum_step.get("vps_to_win", 6),
        opponents=curriculum_step.get("opponents", ["WeightedRandomPlayer"]),
    )
