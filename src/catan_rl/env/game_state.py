"""
Game state serialization and deserialization utilities.

Catanatron Game/State objects need to be serialized for:
1. Storing in HuggingFace Datasets (JSON/pickle)
2. Cloning for parallel simulation (GRPO reward computation)
3. Saving and resuming games

This module provides:
- State serialization via pickle (for faithful reproduction)
- Game cloning via Game.copy() (for fast forking)
- State reconstruction from pickled bytes
"""

import pickle
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def serialize_game(game: Any) -> bytes:
    """
    Pickle a Catanatron Game object for storage.

    Args:
        game: A catanatron.game.Game instance

    Returns:
        bytes: Pickled game data
    """
    try:
        return pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Failed to serialize game: {e}")
        raise


def deserialize_game(data: bytes) -> Any:
    """
    Reconstruct a Game from pickled bytes.

    Args:
        data: Pickled game data

    Returns:
        catanatron.game.Game instance
    """
    try:
        return pickle.loads(data)
    except Exception as e:
        logger.error(f"Failed to deserialize game: {e}")
        raise


def clone_game(game: Any) -> Any:
    """
    Create a deep copy of a Game using its built-in copy() method.

    This is the preferred method for fast forking during simulation.

    Args:
        game: A catanatron.game.Game instance

    Returns:
        A deep copy of the game
    """
    if hasattr(game, 'copy'):
        return game.copy()
    else:
        # Fallback to pickle-based cloning
        return deserialize_game(serialize_game(game))


def serialize_state(state: Any) -> bytes:
    """
    Pickle a catanatron State object.

    Args:
        state: A catanatron.state.State instance

    Returns:
        bytes: Pickled state data
    """
    try:
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Failed to serialize state: {e}")
        raise


def deserialize_state(data: bytes) -> Any:
    """
    Reconstruct a State from pickled bytes.

    Args:
        data: Pickled state data

    Returns:
        catanatron.state.State instance
    """
    try:
        return pickle.loads(data)
    except Exception as e:
        logger.error(f"Failed to deserialize state: {e}")
        raise


def get_game_info(game: Any) -> dict:
    """
    Extract summary info from a Game for logging/debugging.

    Args:
        game: catanatron.game.Game instance

    Returns:
        Dict with game summary info
    """
    info = {
        "num_turns": 0,
        "is_done": False,
        "winner": None,
        "current_player": None,
        "players": [],
    }

    try:
        state = game.state
        info["num_turns"] = state.num_turns
        info["current_player"] = str(state.current_color)
        info["is_done"] = game.is_done() if hasattr(game, 'is_done') else False

        if info["is_done"] and hasattr(game, 'winner'):
            info["winner"] = str(game.winner)
    except Exception as e:
        logger.warning(f"Could not extract game info: {e}")

    return info
