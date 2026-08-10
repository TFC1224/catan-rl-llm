"""
Head-to-head tournament evaluation for Catan agents.

Runs multiple games of the trained agent against various opponent bots
and collects win rate statistics and detailed game metrics.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm

from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer

from ..env.catan_env import make_catan_env

logger = logging.getLogger(__name__)


def run_tournament(
    agent: Any,
    opponents: List[str],
    num_games: int = 100,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    player_index: int = 0,
    control_turn_order: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a tournament between the agent and specified opponents.

    Args:
        agent: A CatanAgent instance
        opponents: List of opponent names (e.g., ["WeightedRandomPlayer"])
        num_games: Number of games to play
        map_type: "MINI" or "BASE"
        vps_to_win: Victory points to win
        player_index: Agent's player index (0 = BLUE, first player)
        control_turn_order: If True, also run games with agent as RED (turn-order control)
        seed: Random seed for reproducibility

    Returns:
        Dict with:
        - "win_rate": float
        - "loss_rate": float
        - "draw_rate": float
        - "avg_vp_margin": float
        - "avg_turns": float
        - "action_validity_rate": float
        - "games": list of individual game results
    """
    logger.info(f"Tournament: {num_games} games on {map_type} map ({vps_to_win}VP)")
    logger.info(f"Opponents: {opponents}")

    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_vp_margin": 0,
        "total_turns": 0,
        "total_valid_actions": 0,
        "total_actions": 0,
        "games": [],
    }

    total_games = num_games
    if control_turn_order:
        total_games = num_games * 2  # Half as first player, half as second

    pbar = tqdm.tqdm(total=total_games, desc="Tournament")

    # Run half the games with agent as BLUE (first player)
    games_as_first = num_games // 2 if control_turn_order else num_games
    for i in range(games_as_first):
        try:
            game_result = _play_eval_game(
                agent=agent,
                opponents=opponents,
                map_type=map_type,
                vps_to_win=vps_to_win,
                player_index=0,  # BLUE = first
                seed=seed + i,
            )
            results = _accumulate_result(results, game_result)
        except Exception as e:
            logger.warning(f"Game failed: {e}")
        pbar.update(1)

    # Run remaining games with agent as RED (second player) for balance
    if control_turn_order:
        games_as_second = num_games - games_as_first
        for i in range(games_as_second):
            try:
                game_result = _play_eval_game(
                    agent=agent,
                    opponents=opponents,
                    map_type=map_type,
                    vps_to_win=vps_to_win,
                    player_index=1,  # RED = second
                    seed=seed + games_as_first + i,
                )
                results = _accumulate_result(results, game_result)
            except Exception as e:
                logger.warning(f"Game failed: {e}")
            pbar.update(1)

    pbar.close()

    total = results["wins"] + results["losses"] + results["draws"]
    if total == 0:
        total = 1

    return {
        "win_rate": results["wins"] / total,
        "loss_rate": results["losses"] / total,
        "draw_rate": results["draws"] / total,
        "avg_vp_margin": results["total_vp_margin"] / total,
        "avg_turns": results["total_turns"] / total,
        "action_validity_rate": (
            results["total_valid_actions"] / max(results["total_actions"], 1)
        ),
        "games": results["games"],
        "opponent": opponents[0] if opponents else "Unknown",
        "map_type": map_type,
        "vps_to_win": vps_to_win,
    }


def _play_eval_game(
    agent: Any,
    opponents: List[str],
    map_type: str,
    vps_to_win: int,
    player_index: int,
    seed: int,
) -> Dict[str, Any]:
    """Play a single evaluation game."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    env = make_catan_env(
        map_type=map_type,
        vps_to_win=vps_to_win,
        opponents=opponents,
    )

    obs = env.reset()
    done = False
    turn = 0
    num_valid = 0
    num_total = 0

    while not done and turn < 200:
        state = env.game.state
        int_actions = env.get_valid_actions()
        playable = state.playable_actions

        if not int_actions:
            break

        # Agent's turn — CatanatronEnv always presents P0 decisions after auto-advancing
        agent_action = agent.act(
            observation=state,
            valid_actions=playable,
            player_index=player_index,
        )

        num_total += 1
        if agent_action.is_valid:
            num_valid += 1

        action_idx = int_actions[min(agent_action.action_index, len(int_actions) - 1)] if int_actions else 0
        step_result = env.step(action_idx)
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated

        turn += 1

    # Determine outcome
    outcome = _get_outcome(env, player_index)
    agent_vp = env.game.state.player_state.get(f"P{player_index}_ACTUAL_VICTORY_POINTS", 0)
    opponent_vp = env.game.state.player_state.get("P1_ACTUAL_VICTORY_POINTS", 0)

    env.close()

    return {
        "outcome": outcome,
        "agent_vp": agent_vp,
        "opponent_vp": opponent_vp,
        "vp_margin": agent_vp - opponent_vp,
        "turns": turn,
        "valid_actions": num_valid,
        "total_actions": num_total,
    }


def _get_outcome(env: Any, player_index: int) -> str:
    """Get game outcome."""
    try:
        if not env.game.is_done():
            return "TIMEOUT"
        winner = getattr(env.game, 'winner', None)
        if winner is None:
            return "DRAW"
        color_map = {0: "BLUE", 1: "RED", 2: "WHITE", 3: "ORANGE"}
        if str(winner).upper() == color_map.get(player_index, ""):
            return "WIN"
        return "LOSS"
    except Exception:
        return "UNKNOWN"


def _accumulate_result(
    results: Dict,
    game_result: Dict,
) -> Dict:
    """Accumulate a single game result into the tournament results."""
    outcome = game_result.get("outcome", "UNKNOWN")
    if outcome == "WIN":
        results["wins"] += 1
    elif outcome == "LOSS":
        results["losses"] += 1
    else:
        results["draws"] += 1

    results["total_vp_margin"] += game_result.get("vp_margin", 0)
    results["total_turns"] += game_result.get("turns", 0)
    results["total_valid_actions"] += game_result.get("valid_actions", 0)
    results["total_actions"] += game_result.get("total_actions", 0)
    results["games"].append(game_result)

    return results
