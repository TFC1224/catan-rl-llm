"""
Fast parallel game simulation for GRPO reward computation.

The GRPO reward function needs to evaluate candidate actions by
simulating the game from the current state to completion. This module
provides single-threaded and parallel simulation functions.

Design:
- Clone the game at decision point via game.copy()
- Execute the candidate action via game.execute()
- Simulate to completion using game.play_tick() for opponents
  and a fast heuristic bot for the agent's subsequent turns
- Average over multiple rollouts for stochastic stability (dice)

Key Catanatron API (v3.2.1):
- Game.execute(action): execute a catan Action on the game
- Game.play_tick(): execute one tick (bot decides + acts)
- Game.copy(): deep-copy the game state
- Game.winning_color(): returns winning Color or None
- State.current_color(): returns current player's Color (method, not property)
- State.playable_actions: list of Action namedtuples
- Bot.decide(game, playable_actions): bot returns Action
"""

import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Optional

import numpy as np

from .game_state import clone_game
from .reward import terminal_reward

logger = logging.getLogger(__name__)


def simulate_from_state(
    serialized_game: bytes,
    action: Any,
    player_index: int = 0,
    num_rollouts: int = 3,
) -> float:
    """
    Clone the game, apply the candidate action, then simulate to completion.

    Because Catan involves dice rolls, we run multiple rollouts and average.

    Args:
        serialized_game: Pickled Game object at the decision point
        action: The candidate catan Action object to evaluate
        player_index: Agent's player index (0 = BLUE)
        num_rollouts: Number of simulations per candidate

    Returns:
        float: Average reward in [-1, 1]
    """
    try:
        game = pickle.loads(serialized_game)
    except Exception as e:
        logger.error(f"Failed to deserialize game: {e}")
        return 0.0

    # Agent's color
    agent_color_map = {0: "BLUE", 1: "RED", 2: "WHITE", 3: "ORANGE"}
    agent_color_str = agent_color_map.get(player_index, "BLUE")

    # Use VictoryPointPlayer for agent's follow-up turns during simulation.
    # WeightedRandomPlayer is too weak (loses ~90% of games), making all actions
    # look equally bad. VictoryPointPlayer can capitalize on good positions,
    # creating a meaningful reward signal: good actions → strong position → wins.
    from catanatron.players.search import VictoryPointPlayer
    from catanatron.models.player import Color

    color_enum_map = {
        0: Color.BLUE, 1: Color.RED, 2: Color.WHITE, 3: Color.ORANGE
    }
    agent_color = color_enum_map.get(player_index, Color.BLUE)
    fast_bot = VictoryPointPlayer(agent_color)

    outcomes = []
    for rollout in range(num_rollouts):
        try:
            game_clone = clone_game(game)
            max_ticks = 500  # Safety limit

            # Execute the candidate action (P0's chosen action)
            game_clone.execute(action)

            # Simulate to completion
            for tick in range(max_ticks):
                winner = game_clone.winning_color()
                if winner is not None:
                    break

                state = game_clone.state
                current_color = str(state.current_color())

                if agent_color_str in current_color.upper():
                    # Agent's turn — use fast heuristic bot
                    playable = list(state.playable_actions)
                    if not playable:
                        break
                    bot_action = fast_bot.decide(game_clone, playable)
                    if bot_action is not None:
                        game_clone.execute(bot_action)
                else:
                    # Opponent's turn — play_tick handles it
                    game_clone.play_tick()

            # Determine outcome
            winner = game_clone.winning_color()
            if winner is not None and str(winner).upper() == agent_color_str:
                outcomes.append(1.0)
            elif winner is not None:
                outcomes.append(-1.0)
            else:
                outcomes.append(0.0)  # Draw or timeout

        except Exception as e:
            logger.debug(f"Simulation rollout {rollout} failed: {e}")
            outcomes.append(0.0)

    return float(np.mean(outcomes)) if outcomes else 0.0


def batch_simulate(
    states: List[bytes],
    candidate_actions: List[List[Any]],  # (num_states, K) list of candidate action lists
    player_index: int = 0,
    num_rollouts: int = 3,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Parallel simulation for all candidates across all states.

    This is the workhorse for GRPO reward computation — it evaluates
    K candidate actions for each of B game states in parallel.

    Args:
        states: List of B serialized game states
        candidate_actions: List of B lists, each containing K candidate catan Action objects
        player_index: Agent's player index
        num_rollouts: Number of rollouts per candidate
        num_workers: Number of parallel workers

    Returns:
        np.ndarray of shape (B, K) with average rewards
    """
    B = len(states)
    if B == 0:
        return np.zeros((0, 0))
    K = len(candidate_actions[0]) if candidate_actions else 0
    rewards = np.zeros((B, K))

    if K == 0:
        return rewards

    # Build all tasks
    tasks = []
    for i in range(B):
        for j in range(K):
            tasks.append({
                "serialized_game": states[i],
                "action": candidate_actions[i][j],
                "player_index": player_index,
                "num_rollouts": num_rollouts,
            })

    # Run in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx, task in enumerate(tasks):
            future = executor.submit(simulate_from_state, **task)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            i = idx // K
            j = idx % K
            try:
                rewards[i, j] = future.result()
            except Exception as e:
                logger.warning(f"Simulation failed for ({i},{j}): {e}")
                rewards[i, j] = 0.0

    return rewards
