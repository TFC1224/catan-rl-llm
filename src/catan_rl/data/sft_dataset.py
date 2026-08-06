"""
SFT dataset generation from expert bot gameplay.

Plays games with a strong bot (VictoryPointPlayer) against weaker opponents
and records (observation, action) pairs for supervised fine-tuning.

The generated dataset teaches the model to:
1. Output valid JSON action format
2. Follow basic Catan strategy (from expert demonstrations)
3. Handle all game phases (initial placement, mid-game, end-game)
"""

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm

from catanatron.models.player import Color
from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer

logger = logging.getLogger(__name__)


def generate_sft_data_from_bot(
    bot_class: Any,
    num_games: int = 500,
    map_type: str = "MINI",
    vps_to_win: int = 6,
    output_dir: Optional[str] = None,
    train_split: float = 0.9,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Play games with a bot as "expert" and record (state, action) pairs.

    The expert bot plays as BLUE (player 0) against WeightedRandom opponents.
    Every action the expert takes is recorded with the game state at that moment.

    Args:
        bot_class: Bot constructor (e.g., VictoryPointPlayer)
        num_games: Number of games to play
        map_type: "MINI" or "BASE"
        vps_to_win: Victory points to win
        output_dir: Optional directory to save intermediate data
        train_split: Fraction of data for training (rest for validation)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_records, val_records) — each is a list of dicts with
        keys: "observation", "action", "game_phase", "turn_number", "player_index"
    """
    random.seed(seed)
    np.random.seed(seed)

    all_records = []
    games_completed = 0
    total_turns = 0

    logger.info(f"Generating SFT data: {num_games} games, {map_type} map, {vps_to_win}VP")

    pbar = tqdm.tqdm(total=num_games, desc="Generating SFT data")

    while games_completed < num_games:
        env = None
        try:
            # Create the expert bot (BLUE = player 0).
            # Must use Color enum (not string) because state.color_to_index uses Color.BLUE as key.
            # The CatanatronEnv auto-plays enemy bot turns via _advance_until_p0_decision(),
            # so we only need to handle the expert's own turns.
            expert = bot_class(Color.BLUE)
            enemy_red = WeightedRandomPlayer(Color.RED)
            enemies = [enemy_red]

            env = CatanatronEnv(config={
                "map_type": map_type,
                "vps_to_win": vps_to_win,
                "enemies": enemies,
                "representation": "mixed",
            })

            game_records = []

            # After reset(), env auto-advances to P0's first decision point
            obs = env.reset()
            done = False
            turn = 0

            while not done:
                state = env.game.state
                playable = state.playable_actions
                valid_actions = env.get_valid_actions()

                if not valid_actions:
                    break

                # Expert decides the best action
                # decide() signature: (game: Game, playable_actions) -> Action
                expert_action = expert.decide(env.game, playable)

                # Record the state and chosen action
                record = _create_record(
                    game_state=state,
                    valid_actions=playable,
                    chosen_action=expert_action,
                    player_index=0,
                    turn_number=turn,
                    vps_to_win=vps_to_win,
                )

                if record is not None:
                    game_records.append(record)

                # Find the integer action index and step the environment.
                # env.step() auto-plays all enemy turns and returns the next P0 observation.
                action_idx = _find_action_index(expert_action, valid_actions, playable)
                if action_idx is not None:
                    step_result = env.step(action_idx)
                else:
                    step_result = env.step(valid_actions[0])

                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                turn += 1
                total_turns += 1

            # Game completed
            all_records.extend(game_records)
            games_completed += 1
            pbar.update(1)

        except Exception as e:
            logger.warning(f"Game failed: {e}")
            continue
        finally:
            try:
                env.close()
            except Exception:
                pass

    pbar.close()

    # Shuffle and split
    random.shuffle(all_records)
    split_idx = int(len(all_records) * train_split)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    logger.info(
        f"SFT data generation complete: {games_completed} games, "
        f"{len(all_records)} total records "
        f"({len(train_records)} train / {len(val_records)} val)"
    )

    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
            for r in train_records:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(output_dir, "val.jsonl"), "w") as f:
            for r in val_records:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved SFT data to {output_dir}")

    return train_records, val_records


def _create_record(
    game_state: Any,
    valid_actions: List[Any],
    chosen_action: Any,
    player_index: int,
    turn_number: int,
    vps_to_win: int,
) -> Optional[Dict]:
    """
    Create a single SFT record from a game state and action.

    Returns None if the record couldn't be created (e.g., empty actions).
    """
    try:
        from ..agent.observation import format_catan_observation
        from ..agent.prompts import get_system_prompt

        # Format observation
        obs_text = format_catan_observation(
            game_state=game_state,
            valid_actions=valid_actions,
            player_index=player_index,
            verbose=True,
        )

        # Format chosen action as JSON
        action_json = _action_to_json(chosen_action, valid_actions)

        # Get system prompt
        system_prompt = get_system_prompt(version="v1", vps_to_win=vps_to_win)

        return {
            "system_prompt": system_prompt,
            "observation": obs_text,
            "action": action_json,
            "game_phase": str(game_state.current_prompt),
            "turn_number": turn_number,
            "player_index": player_index,
        }
    except Exception as e:
        logger.warning(f"Failed to create record: {e}")
        return None


def _action_to_json(action: Any, valid_actions: List[Any]) -> str:
    """
    Convert a catanatron Action to the JSON format we want the model to output.

    If the action is in the valid_actions list, we use its index.
    Otherwise, we fall back to "END_TURN".
    """
    try:
        # Find the index of this action in the valid_actions list
        action_str = str(action)
        for i, va in enumerate(valid_actions):
            if str(va) == action_str:
                return json.dumps({"action_number": i})
        return json.dumps({"action_number": 0})
    except Exception:
        return json.dumps({"action_number": 0})


def _find_action_index(
    action: Any,
    int_actions: List[int],
    rich_actions: List[Any],
) -> Optional[int]:
    """
    Find the integer action index for a given Action object.

    Tries to match by string representation and by action_type+value.

    Args:
        action: The Action namedtuple to find
        int_actions: Integer action indices from env.get_valid_actions()
        rich_actions: Rich Action objects from state.playable_actions

    Returns:
        Integer action index, or None if not found
    """
    if int_actions is None:
        return 0  # Fallback to first action

    action_str = str(action)

    # First try: match by string representation of rich actions
    for i, ra in enumerate(rich_actions):
        if str(ra) == action_str:
            if i < len(int_actions):
                return int_actions[i]

    # Second try: match by action_type and value
    try:
        target_type = getattr(action, 'action_type', None)
        target_value = getattr(action, 'value', None)
        if target_type is not None and target_value is not None:
            for i, ra in enumerate(rich_actions):
                ra_type = getattr(ra, 'action_type', None)
                ra_value = getattr(ra, 'value', None)
                if ra_type == target_type and ra_value == target_value:
                    if i < len(int_actions):
                        return int_actions[i]
    except Exception:
        pass

    # Fallback: return first valid action
    if int_actions:
        return int_actions[0]
    return None
