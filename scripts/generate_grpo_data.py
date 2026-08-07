#!/usr/bin/env python3
"""
Generate GRPO rollout data using fast bot-vs-bot gameplay.

Unlike scripts/rollout.py (which uses the model and is slow), this script
uses bot opponents (VictoryPointPlayer) to quickly generate game states for
GRPO training. The model only needs the game states, not the action choices —
those are generated during GRPO training.

Usage:
    python scripts/generate_grpo_data.py --num_games 100 --output data/grpo/iter1/
"""

import argparse
import json
import logging
import os
import pickle
import sys
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from catanatron.models.player import Color
from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from transformers import AutoTokenizer

from src.catan_rl.agent.observation import format_catan_observation
from src.catan_rl.agent.prompts import get_system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_grpo_data(
    num_games: int = 100,
    output_dir: str = "data/grpo/rollout/",
    map_type: str = "MINI",
    vps_to_win: int = 6,
    opponents: list = None,
    seed: int = 42,
):
    """Generate GRPO rollout data using bot-vs-bot gameplay."""
    if opponents is None:
        opponents = ["WeightedRandomPlayer"]

    # Map opponent names to classes
    opponent_registry = {
        "WeightedRandomPlayer": WeightedRandomPlayer,
        "VictoryPointPlayer": VictoryPointPlayer,
        "AlphaBetaPlayer": VictoryPointPlayer,
        "ValueFunctionPlayer": VictoryPointPlayer,
    }

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(
        '/root/autodl-tmp/Qwen/Qwen3-8B/',
        trust_remote_code=True,
    )

    all_records = []
    games_completed = 0
    outcomes = {"WIN": 0, "LOSS": 0, "DRAW": 0}

    logger.info(f"Generating GRPO data: {num_games} games, {map_type} map, {vps_to_win}VP")
    logger.info(f"Opponents: {opponents}")

    for game_idx in range(num_games):
        try:
            # Setup bot players
            expert = VictoryPointPlayer(Color.BLUE)
            enemies = []
            for i, opp_name in enumerate(opponents):
                color = [Color.RED, Color.WHITE, Color.ORANGE][i]
                bot_cls = opponent_registry.get(opp_name, WeightedRandomPlayer)
                enemies.append(bot_cls(color))

            env = CatanatronEnv(config={
                "map_type": map_type,
                "vps_to_win": vps_to_win,
                "enemies": enemies,
                "representation": "mixed",
            })

            env.reset()
            done = False
            game_records = 0

            while not done:
                state = env.game.state
                playable = list(state.playable_actions)
                int_actions = env.get_valid_actions()

                if not int_actions:
                    break

                # Serialize the game at this decision point
                game_bytes = pickle.dumps(env.game)

                # Format observation + system prompt
                obs_text = format_catan_observation(
                    state, playable, player_index=0, verbose=True
                )
                system_prompt = get_system_prompt(version="v1", vps_to_win=vps_to_win)

                # Build full chat-formatted prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_text},
                ]
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Encode data
                valid_json = json.dumps([str(a) for a in playable])
                int_json = json.dumps(int_actions)

                all_records.append({
                    "prompt": full_prompt,
                    "serialized_game": base64.b64encode(game_bytes).decode("utf-8"),
                    "valid_actions": valid_json,
                    "int_actions": int_json,
                    "phase": str(state.current_prompt),
                })
                game_records += 1

                # Expert bot decides the action and we step
                expert_action = expert.decide(env.game, playable)
                action_str = str(expert_action)
                action_idx = int_actions[0]  # default
                for i, pa in enumerate(playable):
                    if str(pa) == action_str and i < len(int_actions):
                        action_idx = int_actions[i]
                        break

                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated

            # Track outcome
            winner = env.game.winning_color()
            if winner is not None:
                if "BLUE" in str(winner).upper():
                    outcomes["WIN"] += 1
                else:
                    outcomes["LOSS"] += 1
            else:
                outcomes["DRAW"] += 1

            env.close()
            games_completed += 1

            if (game_idx + 1) % 20 == 0:
                logger.info(
                    f"  Game {game_idx + 1}/{num_games}: "
                    f"{len(all_records)} records | "
                    f"W:{outcomes['WIN']} L:{outcomes['LOSS']} D:{outcomes['DRAW']}"
                )

        except Exception as e:
            logger.warning(f"Game {game_idx + 1} failed: {e}")
            continue

    # Save records
    output_path = os.path.join(output_dir, "rollout.jsonl")
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"=" * 60)
    logger.info(f"GRPO data generation complete!")
    logger.info(f"  Games: {games_completed}")
    logger.info(f"  Records: {len(all_records)}")
    logger.info(f"  Outcomes: W:{outcomes['WIN']} L:{outcomes['LOSS']} D:{outcomes['DRAW']}")
    logger.info(f"  Avg records/game: {len(all_records) / max(games_completed, 1):.0f}")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO rollout data")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games")
    parser.add_argument("--output", type=str, default="data/grpo/rollout/", help="Output dir")
    parser.add_argument("--map", type=str, default="MINI", help="Map type")
    parser.add_argument("--vps", type=int, default=6, help="VPs to win")
    parser.add_argument("--opponents", type=str, nargs="+", default=["WeightedRandomPlayer"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_grpo_data(
        num_games=args.num_games,
        output_dir=args.output,
        map_type=args.map,
        vps_to_win=args.vps,
        opponents=args.opponents,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
