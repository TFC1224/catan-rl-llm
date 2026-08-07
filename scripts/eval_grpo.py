#!/usr/bin/env python3
"""
Quick evaluation script for GRPO-trained Catan agent.

Runs a few games to measure action validity rate and win rate
against baseline opponents.

Usage:
    python scripts/eval_grpo.py --model checkpoints/grpo/iter1/ --games 10
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catanatron.models.player import Color
from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.catan_rl.agent.qwen_agent import QwenCatanAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(model_path: str, num_games: int = 10, map_type: str = "MINI", vps: int = 6):
    """Evaluate a trained model against WeightedRandomPlayer."""

    logger.info(f"Loading agent from: {model_path}")
    agent = QwenCatanAgent(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        lora_path=model_path,
        load_in_4bit=True,
    )

    results = {"WIN": 0, "LOSS": 0, "DRAW": 0, "valid_actions": 0, "total_actions": 0}

    for game_idx in range(num_games):
        try:
            enemies = [WeightedRandomPlayer(Color.RED)]
            env = CatanatronEnv(config={
                "map_type": map_type,
                "vps_to_win": vps,
                "enemies": enemies,
                "representation": "mixed",
            })

            obs = env.reset()
            done = False
            agent.reset_episode()
            turn = 0

            while not done and turn < 300:
                state = env.game.state
                playable = list(state.playable_actions)
                int_actions = env.get_valid_actions()

                if not int_actions:
                    break

                # Agent decision
                agent_action = agent.act(state, playable, player_index=0)
                results["total_actions"] += 1

                if agent_action.is_valid:
                    results["valid_actions"] += 1

                # Map to action space index
                seq_idx = agent_action.action_index
                if seq_idx is None or seq_idx < 0 or seq_idx >= len(int_actions):
                    seq_idx = 0
                action_idx = int_actions[seq_idx]

                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                agent.assign_reward(reward)
                turn += 1

            # Outcome
            winner = env.game.winning_color()
            if winner and "BLUE" in str(winner).upper():
                results["WIN"] += 1
            elif winner:
                results["LOSS"] += 1
            else:
                results["DRAW"] += 1

            env.close()

            logger.info(
                f"Game {game_idx+1}/{num_games}: "
                f"W:{results['WIN']} L:{results['LOSS']} D:{results['DRAW']} | "
                f"Valid: {results['valid_actions']}/{results['total_actions']}"
            )

        except Exception as e:
            logger.error(f"Game {game_idx+1} crashed: {e}")
            import traceback; traceback.print_exc()
            continue

    # Summary
    total = results["WIN"] + results["LOSS"] + results["DRAW"]
    win_rate = results["WIN"] / max(total, 1) * 100
    validity = results["valid_actions"] / max(results["total_actions"], 1) * 100

    logger.info("=" * 50)
    logger.info(f"Evaluation Complete: {total} games")
    logger.info(f"  Win rate: {win_rate:.1f}% ({results['WIN']}/{total})")
    logger.info(f"  Loss rate: {results['LOSS'] / max(total, 1) * 100:.1f}%")
    logger.info(f"  Draw rate: {results['DRAW'] / max(total, 1) * 100:.1f}%")
    logger.info(f"  Action validity: {validity:.1f}% ({results['valid_actions']}/{results['total_actions']})")
    logger.info("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--map", type=str, default="MINI")
    parser.add_argument("--vps", type=int, default=6)
    args = parser.parse_args()

    evaluate(args.model, args.games, args.map, args.vps)


if __name__ == "__main__":
    main()
