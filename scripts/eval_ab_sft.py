#!/usr/bin/env python3
"""
Evaluate the AlphaBeta SFT-trained model against bot opponents.

Usage:
    python scripts/eval_ab_sft.py \
        --model checkpoints/ab_sft/checkpoint-200/ \
        --games 30 \
        --opponent weighted_random
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import player_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LLMCatanPlayer(Player):
    """Wraps a QwenCatanAgent as a catanatron Player."""

    def __init__(self, color, agent):
        super().__init__(color)
        self.agent = agent
        self.total_actions = 0
        self.valid_actions = 0
        self.action_types: List[str] = []

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            self.total_actions += 1
            self.valid_actions += 1
            self.action_types.append(actions[0].action_type.name)
            return actions[0]

        try:
            agent_action = self.agent.act(
                observation=game.state,
                valid_actions=actions,
                player_index=0,
            )
            self.total_actions += 1
            idx = agent_action.action_index

            if 0 <= idx < len(actions):
                self.valid_actions += 1
                self.action_types.append(actions[idx].action_type.name)
                return actions[idx]
            else:
                logger.debug(f"Invalid idx {idx}/{len(actions)}, fallback 0")
                self.action_types.append(actions[0].action_type.name)
                return actions[0]
        except Exception as e:
            logger.warning(f"Agent error: {e}")
            self.total_actions += 1
            self.action_types.append("ERROR")
            return random.choice(actions)

    @property
    def validity_rate(self):
        return self.valid_actions / max(self.total_actions, 1)


def load_agent(checkpoint_path: str, temperature: float = 0.1, device: str = "cuda"):
    """Load the trained QwenCatanAgent with LoRA adapter."""
    from src.catan_rl.agent.qwen_agent import QwenCatanAgent

    logger.info(f"Loading agent from: {checkpoint_path}")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        device=device,
        load_in_4bit=True,
        lora_path=checkpoint_path,
        prompt_version="v1",
    )
    agent.temperature = temperature
    logger.info(f"Agent loaded (temperature={temperature})")
    return agent


def play_game(
    agent_wrapper: LLMCatanPlayer,
    opponents: List[Player],
    vps_to_win: int = 10,
    seed: int = 42,
) -> dict:
    """Play a single game and return results."""
    random.seed(seed)
    np.random.seed(seed)

    all_players = [agent_wrapper] + opponents
    random.shuffle(all_players)

    try:
        game = Game(all_players, vps_to_win=vps_to_win)
        winner_color = game.play()
    except Exception as e:
        logger.warning(f"Game error (seed={seed}): {e}")
        return {"outcome": "ERROR", "error": str(e), "turns": 0, "agent_vp": 0}

    agent_color = agent_wrapper.color

    if winner_color is None:
        outcome = "DRAW"
    elif winner_color == agent_color:
        outcome = "WIN"
    else:
        outcome = "LOSS"

    # Get agent VPs
    try:
        key = player_key(game.state, agent_color)
        agent_vp = game.state.player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", 0)
    except Exception:
        agent_vp = 0

    return {
        "outcome": outcome,
        "turns": game.state.num_turns,
        "agent_vp": agent_vp,
        "agent_color": str(agent_color),
        "winner": str(winner_color) if winner_color else "None",
        "seed": seed,
    }


def run_evaluation(
    checkpoint_path: str,
    num_games: int = 30,
    opponent_type: str = "weighted_random",
    num_players: int = 4,
    vps_to_win: int = 10,
    temperature: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
):
    """Run full evaluation: N games, track metrics, print results."""
    random.seed(seed)
    np.random.seed(seed)

    # Load agent once
    qwen_agent = load_agent(checkpoint_path, temperature, device)

    results = []
    all_action_types = []
    t_start = time.time()

    for i in range(num_games):
        game_seed = seed + i * 100

        # Create fresh players each game
        colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        random.shuffle(colors)

        agent_color = colors[0]
        opponent_colors = colors[1:1 + num_players - 1]

        player = LLMCatanPlayer(agent_color, qwen_agent)

        opponents = []
        for oc in opponent_colors:
            if opponent_type == "random":
                opponents.append(RandomPlayer(oc))
            else:
                opponents.append(WeightedRandomPlayer(oc))

        result = play_game(player, opponents, vps_to_win, game_seed)
        result["validity_rate"] = player.validity_rate
        result["total_actions"] = player.total_actions
        all_action_types.extend(player.action_types)
        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            wins_so_far = sum(1 for r in results if r["outcome"] == "WIN")
            logger.info(
                f"Game {i+1}/{num_games} | "
                f"Wins: {wins_so_far}/{i+1} ({wins_so_far/(i+1)*100:.1f}%) | "
                f"Elapsed: {elapsed:.0f}s"
            )

    total_time = time.time() - t_start

    # Compute statistics
    wins = sum(1 for r in results if r["outcome"] == "WIN")
    losses = sum(1 for r in results if r["outcome"] == "LOSS")
    draws = sum(1 for r in results if r["outcome"] == "DRAW")
    errors = sum(1 for r in results if r["outcome"] == "ERROR")
    completed = num_games - errors

    valid_rates = [r["validity_rate"] for r in results if r["outcome"] != "ERROR"]
    avg_validity = np.mean(valid_rates) if valid_rates else 0
    turns = [r["turns"] for r in results if r["turns"] > 0]
    avg_turns = np.mean(turns) if turns else 0
    agent_vps = [r["agent_vp"] for r in results if r["outcome"] != "ERROR"]

    # Action type distribution
    action_counts = Counter(all_action_types)

    logger.info("=" * 60)
    logger.info("  AlphaBeta SFT Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Opponent: {opponent_type} | Temp: {temperature}")
    logger.info(f"  Players: {num_players} | VP target: {vps_to_win}")
    logger.info(f"  Games: {num_games}")
    logger.info("-" * 60)
    logger.info(f"  Wins:   {wins:>4} ({wins/max(completed,1)*100:5.1f}%)")
    logger.info(f"  Losses: {losses:>4} ({losses/max(completed,1)*100:5.1f}%)")
    logger.info(f"  Draws:  {draws:>4} ({draws/max(completed,1)*100:5.1f}%)")
    if errors:
        logger.info(f"  Errors: {errors:>4}")
    logger.info(f"  Win rate: {wins/max(completed,1)*100:.1f}%")
    logger.info(f"  Action validity: {avg_validity:.1%}")
    logger.info(f"  Avg game turns: {avg_turns:.1f}")
    logger.info(f"  Avg agent VP: {np.mean(agent_vps):.1f}" if agent_vps else "")
    logger.info(f"  Avg time/game: {total_time/max(completed,1):.0f}s")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info("-" * 60)
    logger.info("  Action type distribution:")
    for action_type, count in action_counts.most_common(15):
        logger.info(f"    {action_type}: {count}")
    logger.info("=" * 60)

    # Save results
    output_path = os.path.join(
        os.path.dirname(checkpoint_path.rstrip('/')),
        f"eval_{opponent_type}_{num_games}g.json"
    )
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "checkpoint": checkpoint_path,
                "opponent": opponent_type,
                "num_games": num_games,
                "num_players": num_players,
                "vps_to_win": vps_to_win,
                "temperature": temperature,
            },
            "summary": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "errors": errors,
                "win_rate": wins / max(completed, 1),
                "avg_validity": float(avg_validity),
                "avg_turns": float(avg_turns),
                "avg_agent_vp": float(np.mean(agent_vps)) if agent_vps else 0,
                "total_time_s": total_time,
                "action_distribution": dict(action_counts.most_common()),
            },
            "games": results,
        }, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AB SFT trained model")
    parser.add_argument("--model", type=str, default="checkpoints/ab_sft/checkpoint-200/")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--opponent", type=str, default="weighted_random",
                        choices=["random", "weighted_random"])
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--vp", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.model,
        num_games=args.games,
        opponent_type=args.opponent,
        num_players=args.num_players,
        vps_to_win=args.vp,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
