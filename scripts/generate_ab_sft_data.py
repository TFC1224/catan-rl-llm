#!/usr/bin/env python3
"""
Generate SFT training data using AlphaBetaPlayer from DarekYu's Catanatron fork.

For each game state where AlphaBeta makes a decision:
1. Enumerate all valid actions
2. Score each action with AlphaBeta's minimax value function
3. Record the best action as the training target

This produces high-quality SFT data by learning from AlphaBeta's strategic decisions.

Usage:
    python scripts/generate_ab_sft_data.py \
        --num_games 200 \
        --output data/ab_sft/ \
        --vp 10
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np

# Add fork paths BEFORE standard catanatron so fork modules take precedence
_FORK_CORE = os.path.join(os.path.dirname(__file__), '..', '..', 'Catanatron-main', 'catanatron')
_FORK_EXP = os.path.join(os.path.dirname(__file__), '..', '..', 'Catanatron-main', 'catanatron_experimental')
sys.path.insert(0, _FORK_CORE)
sys.path.insert(0, _FORK_EXP)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer, Player
from catanatron.players.minimax import AlphaBetaPlayer, get_value_fn, expand_spectrum, DEFAULT_WEIGHTS
from catanatron.players.weighted_random import WeightedRandomPlayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Collection Player — records AlphaBeta's decisions
# =============================================================================

class AlphaBetaDataCollector(Player):
    """
    Plays using AlphaBeta (depth=2) and records:
    - Formatted observation of the game state
    - The action AlphaBeta chooses (as action_number)
    - All scored actions for quality metrics

    This player is placed in a game alongside weaker opponents.
    Every decision it makes becomes an SFT training example.
    """

    def __init__(self, color, depth: int = 2):
        super().__init__(color)
        self.depth = depth
        self.records: list = []
        self.decision_count: int = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)

        if len(actions) == 0:
            return None

        # Score all actions with AlphaBeta's value function
        vf = get_value_fn("base_fn", DEFAULT_WEIGHTS, None)

        scored = []
        for i, action in enumerate(actions):
            try:
                if len(actions) > 1:
                    outcomes = expand_spectrum(game, actions)
                    action_outcomes = outcomes.get(action, [])
                    if action_outcomes:
                        score = sum(p * vf(g, self.color) for g, p in action_outcomes)
                    else:
                        gc = game.copy()
                        gc.execute(action)
                        score = vf(gc, self.color)
                else:
                    gc = game.copy()
                    gc.execute(action)
                    score = vf(gc, self.color)
                scored.append((i, action, score))
            except Exception:
                scored.append((i, action, float("-inf")))

        scored.sort(key=lambda x: x[2], reverse=True)
        best_idx, best_action, best_score = scored[0]

        # Only record decisions with multiple distinct actions (skip ROLL, END_TURN)
        if len(actions) <= 1:
            return best_action

        # Format observation using our standard formatter
        try:
            from src.catan_rl.agent.observation import format_catan_observation
            from src.catan_rl.agent.prompts import get_system_prompt

            obs_text = format_catan_observation(
                game.state, actions, player_index=0, verbose=True
            )
            sys_prompt = get_system_prompt(version="v1", vps_to_win=10)

            self.records.append({
                "system_prompt": sys_prompt,
                "observation": obs_text,
                "action": json.dumps({"action_number": best_idx}),
                "game_phase": self._infer_phase(game),
                "ab_score": float(best_score),
                "ab_spread": float(scored[0][2] - scored[-1][2]) if len(scored) > 1 else 0.0,
                "num_actions": len(actions),
                "chosen_action_type": best_action.action_type.name,
            })
            self.decision_count += 1
        except Exception as e:
            logger.debug(f"Failed to format observation: {e}")

        return best_action

    def _infer_phase(self, game) -> str:
        """Infer the game phase from state."""
        state = game.state
        if state.num_turns < 8:
            return "early"
        elif state.num_turns < 25:
            return "mid"
        else:
            return "late"


# =============================================================================
# Main Data Generation
# =============================================================================

def generate_ab_sft_data(
    num_games: int = 200,
    num_players: int = 4,
    vp_target: int = 10,
    output_dir: str = "data/ab_sft/",
    ab_depth: int = 2,
    opponent_type: str = "weighted_random",
    seed: int = 42,
):
    """
    Play N games with AlphaBeta as primary decision-maker,
    recording every decision for SFT training.

    Args:
        num_games: Number of games to play
        num_players: Total number of players (3-4)
        vp_target: Victory points needed to win (default 10)
        output_dir: Where to save train.jsonl and val.jsonl
        ab_depth: AlphaBeta search depth (2 is good balance of strength vs speed)
        opponent_type: "random" or "weighted_random"
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:num_players]

    all_records = []
    game_durations = []
    ab_wins = 0
    total_decisions = 0

    t_start = time.time()

    for game_idx in range(1, num_games + 1):
        # Create players: AlphaBeta collector + opponents
        collector = AlphaBetaDataCollector(Color.RED, depth=ab_depth)

        opponents = []
        for i, color in enumerate(colors[1:], 1):
            if opponent_type == "weighted_random":
                opponents.append(WeightedRandomPlayer(color))
            else:
                opponents.append(RandomPlayer(color))

        # Shuffle player order so AlphaBeta isn't always first
        all_players = [collector] + opponents
        random.shuffle(all_players)

        try:
            game = Game(all_players, vps_to_win=vp_target)
            winner = game.play()

            duration = game.state.num_turns
            game_durations.append(duration)
            total_decisions += collector.decision_count

            if winner == Color.RED:
                ab_wins += 1

            all_records.extend(collector.records)

        except Exception as e:
            logger.warning(f"Game {game_idx} error: {e}")
            continue

        if game_idx % 20 == 0:
            elapsed = time.time() - t_start
            rate = game_idx / elapsed * 60
            logger.info(
                f"Game {game_idx}/{num_games} | "
                f"Records: {len(all_records)} | "
                f"AB wins: {ab_wins}/{game_idx} ({ab_wins/game_idx*100:.1f}%) | "
                f"Avg turns: {np.mean(game_durations[-20:]):.1f} | "
                f"Rate: {rate:.1f} games/min"
            )

    total_time = time.time() - t_start

    # Shuffle and split 90/10
    random.shuffle(all_records)
    split = int(len(all_records) * 0.9)
    train = all_records[:split]
    val = all_records[split:]

    for name, data in [("train", train), ("val", val)]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics
    spreads = [r["ab_spread"] for r in all_records if r.get("ab_spread", 0) > 0]
    action_types = defaultdict(int)
    for r in all_records:
        action_types[r.get("chosen_action_type", "?")] += 1

    logger.info("=" * 60)
    logger.info("AlphaBeta SFT Data Generation Complete!")
    logger.info(f"  Games played: {num_games}")
    logger.info(f"  Total records: {len(all_records)}")
    logger.info(f"  Train/Val: {len(train)}/{len(val)}")
    logger.info(f"  AlphaBeta win rate: {ab_wins}/{num_games} ({ab_wins/num_games*100:.1f}%)")
    logger.info(f"  Avg game duration: {np.mean(game_durations):.1f} turns")
    logger.info(f"  Avg decisions/game: {total_decisions/num_games:.1f}")
    logger.info(f"  Avg score spread: {np.mean(spreads):.2f}" if spreads else "  No spreads")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Top actions: {dict(sorted(action_types.items(), key=lambda x: -x[1])[:8])}")
    logger.info(f"  Saved to: {output_dir}")
    logger.info("=" * 60)

    return all_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=200)
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--vp", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/ab_sft/")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--opponent", type=str, default="weighted_random",
                        choices=["random", "weighted_random"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_ab_sft_data(
        num_games=args.num_games,
        num_players=args.num_players,
        vp_target=args.vp,
        output_dir=args.output,
        ab_depth=args.depth,
        opponent_type=args.opponent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
