#!/usr/bin/env python3
"""
Evaluate SFT model with Value Function guardrail (VF-Guard).

VF-Guard: LLM proposes actions, VF scores all candidates, picks the best.
This combines LLM strategic understanding with VF tactical optimization.

Usage:
    python scripts/eval_vf_guard.py --games 20 --opponent weighted_random
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Catanatron-main', 'catanatron'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Catanatron-main', 'catanatron_experimental'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import player_key
from catanatron.players.minimax import get_value_fn
from catan_rl.rl.value import CONTENDER_WEIGHTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VFGuardPlayer(Player):
    """
    LLM proposes actions, Value Function scores all candidates, picks the best.

    This combines:
    - LLM: strategic reasoning (what TYPE of action)
    - VF: tactical optimization (which SPECIFIC instance)
    """

    def __init__(self, color, agent, vf):
        super().__init__(color)
        self.agent = agent
        self.vf = vf
        self.total_decisions = 0
        self.vf_overrides = 0
        self.action_types: List[str] = []

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            self.total_decisions += 1
            self.action_types.append(actions[0].action_type.name)
            return actions[0]

        self.total_decisions += 1

        # 1. LLM proposes action
        try:
            agent_action = self.agent.act(
                observation=game.state,
                valid_actions=actions,
                player_index=0,
            )
            llm_idx = agent_action.action_index
            if not (0 <= llm_idx < len(actions)):
                llm_idx = 0
        except Exception:
            llm_idx = 0

        # 2. VF scores ALL actions (milliseconds)
        best_idx, best_score = 0, float('-inf')
        llm_score = None
        for i, action in enumerate(actions):
            gc = game.copy()
            gc.execute(action)
            score = self.vf(gc, self.color)
            if score > best_score:
                best_score = score
                best_idx = i
            if i == llm_idx:
                llm_score = score

        # 3. Use VF best if it beats LLM choice
        if best_idx != llm_idx and llm_score is not None and best_score > llm_score:
            self.vf_overrides += 1
            chosen = actions[best_idx]
        else:
            chosen = actions[llm_idx]

        self.action_types.append(chosen.action_type.name)
        return chosen

    @property
    def override_rate(self):
        return self.vf_overrides / max(self.total_decisions, 1)


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
    agent_wrapper: VFGuardPlayer,
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
    num_games: int = 20,
    opponent_type: str = "weighted_random",
    num_players: int = 4,
    vps_to_win: int = 10,
    temperature: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
):
    """Run full evaluation of VF-Guard."""
    random.seed(seed)
    np.random.seed(seed)

    # Load agent and VF
    qwen_agent = load_agent(checkpoint_path, temperature, device)
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS)
    logger.info("Value Function loaded (contender_fn)")

    results = []
    all_action_types = []
    all_overrides = []
    t_start = time.time()

    for i in range(num_games):
        game_seed = seed + i * 100

        colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        random.shuffle(colors)

        agent_color = colors[0]
        opponent_colors = colors[1:1 + num_players - 1]

        player = VFGuardPlayer(agent_color, qwen_agent, vf)

        opponents = []
        for oc in opponent_colors:
            if opponent_type == "random":
                opponents.append(RandomPlayer(oc))
            else:
                opponents.append(WeightedRandomPlayer(oc))

        result = play_game(player, opponents, vps_to_win, game_seed)
        result["overrides"] = player.vf_overrides
        result["total_decisions"] = player.total_decisions
        result["override_rate"] = player.override_rate
        all_action_types.extend(player.action_types)
        all_overrides.append(player.vf_overrides)
        results.append(result)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            wins_so_far = sum(1 for r in results if r["outcome"] == "WIN")
            logger.info(
                f"Game {i+1}/{num_games} | "
                f"Wins: {wins_so_far}/{i+1} ({wins_so_far/(i+1)*100:.1f}%) | "
                f"Avg overrides: {np.mean(all_overrides):.0f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

    total_time = time.time() - t_start

    # Statistics
    wins = sum(1 for r in results if r["outcome"] == "WIN")
    losses = sum(1 for r in results if r["outcome"] == "LOSS")
    draws = sum(1 for r in results if r["outcome"] == "DRAW")
    errors = sum(1 for r in results if r["outcome"] == "ERROR")
    completed = num_games - errors

    turns = [r["turns"] for r in results if r["turns"] > 0]
    avg_turns = np.mean(turns) if turns else 0
    agent_vps = [r["agent_vp"] for r in results if r["outcome"] != "ERROR"]

    action_counts = Counter(all_action_types)

    logger.info("=" * 60)
    logger.info("  VF-Guard Evaluation Results")
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
    logger.info(f"  Avg agent VP: {np.mean(agent_vps):.1f}" if agent_vps else "")
    logger.info(f"  Avg VF overrides/game: {np.mean(all_overrides):.0f}")
    logger.info(f"  Avg turns/game: {avg_turns:.1f}")
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
        f"eval_vf_guard_{opponent_type}_{num_games}g.json"
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
                "method": "vf_guard",
            },
            "summary": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "errors": errors,
                "win_rate": wins / max(completed, 1),
                "avg_agent_vp": float(np.mean(agent_vps)) if agent_vps else 0,
                "avg_turns": float(avg_turns),
                "avg_overrides": float(np.mean(all_overrides)),
                "total_time_s": total_time,
                "action_distribution": dict(action_counts.most_common()),
            },
            "games": results,
        }, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VF-Guard (LLM + VF scoring)")
    parser.add_argument("--model", type=str, default="checkpoints/ab_sft/checkpoint-200/")
    parser.add_argument("--games", type=int, default=20)
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
