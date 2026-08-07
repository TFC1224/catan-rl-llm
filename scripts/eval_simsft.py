#!/usr/bin/env python3
"""
Evaluate SimSFT model: compare win rate, action validity, and VP margin
against the baseline SFT model.

Plays 20 games per model vs WeightedRandomPlayer on MINI map (6VP).

Usage:
    python scripts/eval_simsft.py
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catanatron.models.player import Color
from catanatron_gym.envs.catanatron_env import CatanatronEnv
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.catan_rl.agent.observation import format_catan_observation
from src.catan_rl.agent.action_parser import parse_action
from src.catan_rl.agent.prompts import get_system_prompt
from src.catan_rl.training.utils import (
    load_model_and_tokenizer,
    load_lora_checkpoint,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import torch


def load_agent(lora_path: str, model_name: str = "/root/autodl-tmp/Qwen/Qwen3-8B/"):
    """Load model with LoRA adapter."""
    base_model, tokenizer = load_model_and_tokenizer(model_name, load_in_4bit=True)
    model = load_lora_checkpoint(base_model, lora_path)
    model.eval()
    return model, tokenizer


def play_one_game(model, tokenizer, env, verbose: bool = False) -> dict:
    """
    Play one game with the model as BLUE vs bot opponents.

    Returns dict with: win, loss, draw, valid_actions, invalid_actions, vp
    """
    obs, info = env.reset()
    done = False
    stats = {"valid": 0, "invalid": 0, "total": 0}

    while not done:
        state = env.game.state
        playable = list(state.playable_actions)

        if not playable:
            break

        # Format observation and prompt
        obs_text = format_catan_observation(state, playable, player_index=0, verbose=True)
        sys_prompt = get_system_prompt(version="v1", vps_to_win=6)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": obs_text},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Generate action
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.9,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse action
        agent_action = parse_action(response, playable)
        stats["total"] += 1

        if verbose:
            print(f"  Response: {response[:100]}...")
            print(f"  Valid: {agent_action.is_valid}, Index: {agent_action.action_index}")

        if agent_action.is_valid and 0 <= agent_action.action_index < len(playable):
            stats["valid"] += 1
            # Map to action space index
            int_actions = env.get_valid_actions()
            if agent_action.action_index < len(int_actions):
                action_idx = int_actions[agent_action.action_index]
            else:
                action_idx = int_actions[0]  # fallback
        else:
            stats["invalid"] += 1
            # Fallback: use first valid action
            int_actions = env.get_valid_actions()
            action_idx = int_actions[0] if int_actions else 0

        obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated

    # Determine outcome
    winner = env.game.winning_color()
    if winner is not None:
        if "BLUE" in str(winner).upper():
            stats["outcome"] = "WIN"
        else:
            stats["outcome"] = "LOSS"
    else:
        stats["outcome"] = "DRAW"

    # Get agent's VP
    try:
        agent_vp = env.game.state.player_state[f"P0_VICTORY_POINTS"]
    except:
        agent_vp = 0

    stats["vp"] = agent_vp
    env.close()

    return stats


def evaluate_model(model, tokenizer, model_name: str, num_games: int = 20) -> dict:
    """Evaluate a model over multiple games."""
    all_stats = []
    outcomes = {"WIN": 0, "LOSS": 0, "DRAW": 0}
    total_valid = 0
    total_invalid = 0
    total_actions = 0

    for i in range(num_games):
        # Create env with WeightedRandomPlayer opponent
        opponent = WeightedRandomPlayer(Color.RED)

        env = CatanatronEnv(config={
            "map_type": "MINI",
            "vps_to_win": 6,
            "enemies": [opponent],
            "representation": "mixed",
        })

        try:
            stats = play_one_game(model, tokenizer, env)
            all_stats.append(stats)
            outcomes[stats["outcome"]] += 1
            total_valid += stats["valid"]
            total_invalid += stats["invalid"]
            total_actions += stats["total"]
        except Exception as e:
            logger.error(f"Game {i+1} failed: {e}")
            outcomes["LOSS"] += 1

        if (i + 1) % 5 == 0:
            print(f"  [{model_name}] Game {i+1}/{num_games}: W:{outcomes['WIN']} L:{outcomes['LOSS']} D:{outcomes['DRAW']}")

    win_rate = outcomes["WIN"] / max(num_games, 1)
    validity = total_valid / max(total_actions, 1)
    avg_vp = np.mean([s["vp"] for s in all_stats]) if all_stats else 0

    return {
        "model": model_name,
        "games": num_games,
        "win_rate": win_rate,
        "wins": outcomes["WIN"],
        "losses": outcomes["LOSS"],
        "draws": outcomes["DRAW"],
        "action_validity": validity,
        "valid_actions": total_valid,
        "invalid_actions": total_invalid,
        "avg_vp": avg_vp,
    }


def main():
    model_name = "/root/autodl-tmp/Qwen/Qwen3-8B/"
    num_games = 10

    print("=" * 60)
    print("  SimSFT vs Baseline SFT Evaluation")
    print("=" * 60)

    # Evaluate baseline SFT
    print("\n[1/2] Evaluating Baseline SFT model...")
    try:
        base_model, tokenizer = load_agent("checkpoints/sft/")
        baseline_results = evaluate_model(base_model, tokenizer, "Baseline SFT", num_games)
        del base_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Baseline SFT eval failed: {e}")
        baseline_results = None

    # Evaluate SimSFT
    print("\n[2/2] Evaluating SimSFT model...")
    try:
        simsft_model, tokenizer = load_agent("checkpoints/simsft/iter1/")
        simsft_results = evaluate_model(simsft_model, tokenizer, "SimSFT", num_games)
        del simsft_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"SimSFT eval failed: {e}")
        simsft_results = None

    # Report
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    for result in [baseline_results, simsft_results]:
        if result is None:
            continue
        print(f"\n--- {result['model']} ---")
        print(f"  Win Rate:    {result['win_rate']:.1%} ({result['wins']}W/{result['losses']}L/{result['draws']}D)")
        print(f"  Validity:    {result['action_validity']:.1%} ({result['valid_actions']}/{result['valid_actions'] + result['invalid_actions']})")
        print(f"  Avg VP:      {result['avg_vp']:.1f}")

    # Comparison
    if baseline_results and simsft_results:
        print(f"\n--- Delta ---")
        wr_delta = simsft_results["win_rate"] - baseline_results["win_rate"]
        val_delta = simsft_results["action_validity"] - baseline_results["action_validity"]
        vp_delta = simsft_results["avg_vp"] - baseline_results["avg_vp"]
        print(f"  Win Rate:    {wr_delta:+.1%}")
        print(f"  Validity:    {val_delta:+.1%}")
        print(f"  Avg VP:      {vp_delta:+.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
