#!/usr/bin/env python3
"""
Simulation-Guided SFT: enumerate all valid actions per state, simulate each,
and use the best action as a new SFT training target.

This approach doesn't require model generation diversity — it directly
evaluates every valid action via game simulation and picks the best one.

Pipeline:
1. Load SFT model
2. For each game state: enumerate all valid actions
3. Simulate each action 5 times → average reward
4. Create new SFT data: (state → best_simulated_action)
5. Fine-tune on this improved data

Usage:
    python scripts/simsft.py --data data/grpo/iter1/ --output data/simsft/iter1/
    python scripts/train_sft.py --data data/simsft/iter1/ --output checkpoints/simsft/iter1/
"""

import argparse
import json
import logging
import os
import pickle
import sys
import base64
import random
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def simulate_all_actions(
    serialized_game: bytes,
    playable_actions: list,
    int_actions: list,
    player_index: int = 0,
    num_rollouts: int = 5,
    max_actions: int = 20,
) -> dict:
    """
    Simulate all valid actions for a game state and return the best.

    Args:
        serialized_game: Pickled Game object
        playable_actions: List of catan Action objects
        int_actions: List of action space indices
        player_index: Agent's player index
        num_rollouts: Rollouts per action for stable estimate
        max_actions: Max actions to evaluate (actions beyond this are skipped)

    Returns:
        Dict with best_action_index, best_action_reward, all_rewards
    """
    from catanatron_gym.envs.catanatron_env import from_action_space
    from src.catan_rl.env.simulator import simulate_from_state

    # Limit to max_actions for speed
    n_actions = min(len(int_actions), max_actions)
    rewards = []

    for i in range(n_actions):
        # Get the catan Action
        try:
            catan_action = from_action_space(int_actions[i], playable_actions)
        except Exception:
            rewards.append(-1.0)
            continue

        # Simulate
        try:
            r = simulate_from_state(
                serialized_game=serialized_game,
                action=catan_action,
                player_index=player_index,
                num_rollouts=num_rollouts,
            )
            rewards.append(r)
        except Exception:
            rewards.append(-1.0)

    # Find best action
    best_idx = int(np.argmax(rewards))
    best_reward = rewards[best_idx]

    return {
        "best_action_index": best_idx,
        "best_action_reward": float(best_reward),
        "all_rewards": [float(r) for r in rewards],
        "num_actions_evaluated": n_actions,
    }


def generate_simsft_data(
    input_path: str,
    output_dir: str,
    max_records: int = 1000,
    num_rollouts: int = 5,
    max_actions_per_state: int = 20,
    sample_action_count: Optional[int] = None,
    seed: int = 42,
):
    """
    Generate Simulation-Guided SFT data from rollout records.

    For each game state, enumerate all valid actions, simulate each, and
    record the best action as the new SFT target.

    Args:
        input_path: Path to rollout.jsonl
        output_dir: Directory to save simsft data
        max_records: Max records to process
        num_rollouts: Simulation rollouts per action
        max_actions_per_state: Max actions to evaluate per state
        sample_action_count: If set, only process states with this many actions
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load records
    with open(input_path) as f:
        all_records = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(all_records)} records from {input_path}")

    # Sample records
    if max_records and len(all_records) > max_records:
        # Stratified sampling: prefer states with more actions (more decision points)
        all_records = random.sample(all_records, max_records)

    logger.info(f"Processing {len(all_records)} records")

    from catanatron_gym.envs.catanatron_env import from_action_space
    from src.catan_rl.env.simulator import simulate_from_state
    from src.catan_rl.agent.observation import format_catan_observation
    from src.catan_rl.agent.prompts import get_system_prompt

    improved_records = []
    total_actions = 0
    reward_improvements = []

    for idx, record in enumerate(all_records):
        try:
            game_bytes = base64.b64decode(record["serialized_game"])
            game = pickle.loads(game_bytes)
            state = game.state
            playable = list(state.playable_actions)
            int_actions = json.loads(record["int_actions"])

            if len(int_actions) <= 1:
                # Only one valid action — no choice, skip
                continue

            # Simulate all actions
            result = simulate_all_actions(
                serialized_game=game_bytes,
                playable_actions=playable,
                int_actions=int_actions,
                num_rollouts=num_rollouts,
                max_actions=max_actions_per_state,
            )

            best_idx = result["best_action_index"]
            best_reward = result["best_action_reward"]

            # The original SFT action (from VictoryPointPlayer) had some index
            # We compare: new best action vs original
            original_action_reward = result["all_rewards"][0] if result["all_rewards"] else 0
            improvement = best_reward - original_action_reward
            reward_improvements.append(improvement)

            # Rebuild observation + system prompt from game state
            # (needed because SFT trainer uses separate fields, not pre-built prompt)
            obs_text = format_catan_observation(
                state, playable, player_index=0, verbose=True
            )
            sys_prompt = get_system_prompt(version="v1", vps_to_win=6)

            # Create improved record in SFT-compatible format
            improved_record = {
                "system_prompt": sys_prompt,
                "observation": obs_text,
                "action": json.dumps({"action_number": best_idx}),
                "game_phase": record.get("phase", ""),
                "reward": best_reward,
                "improvement": improvement,
            }
            improved_records.append(improved_record)
            total_actions += result["num_actions_evaluated"]

        except Exception as e:
            logger.warning(f"Record {idx} failed: {e}")
            continue

        if (idx + 1) % 50 == 0:
            avg_imp = np.mean(reward_improvements[-50:]) if reward_improvements else 0
            logger.info(
                f"  Record {idx+1}/{len(all_records)}: "
                f"{len(improved_records)} improved | "
                f"avg improvement: {avg_imp:.3f} | "
                f"total sims: {total_actions}"
            )

    # Save
    output_path = os.path.join(output_dir, "train.jsonl")
    with open(output_path, "w") as f:
        for r in improved_records:
            f.write(json.dumps(r) + "\n")

    avg_improvement = np.mean(reward_improvements) if reward_improvements else 0
    pos_improvements = sum(1 for x in reward_improvements if x > 0)

    logger.info("=" * 60)
    logger.info(f"SimSFT data generation complete!")
    logger.info(f"  Records processed: {len(all_records)}")
    logger.info(f"  Improved records: {len(improved_records)}")
    logger.info(f"  Total simulations: {total_actions}")
    logger.info(f"  Avg reward improvement: {avg_improvement:.4f}")
    logger.info(f"  States with better action found: {pos_improvements}/{len(reward_improvements)} ({pos_improvements/max(len(reward_improvements),1)*100:.1f}%)")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate Simulation-Guided SFT data")
    parser.add_argument("--data", type=str, default="data/grpo/iter1/rollout.jsonl")
    parser.add_argument("--output", type=str, default="data/simsft/iter1/")
    parser.add_argument("--max_records", type=int, default=500, help="Max records to process")
    parser.add_argument("--rollouts", type=int, default=5, help="Simulation rollouts per action")
    parser.add_argument("--max_actions", type=int, default=20, help="Max actions to evaluate per state")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_simsft_data(
        input_path=args.data,
        output_dir=args.output,
        max_records=args.max_records,
        num_rollouts=args.rollouts,
        max_actions_per_state=args.max_actions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
