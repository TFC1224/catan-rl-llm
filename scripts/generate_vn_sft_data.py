#!/usr/bin/env python3
"""
Use a trained Catan Value Network to score actions and generate improved SFT data.

For each game state:
1. Enumerate all valid actions
2. Score each action with the value network (single forward pass, ~1ms per action)
3. Select the action with highest predicted win probability
4. Create SFT training data: (observation -> best_action)

This is MUCH faster than simulation (milliseconds vs seconds per action)
and provides meaningful action discrimination via the learned value function.

Supports both:
- Our trained model (checkpoints/rl_value/value_network.pt)
- DarekYu's pre-trained model (Catanatron-main/rl_selfplay_model.pt)

Usage:
    python scripts/generate_vn_sft_data.py \
        --model Catanatron-main/rl_selfplay_model.pt \
        --data data/grpo/iter1/rollout.jsonl \
        --output data/vn_sft/iter1/ \
        --max_records 2000
"""

import argparse
import base64
import json
import logging
import os
import pickle
import random
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Catanatron-main', 'catanatron'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Catanatron-main', 'catanatron_experimental'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def score_actions_with_value_net(
    game,
    color,
    playable_actions: list,
    model,
    device: str = "cuda",
) -> list:
    """
    Score all valid actions using the value network.

    For each action:
    1. Clone the game
    2. Execute the action
    3. Extract features from the resulting state
    4. Score with the value network

    Returns list of (action_index, action, score) sorted by score descending.
    """
    from src.catan_rl.rl.value_network import extract_features

    results = []

    for i, action in enumerate(playable_actions):
        try:
            gc = game.copy()
            gc.execute(action)
            features = extract_features(gc, color)
            x = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(x).item()
            results.append((i, action, score))
        except Exception:
            results.append((i, action, 0.0))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def load_value_network(model_path: str, device: str = "cuda"):
    """Load a CatanValueNetwork from checkpoint."""
    from src.catan_rl.rl.value_network import CatanValueNetwork
    model = CatanValueNetwork.load(model_path).to(device)
    model.eval()
    logger.info(f"Loaded value network from {model_path}")
    return model


def generate_vn_sft_data(
    model_path: str,
    data_path: str,
    output_dir: str,
    max_records: int = 2000,
    min_actions: int = 2,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Generate SFT data using value network action scoring.

    Args:
        model_path: Path to trained value network
        data_path: Path to rollout JSONL
        output_dir: Where to save train.jsonl and val.jsonl
        max_records: Max game states to process
        min_actions: Only process states with at least this many actions
        device: "cuda" or "cpu"
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_value_network(model_path, device)

    # Load rollout data
    logger.info(f"Loading rollout data from {data_path}...")
    with open(data_path) as f:
        all_records = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(all_records)} records")

    # Sample records
    if max_records and len(all_records) > max_records:
        all_records = random.sample(all_records, max_records)

    # Import observation formatter
    from src.catan_rl.agent.observation import format_catan_observation
    from src.catan_rl.agent.prompts import get_system_prompt

    improved_records = []
    total_scored = 0
    score_spreads = []
    best_was_first = 0  # count when the first action (original) was best

    for idx, record in enumerate(all_records):
        try:
            game_bytes = base64.b64decode(record["serialized_game"])
            game = pickle.loads(game_bytes)
            state = game.state
            playable = list(state.playable_actions)
            int_actions = json.loads(record["int_actions"])

            if len(int_actions) < min_actions:
                continue

            # Score all actions with the value network
            scored = score_actions_with_value_net(
                game, state.colors[0], playable, model, device,
            )

            best_idx, best_action, best_score = scored[0]
            worst_score = scored[-1][2]
            spread = best_score - worst_score
            score_spreads.append(spread)

            if best_idx == 0:
                best_was_first += 1

            # Format observation
            obs_text = format_catan_observation(state, playable, player_index=0, verbose=True)
            sys_prompt = get_system_prompt(version="v1", vps_to_win=6)

            improved_records.append({
                "system_prompt": sys_prompt,
                "observation": obs_text,
                "action": json.dumps({"action_number": best_idx}),
                "game_phase": record.get("phase", ""),
                "vn_score": float(best_score),
                "vn_spread": float(spread),
                "num_actions": len(playable),
            })
            total_scored += len(playable)

        except Exception as e:
            logger.debug(f"Record {idx} failed: {e}")
            continue

        if (idx + 1) % 200 == 0:
            avg_spread = np.mean(score_spreads[-200:]) if score_spreads else 0
            logger.info(
                f"  Record {idx+1}/{len(all_records)}: "
                f"{len(improved_records)} improved | "
                f"avg spread: {avg_spread:.4f} | "
                f"chose non-[0]: {len(improved_records) - best_was_first}/{len(improved_records)}"
            )

    # Split 90/10 train/val
    random.shuffle(improved_records)
    split = int(len(improved_records) * 0.9)
    train = improved_records[:split]
    val = improved_records[split:]

    for name, data in [("train", train), ("val", val)]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    avg_spread = np.mean(score_spreads) if score_spreads else 0
    pct_changed = (len(improved_records) - best_was_first) / max(len(improved_records), 1) * 100

    logger.info("=" * 60)
    logger.info("Value Network SFT data generation complete!")
    logger.info(f"  Records processed: {len(all_records)}")
    logger.info(f"  Improved records: {len(improved_records)}")
    logger.info(f"  Train/Val: {len(train)}/{len(val)}")
    logger.info(f"  Total actions scored: {total_scored}")
    logger.info(f"  Avg score spread: {avg_spread:.4f}")
    logger.info(f"  Chose non-[0] action: {len(improved_records) - best_was_first}/{len(improved_records)} ({pct_changed:.1f}%)")
    logger.info(f"  Saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to value network .pt")
    parser.add_argument("--data", type=str, default="data/grpo/iter1/rollout.jsonl")
    parser.add_argument("--output", type=str, default="data/vn_sft/iter1/")
    parser.add_argument("--max_records", type=int, default=2000)
    parser.add_argument("--min_actions", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_vn_sft_data(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_records=args.max_records,
        min_actions=args.min_actions,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
