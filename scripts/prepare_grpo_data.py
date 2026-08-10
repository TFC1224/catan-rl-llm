#!/usr/bin/env python3
"""
Convert existing GRPO rollout data to training format with VF scores.

For each record, deserializes the game state, scores ALL valid actions with VF,
and outputs (observation, [scored_actions]) for GRPO training.

This is much faster than generating new games — pure data processing.

Usage:
    python scripts/prepare_grpo_data.py --input data/grpo/iter1/rollout.jsonl \
        --output data/grpo/grpo_train.jsonl --max_records 5000
"""

import argparse, json, logging, os, pickle, random, sys, time
import codecs, base64
import numpy as np

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CATANATRON_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CATANATRON_ROOT, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path: sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.models.board import STATIC_GRAPH
from catanatron.game import generate_playable_actions
from catanatron.players.minimax import get_value_fn, expand_spectrum
from catan_rl.rl.value import CONTENDER_WEIGHTS
from catan_rl.agent.observation import format_catan_observation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def patch_old_game(game):
    """Add missing attributes to old pickled Game objects for modern Catanatron.

    Old pickled Game objects don't have newer attributes that Game.copy(),
    State.copy(), and Board.copy() require. This patches them with sensible defaults.
    """
    # Board-level (patch FIRST, as generate_playable_actions accesses board attrs)
    b = game.state.board
    if not hasattr(b, 'buildable_subgraph'):
        b.buildable_subgraph = STATIC_GRAPH.subgraph(b.map.land_nodes)
    if not hasattr(b, 'buildable_edges_cache'):
        b.buildable_edges_cache = {}
    if not hasattr(b, 'player_port_resources_cache'):
        b.player_port_resources_cache = {}

    # State-level
    s = game.state
    if not hasattr(s, 'friendly_robber'):
        s.friendly_robber = False
    if not hasattr(s, 'action_records'):
        s.action_records = []
    if not hasattr(s, 'discard_counts'):
        s.discard_counts = [0, 0, 0, 0]
    if not hasattr(s, 'is_resolving_trade'):
        s.is_resolving_trade = False
    if not hasattr(s, 'current_trade'):
        s.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if not hasattr(s, 'acceptees'):
        s.acceptees = (False, False, False, False)

    # Game-level
    if not hasattr(game, 'friendly_robber'):
        game.friendly_robber = False
    if not hasattr(game, 'playable_actions'):
        game.playable_actions = generate_playable_actions(game.state)


def deserialize_game(serialized_str: str):
    """Deserialize pickled+base64 game state, patching missing attributes."""
    try:
        data = base64.b64decode(serialized_str)
        game = pickle.loads(data)
        patch_old_game(game)
        return game
    except Exception as e:
        try:
            # Try alternative: hex-encoded
            data = codecs.decode(serialized_str.encode(), 'hex') if len(serialized_str) < 1000 else serialized_str
            game = pickle.loads(data)
            patch_old_game(game)
            return game
        except Exception:
            return None


def score_actions(game, color, vf, actions):
    """Score ALL valid actions with VF."""
    scored = []
    for i, action in enumerate(actions):
        try:
            gc = game.copy()
            gc.execute(action)
            score = vf(gc, color)
        except Exception:
            score = float('-inf')
        scored.append({"index": i, "score": float(score), "action_name": action.action_type.name})
    return scored


def extract_observation(game, actions, player_index=0):
    """Extract observation text from game state."""
    try:
        return format_catan_observation(game.state, actions, player_index, verbose=True)
    except Exception:
        return ""


def process_data(input_path, output_path, max_records=5000, sample_rate=1.0):
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed = 0
    skipped = 0
    t_start = time.time()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            if i >= max_records:
                break
            if random.random() > sample_rate:
                continue

            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Deserialize game
            serialized = rec.get("serialized_game", "")
            if not serialized:
                skipped += 1
                continue

            game = deserialize_game(serialized)
            if game is None:
                skipped += 1
                continue

            # Get player color
            color = game.state.colors[0]  # First player

            # Get valid actions from game state
            try:
                actions = list(game.state.playable_actions)
            except Exception:
                try:
                    actions = list(game.playable_actions)
                except Exception:
                    skipped += 1
                    continue

            if len(actions) <= 1:
                skipped += 1
                continue

            # Score all actions with VF
            scored = score_actions(game, color, vf, actions)

            # Build observation
            obs = extract_observation(game, actions, 0)
            if not obs:
                obs = rec.get("prompt", "")  # Fallback to stored prompt

            # Find best index
            valid_scores = [(a["index"], a["score"]) for a in scored if a["score"] > float('-inf')]
            if not valid_scores:
                skipped += 1
                continue
            best_idx = max(valid_scores, key=lambda x: x[1])[0]

            # Normalize scores within group (GRPO: relative)
            scores = [a["score"] for a in scored]
            min_s, max_s = min(scores), max(scores)
            if max_s > min_s:
                for a in scored:
                    a["norm_score"] = (a["score"] - min_s) / (max_s - min_s)
            else:
                for a in scored:
                    a["norm_score"] = 0.5

            # Output
            fout.write(json.dumps({
                "observation": obs,
                "actions": scored,
                "best_index": best_idx,
                "num_actions": len(actions),
                "turn": game.state.num_turns if hasattr(game.state, 'num_turns') else 0,
            }) + "\n")
            processed += 1

            if processed % 200 == 0:
                elapsed = time.time() - t_start
                logger.info(f"Processed {processed} records ({processed/elapsed:.0f}/s) | skipped={skipped}")

    total_time = time.time() - t_start
    logger.info(f"Done: {processed} records saved to {output_path} ({total_time:.0f}s, {processed/total_time:.0f}/s)")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/grpo/iter1/rollout.jsonl")
    parser.add_argument("--output", type=str, default="data/grpo/grpo_train.jsonl")
    parser.add_argument("--max_records", type=int, default=5000)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    args = parser.parse_args()
    process_data(args.input, args.output, args.max_records, args.sample_rate)


if __name__ == "__main__":
    main()
