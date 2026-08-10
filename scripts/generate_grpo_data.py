#!/usr/bin/env python3
"""
Generate GRPO training data from VF-Guard games.
Records ALL actions with VF scores for group-relative preference learning.

Key difference from VF-Distill v2:
- VF-Distill: only saves BEST action (override-only filtering) → ~439 examples
- GRPO: saves ALL actions with scores → ~10,000+ examples with relative rankings

Usage:
    python scripts/generate_grpo_data.py --games 100 --output data/grpo/
"""

import argparse, json, logging, os, random, sys, time
import numpy as np

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CATANATRON_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CATANATRON_ROOT, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path: sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import get_value_fn
from catan_rl.rl.value import CONTENDER_WEIGHTS
from catan_rl.agent.qwen_agent import QwenCatanAgent
from catan_rl.agent.observation import format_catan_observation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GRPODataCollector(Player):
    """Plays VF-Guard and records ALL (state, actions, vf_scores)."""

    def __init__(self, color, agent, vf):
        super().__init__(color)
        self.agent = agent
        self.vf = vf
        self.records = []
        self.total_decisions = 0
        self.overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total_decisions += 1
            return actions[0] if actions else None

        action_names = {a.action_type.name for a in actions}
        if action_names == {"ROLL"}:
            return actions[0]
        if all(n == "DISCARD_RESOURCE" for n in action_names):
            return self._heuristic_discard(game, actions)
        if all(n == "MOVE_ROBBER" for n in action_names):
            return self._heuristic_robber(game, actions)

        self.total_decisions += 1

        # 1. LLM proposal
        try:
            result = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            llm_idx = result.action_index
            if not (0 <= llm_idx < len(actions)): llm_idx = 0
        except Exception:
            llm_idx = 0

        # 2. VF scores ALL actions (GRPO: keep all scores)
        scored = []
        best_idx, best_score = 0, float('-inf')
        for i, action in enumerate(actions):
            try:
                gc = game.copy(); gc.execute(action)
                score = self.vf(gc, self.color)
            except Exception:
                score = float('-inf')
            scored.append({"index": i, "score": float(score), "action_name": action.action_type.name})
            if score > best_score:
                best_score, best_idx = score, i

        was_override = (best_idx != llm_idx)

        # 3. Record ALL actions (override + 30% sample)
        if was_override or random.random() < 0.3:
            obs = format_catan_observation(game.state, actions, 0)
            self.records.append({
                "observation": obs,
                "actions": scored,
                "best_index": best_idx,
                "llm_index": llm_idx,
                "num_actions": len(actions),
                "was_override": was_override,
                "best_score": float(best_score),
                "turn": game.state.num_turns,
            })

        return actions[best_idx]

    def _heuristic_discard(self, game, actions):
        state = game.state; idx = state.colors.index(self.color)
        resources = {r: state.player_state.get(f"P{idx}_{r}_IN_HAND", 0)
                     for r in ["WOOD","BRICK","SHEEP","WHEAT","ORE"]}
        return max(actions, key=lambda a: resources.get(str(a.value), 0))

    def _heuristic_robber(self, game, actions):
        state = game.state
        best_vp, best_color = -1, None
        for i, c in enumerate(state.colors):
            if c == self.color: continue
            vp = state.player_state.get(f"P{i}_VICTORY_POINTS", 0)
            if vp > best_vp: best_vp, best_color = vp, c
        for a in actions:
            if a.value and len(a.value) > 1 and a.value[1] == best_color: return a
        return actions[0]


def generate_data(num_games=100, opponent="weighted_random", seed=42, output_dir="data/grpo"):
    random.seed(seed); np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading agent...")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/", device="cuda", load_in_4bit=True,
        lora_path="/root/autodl-tmp/catan-rl-llm/catan-rl-llm/checkpoints/ab_sft/checkpoint-200/",
        prompt_version="v1")
    agent.max_new_tokens = 16; agent.temperature = 0.1; agent.do_sample = True

    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS)
    opponent_class = WeightedRandomPlayer if opponent == "weighted_random" else None
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    all_records = []; t_start = time.time(); total_games = 0; wins = 0

    for i in range(num_games):
        gs = seed + i * 100
        random.seed(gs); shuffled = list(colors); random.shuffle(shuffled)
        ac = shuffled[0]
        collector = GRPODataCollector(ac, agent, vf)
        opponents = [opponent_class(c) for c in shuffled[1:]]
        all_players = [collector] + opponents; random.shuffle(all_players)

        try:
            game = Game(all_players, vps_to_win=10); winner = game.play()
            if winner == ac: wins += 1
        except Exception as e:
            logger.warning(f"Game {i+1} error: {e}"); continue

        total_games += 1; all_records.extend(collector.records)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            logger.info(f"Game {i+1}/{num_games} | {wins}W/{i+1-wins}L | "
                       f"{len(all_records)} records | {elapsed:.0f}s")

    output_path = os.path.join(output_dir, "grpo_train_data.jsonl")
    with open(output_path, "w") as f:
        for rec in all_records: f.write(json.dumps(rec) + "\n")

    total_time = time.time() - t_start
    overrides = sum(1 for r in all_records if r["was_override"])
    avg_actions = np.mean([r["num_actions"] for r in all_records])
    logger.info(f"Done: {total_games} games, {len(all_records)} records, "
               f"override={overrides/len(all_records)*100:.0f}%, "
               f"avg_actions={avg_actions:.1f}, time={total_time:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="data/grpo")
    args = p.parse_args()
    generate_data(args.games, args.seed, args.output)

if __name__ == "__main__":
    main()
