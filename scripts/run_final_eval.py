#!/usr/bin/env python3
"""
Final comparison evaluation: AB-SFT vs VF-Distill v2 vs RL-Guard vs Hybrid Agent.

Optimizations for faster inference:
- max_new_tokens=16 (responses are ~10 tokens)
- temperature=0 (greedy decoding)
- do_sample=False
- 5 games per method (statistical sampling)

Usage:
    python scripts/run_final_eval.py --games 5 --output results/final_comparison.json
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CATANATRON_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CATANATRON_ROOT, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import get_value_fn
from catan_rl.rl.value import CONTENDER_WEIGHTS
from catan_rl.agent.qwen_agent import QwenCatanAgent
from catan_rl.agent.observation import format_catan_observation
from catanatron_experimental.agent_tools import (
    analyze_position, check_threats, get_best_move,
)
from catanatron_experimental.rl_value_network import CatanValueNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RL_MODEL_PATH = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/rl_selfplay_model2.pt'
MODEL_PATH = '/root/autodl-tmp/Qwen/Qwen3-8B/'
AB_SFT_PATH = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm/checkpoints/ab_sft/checkpoint-200/'
VF_DISTILL_PATH = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm/checkpoints/vf_distill_v2/checkpoint-final/'


class OptimizedQwenAgent(QwenCatanAgent):
    """QwenCatanAgent with optimized generation for fast eval."""

    @classmethod
    def from_pretrained(cls, model_name=MODEL_PATH, lora_path=None, device="cuda", **kw):
        agent = super().from_pretrained(
            model_name=model_name, device=device, load_in_4bit=True,
            lora_path=lora_path, prompt_version="v1", **kw,
        )
        # Optimize for fast inference
        agent.max_new_tokens = 16
        agent.temperature = 0.1
        agent.do_sample = True
        return agent


# ==============================================================================
# Player definitions
# ==============================================================================

class StandalonePlayer(Player):
    """Pure LLM — no guardrail, no tools. Used for AB-SFT and VF-Distill eval."""

    def __init__(self, color, agent):
        super().__init__(color)
        self.agent = agent
        self.total_decisions = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total_decisions += 1
            return actions[0] if actions else None
        self.total_decisions += 1
        try:
            result = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            idx = result.action_index
            if not (0 <= idx < len(actions)):
                idx = 0
        except Exception:
            idx = 0
        return actions[idx]


class RLGuardPlayer(Player):
    """LLM proposes → RL Value Network scores ALL → picks best."""

    def __init__(self, color, agent, rl_model):
        super().__init__(color)
        self.agent = agent
        self.rl_model = rl_model
        self.total_decisions = 0
        self.overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total_decisions += 1
            return actions[0] if actions else None
        self.total_decisions += 1

        try:
            result = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            llm_idx = result.action_index
            if not (0 <= llm_idx < len(actions)):
                llm_idx = 0
        except Exception:
            llm_idx = 0

        best_idx, best_score = 0, float('-inf')
        llm_score = None
        for i, action in enumerate(actions):
            try:
                gc = game.copy()
                gc.execute(action)
                score = self.rl_model.predict(gc, self.color)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score, best_idx = score, i
            if i == llm_idx:
                llm_score = score

        if best_idx != llm_idx and llm_score is not None and best_score > llm_score:
            self.overrides += 1
            return actions[best_idx]
        return actions[llm_idx]


class HybridAgentPlayer(Player):
    """LLM with tool-enriched observations + VF guardrail."""

    def __init__(self, color, agent, rl_model, vf):
        super().__init__(color)
        self.agent = agent
        self.rl_model = rl_model
        self.vf = vf
        self.total_decisions = 0
        self.overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total_decisions += 1
            return actions[0] if actions else None
        self.total_decisions += 1

        # Run tools
        tool_info = self._run_tools(game, actions)

        # Build enriched observation
        obs = format_catan_observation(game.state, actions, 0)
        obs = self._enrich_observation(obs, tool_info)

        # LLM with enriched obs
        original_format = self.agent.format_observation
        self.agent.format_observation = lambda s, a, pi, v=True, ad=None: obs
        try:
            result = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            llm_idx = result.action_index
            if not (0 <= llm_idx < len(actions)):
                llm_idx = 0
        except Exception:
            llm_idx = 0
        finally:
            self.agent.format_observation = original_format

        # VF guardrail
        best_idx, best_score = 0, float('-inf')
        llm_score = None
        for i, action in enumerate(actions):
            try:
                gc = game.copy()
                gc.execute(action)
                score = self.vf(gc, self.color)
            except Exception:
                score = float('-inf')
            if score > best_score:
                best_score, best_idx = score, i
            if i == llm_idx:
                llm_score = score

        if best_idx != llm_idx and llm_score is not None and best_score > llm_score:
            self.overrides += 1
            return actions[best_idx]
        return actions[llm_idx]

    def _run_tools(self, game, actions):
        info = {}
        if self.rl_model:
            try:
                info["position"] = analyze_position(game, self.color, self.rl_model)
            except Exception:
                info["position"] = None

        try:
            info["threats"] = check_threats(game, self.color)
        except Exception:
            info["threats"] = None

        if self.rl_model:
            info["best_moves"] = {}
            for goal in ["any", "maximize_production"]:
                try:
                    r = get_best_move(game, self.color, goal, self.rl_model, actions)
                    info["best_moves"][goal] = {
                        "recommended": r.get("recommended"),
                        "recommended_index": r.get("recommended_index"),
                        "score": r.get("score"),
                    }
                except Exception:
                    pass
        return info

    def _enrich_observation(self, obs, tool_info):
        lines = [obs, "", "## Strategic Analysis"]

        pos = tool_info.get("position")
        if pos and isinstance(pos, dict) and "error" not in pos:
            lines.append(f"Win prob: {pos.get('win_probability', '?')}")
            lines.append(f"Assessment: {pos.get('assessment', '?')}")
            lines.append(f"Can build: {pos.get('can_build', [])}")

        threats = tool_info.get("threats")
        if threats and isinstance(threats, dict) and "error" not in threats:
            lines.append(f"Biggest threat: {threats.get('biggest_threat', '?')}")
            lines.append(f"Emergency: {threats.get('emergency', False)}")
            for t in threats.get("threats", [])[:2]:
                lines.append(f"  {t.get('color','?')}: {t.get('vp','?')} VP ({t.get('threat_level','?')})")

        best = tool_info.get("best_moves", {})
        for goal, info in best.items():
            lines.append(f"RL-best ({goal}): #{info['recommended_index']} ({info['recommended']}, s={info['score']:.3f})")

        return "\n".join(lines)


# ==============================================================================
# Evaluation runner
# ==============================================================================

def run_eval(name, player_factory, num_games, opponent_type, seed):
    opponent_class = WeightedRandomPlayer if opponent_type == "weighted_random" else None
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    results = []
    t_start = time.time()

    for i in range(num_games):
        game_seed = seed + i * 100
        random.seed(game_seed)
        shuffled = list(colors)
        random.shuffle(shuffled)

        agent_color = shuffled[0]
        player = player_factory(agent_color)
        opponents = [opponent_class(c) for c in shuffled[1:]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        game_t0 = time.time()
        logger.info(f"[{name}] Starting Game {i+1}/{num_games} (seed={game_seed})...")
        try:
            game = Game(all_players, vps_to_win=10)
            winner = game.play()
            outcome = "WIN" if winner == agent_color else "LOSS"
        except Exception as e:
            logger.warning(f"[{name}] Game {i+1} error: {e}")
            outcome = "ERROR"

        turns = game.state.num_turns if hasattr(game, 'state') else 0
        overrides = getattr(player, 'overrides', 0)
        decisions = getattr(player, 'total_decisions', 0)
        game_time = time.time() - game_t0
        results.append({"outcome": outcome, "turns": turns, "overrides": overrides,
                        "decisions": decisions, "game_time_s": game_time})

        # Clean GPU cache between games to prevent memory fragmentation
        import torch
        torch.cuda.empty_cache()

        wins = sum(1 for r in results if r["outcome"] == "WIN")
        elapsed = time.time() - t_start
        logger.info(f"[{name}] Game {i+1}/{num_games} | {wins}W/{i+1-wins}L | "
                    f"{turns}t/{game_time:.0f}s | {elapsed:.0f}s total")

    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")
    wr = wins / max(completed, 1)
    avg_overrides = np.mean([r.get('overrides', 0) for r in results])
    total_time = time.time() - t_start

    logger.info(f"[{name}] FINAL: {wins}/{completed} ({wr:.1%}) | {total_time:.0f}s ({total_time/60:.1f}min)")
    return {"name": name, "win_rate": wr, "wins": wins, "games": completed,
            "avg_overrides": avg_overrides, "time_s": total_time, "details": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--opponent", type=str, default="weighted_random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated: ab_sft,vf_distill,rl_guard,hybrid,all")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.methods == "all":
        methods = ["ab_sft", "vf_distill", "rl_guard", "hybrid"]
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    logger.info("=" * 60)
    logger.info(f"Final Comparison: {len(methods)} methods × {args.games} games")
    logger.info("=" * 60)

    # Load shared resources
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS)
    rl_model = None
    if "rl_guard" in methods or "hybrid" in methods:
        if os.path.exists(RL_MODEL_PATH):
            rl_model = CatanValueNetwork.load(RL_MODEL_PATH)
            rl_model.eval()
            logger.info("RL model loaded")
        else:
            logger.warning(f"RL model not found: {RL_MODEL_PATH}")

    all_results = []

    for method in methods:
        logger.info(f"\n{'='*40}\n  {method.upper()}\n{'='*40}")

        if method == "ab_sft":
            agent = OptimizedQwenAgent.from_pretrained(lora_path=AB_SFT_PATH, device=args.device)
            factory = lambda c: StandalonePlayer(c, agent)
        elif method == "vf_distill":
            if os.path.exists(VF_DISTILL_PATH):
                agent = OptimizedQwenAgent.from_pretrained(lora_path=VF_DISTILL_PATH, device=args.device)
            else:
                logger.warning(f"VF-Distill checkpoint not found: {VF_DISTILL_PATH}, using AB-SFT")
                agent = OptimizedQwenAgent.from_pretrained(lora_path=AB_SFT_PATH, device=args.device)
            factory = lambda c: StandalonePlayer(c, agent)
        elif method == "rl_guard":
            agent = OptimizedQwenAgent.from_pretrained(lora_path=AB_SFT_PATH, device=args.device)
            factory = lambda c: RLGuardPlayer(c, agent, rl_model)
        elif method == "hybrid":
            agent = OptimizedQwenAgent.from_pretrained(lora_path=AB_SFT_PATH, device=args.device)
            factory = lambda c: HybridAgentPlayer(c, agent, rl_model, vf)
        else:
            logger.warning(f"Unknown method: {method}")
            continue

        result = run_eval(method, factory, args.games, args.opponent, args.seed)
        all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL COMPARISON SUMMARY")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(f"  {r['name']:20s}: {r['win_rate']:.1%} ({r['wins']}/{r['games']})")
    logger.info("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        summary = {r["name"]: {"win_rate": r["win_rate"], "wins": r["wins"],
                                "games": r["games"], "time_s": r["time_s"]}
                   for r in all_results}
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
