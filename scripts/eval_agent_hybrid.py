#!/usr/bin/env python3
"""
Hybrid Agent Evaluation: LLM + Tool-Enriched Observations.

Catanatron Agent Tools approach adapted for Qwen3-8B:
Instead of requiring the LLM to output tool-calling JSON (which needs retraining),
we run the tools FOR the LLM and include their outputs in the observation.

Architecture:
1. Run analyze_position → get win probability, assessment, can_build
2. Run check_threats → get opponent threat levels
3. Enrich observation with tool outputs
4. LLM decides action (using its trained {"action_number": N} format)
5. VF/RL guardrail: score all actions, pick best

This gives the LLM rich strategic information without changing its output format.

Usage:
    python scripts/eval_agent_hybrid.py --games 20 --scorer vf
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

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CATANATRON_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CATANATRON_ROOT, _PROJ, os.path.join(_PROJ, 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer, Player
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


class HybridAgentPlayer(Player):
    """
    LLM with tool-enriched observations + optional RL/VF guardrail.

    Tools run BEFORE the LLM decision and their outputs are included
    in the observation text. This gives the LLM strategic information
    without requiring tool-calling retraining.
    """

    def __init__(self, color, agent, rl_model=None, vf=None, scorer="vf"):
        super().__init__(color)
        self.agent = agent
        self.rl_model = rl_model
        self.vf = vf
        self.scorer = scorer  # "vf", "rl", or "none"
        self.total_decisions = 0
        self.overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            self.total_decisions += 1
            return actions[0]

        self.total_decisions += 1

        # 1. Run tools to enrich observation
        tool_info = self._run_tools(game, actions)

        # 2. Build enriched observation
        obs = format_catan_observation(game.state, actions, 0)
        obs = self._enrich_observation(obs, tool_info)

        # 3. LLM decision with enriched observation
        try:
            # We modify the observation by setting it on a wrapper
            agent_action = self._llm_act(game.state, actions, obs)
            llm_idx = agent_action.action_index
            if not (0 <= llm_idx < len(actions)):
                llm_idx = 0
        except Exception as e:
            logger.warning(f"LLM error: {e}")
            llm_idx = 0

        # 4. Guardrail: score all actions
        if self.scorer != "none":
            best_idx = self._score_actions(game, actions, llm_idx)
            if best_idx != llm_idx:
                self.overrides += 1
            return actions[best_idx]

        return actions[llm_idx]

    def _run_tools(self, game, actions):
        """Run Catanatron agent tools to get strategic analysis."""
        info = {}

        # analyze_position (RL model needed)
        if self.rl_model is not None:
            try:
                info["position"] = analyze_position(game, self.color, self.rl_model)
            except Exception as e:
                info["position"] = {"error": str(e)}
        else:
            info["position"] = None

        # check_threats (no model needed)
        try:
            info["threats"] = check_threats(game, self.color)
        except Exception as e:
            info["threats"] = {"error": str(e)}

        # get_best_move for common goals (RL model needed)
        if self.rl_model is not None:
            info["best_moves"] = {}
            for goal in ["any", "maximize_production", "build_city", "build_settlement"]:
                try:
                    result = get_best_move(game, self.color, goal, self.rl_model, actions)
                    info["best_moves"][goal] = {
                        "recommended": result.get("recommended"),
                        "recommended_index": result.get("recommended_index"),
                        "score": result.get("score"),
                    }
                except Exception:
                    pass

        return info

    def _enrich_observation(self, obs, tool_info):
        """Append tool analysis to observation text."""
        lines = [obs, "", "## Strategic Analysis (auto-generated)"]

        # Position analysis
        pos = tool_info.get("position")
        if pos and "error" not in pos:
            lines.append(f"Win probability: {pos.get('win_probability', 'N/A')}")
            lines.append(f"Assessment: {pos.get('assessment', 'N/A')}")
            lines.append(f"Can build: {pos.get('can_build', [])}")
            lines.append(f"VP needed: {pos.get('vp_needed', 'N/A')}")

        # Threats
        threats = tool_info.get("threats")
        if threats and "error" not in threats:
            lines.append(f"Biggest threat: {threats.get('biggest_threat', 'N/A')}")
            lines.append(f"Emergency: {threats.get('emergency', False)}")
            for t in threats.get("threats", [])[:3]:
                lines.append(f"  {t['color']}: {t['vp']} VP ({t['threat_level']})")

        # Best moves
        best = tool_info.get("best_moves", {})
        if best:
            lines.append("RL-recommended actions:")
            for goal, info in best.items():
                lines.append(f"  {goal}: #{info['recommended_index']} "
                            f"({info['recommended']}, score={info['score']})")

        return "\n".join(lines)

    def _llm_act(self, game_state, actions, enriched_obs):
        """Call LLM with enriched observation."""
        # Hack: temporarily replace the agent's observation formatter
        # The agent.act() calls format_catan_observation internally
        # We need to pass our enriched obs instead
        # Strategy: pass observation as a pre-formatted string via a custom path

        # Since QwenCatanAgent.act() formats the observation itself,
        # we need to bypass that. We'll monkey-patch temporarily.
        original_format = self.agent.format_observation

        def custom_format(s, a, pi, v=True, ad=None):
            return enriched_obs

        self.agent.format_observation = custom_format
        try:
            result = self.agent.act(
                observation=game_state,
                valid_actions=actions,
                player_index=0,
            )
        finally:
            self.agent.format_observation = original_format

        return result

    def _score_actions(self, game, actions, llm_idx):
        """Score all actions and return index of best."""
        best_idx, best_score = 0, float('-inf')
        llm_score = None

        for i, action in enumerate(actions):
            try:
                gc = game.copy()
                gc.execute(action)
                if self.scorer == "rl" and self.rl_model:
                    score = self.rl_model.predict(gc, self.color)
                elif self.scorer == "vf" and self.vf:
                    score = self.vf(gc, self.color)
                else:
                    continue
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_idx = i
            if i == llm_idx:
                llm_score = score

        if best_idx != llm_idx and llm_score is not None and best_score > llm_score:
            return best_idx
        return llm_idx

    @property
    def override_rate(self):
        return self.overrides / max(self.total_decisions, 1)


def load_agent(checkpoint_path, temperature=0.1, device="cuda"):
    logger.info(f"Loading agent from: {checkpoint_path}")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        device=device,
        load_in_4bit=True,
        lora_path=checkpoint_path,
        prompt_version="v1",
    )
    agent.temperature = temperature
    return agent


def load_rl_model():
    if not os.path.exists(RL_MODEL_PATH):
        logger.warning(f"RL model not found: {RL_MODEL_PATH}")
        return None
    model = CatanValueNetwork.load(RL_MODEL_PATH)
    model.eval()
    logger.info(f"RL model loaded from {RL_MODEL_PATH}")
    return model


def run_evaluation(checkpoint_path, num_games=20, opponent_type="weighted_random",
                   scorer="vf", num_players=4, temperature=0.1, device="cuda", seed=42):
    random.seed(seed)
    np.random.seed(seed)

    agent = load_agent(checkpoint_path, temperature, device)
    rl_model = load_rl_model()
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS) if scorer == "vf" else None

    opponent_class = WeightedRandomPlayer if opponent_type == "weighted_random" else RandomPlayer
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    results = []
    overrides_list = []
    t_start = time.time()

    for i in range(num_games):
        game_seed = seed + i * 100
        shuffled = list(colors)
        random.seed(game_seed)
        random.shuffle(shuffled)

        agent_color = shuffled[0]
        player = HybridAgentPlayer(agent_color, agent, rl_model, vf, scorer)
        opponents = [opponent_class(c) for c in shuffled[1:num_players]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        try:
            game_obj = Game(all_players, vps_to_win=10)
            winner = game_obj.play()
        except Exception as e:
            logger.warning(f"Game error (seed={game_seed}): {e}")
            results.append({"outcome": "ERROR"})
            continue

        outcome = "WIN" if winner == agent_color else "LOSS"
        results.append({
            "outcome": outcome,
            "turns": game_obj.state.num_turns,
            "overrides": player.overrides,
            "total_decisions": player.total_decisions,
            "override_rate": player.override_rate,
        })
        overrides_list.append(player.overrides)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            wins = sum(1 for r in results if r["outcome"] == "WIN")
            logger.info(
                f"Game {i+1}/{num_games} | Wins: {wins}/{i+1} "
                f"({wins/(i+1)*100:.0f}%) | "
                f"Avg overrides: {np.mean(overrides_list):.0f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

    total_time = time.time() - t_start
    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")

    logger.info("=" * 60)
    logger.info("  Hybrid Agent Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Scorer: {scorer}")
    logger.info(f"  Games: {num_games} | Win rate: {wins}/{completed} ({wins/max(completed,1)*100:.1f}%)")
    logger.info(f"  Avg overrides/game: {np.mean(overrides_list):.0f}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/ab_sft/checkpoint-200/")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", type=str, default="weighted_random")
    parser.add_argument("--scorer", type=str, default="vf",
                        choices=["vf", "rl", "none"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.model,
        num_games=args.games,
        opponent_type=args.opponent,
        scorer=args.scorer,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
