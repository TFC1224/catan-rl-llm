#!/usr/bin/env python3
"""Larger Hybrid Agent evaluation + ablation study."""

import sys, os, time, random, json, logging, argparse
import numpy as np
import torch

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
from catanatron_experimental.agent_tools import analyze_position, check_threats, get_best_move
from catanatron_experimental.rl_value_network import CatanValueNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RL_MODEL_PATH = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/rl_selfplay_model2.pt'
AB_SFT_PATH = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm/checkpoints/ab_sft/checkpoint-200/'


class HybridForEval(Player):
    def __init__(self, color, agent, rl_model, vf, scorer="vf"):
        super().__init__(color)
        self.agent = agent
        self.rl_model = rl_model
        self.vf = vf
        self.scorer = scorer
        self.total_decisions = 0
        self.overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total_decisions += 1
            return actions[0] if actions else None
        self.total_decisions += 1

        tool_info = self._run_tools(game, actions)
        obs = format_catan_observation(game.state, actions, 0)
        obs = self._enrich_observation(obs, tool_info)

        orig = self.agent.format_observation
        self.agent.format_observation = lambda s, a, pi, v=True, ad=None: obs
        try:
            r = self.agent.act(observation=game.state, valid_actions=actions, player_index=0)
            llm_idx = r.action_index
            if not (0 <= llm_idx < len(actions)):
                llm_idx = 0
        except Exception:
            llm_idx = 0
        finally:
            self.agent.format_observation = orig

        if self.scorer == "none":
            return actions[llm_idx]

        best_idx, best_score = 0, float('-inf')
        for i, action in enumerate(actions):
            try:
                gc = game.copy(); gc.execute(action)
                if self.scorer == "vf":
                    score = self.vf(gc, self.color)
                elif self.scorer == "rl":
                    score = self.rl_model.predict(gc, self.color)
                else:
                    score = 0
            except Exception:
                score = float('-inf')
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx != llm_idx:
            self.overrides += 1
        return actions[best_idx]

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
                    info["best_moves"][goal] = {"recommended": r.get("recommended"),
                                                  "recommended_index": r.get("recommended_index"),
                                                  "score": r.get("score")}
                except Exception:
                    pass
        return info

    def _enrich_observation(self, obs, tool_info):
        lines = [obs, "", "## Strategic Analysis"]
        pos = tool_info.get("position")
        if pos and isinstance(pos, dict) and "error" not in pos:
            lines.append(f"Win prob: {pos.get('win_probability', '?')} | Assessment: {pos.get('assessment', '?')}")
            lines.append(f"Can build: {pos.get('can_build', [])}")
        threats = tool_info.get("threats")
        if threats and isinstance(threats, dict) and "error" not in threats:
            lines.append(f"Biggest threat: {threats.get('biggest_threat', '?')} | Emergency: {threats.get('emergency', False)}")
            for t in threats.get("threats", [])[:2]:
                lines.append(f"  {t.get('color','?')}: {t.get('vp','?')} VP ({t.get('threat_level','?')})")
        for goal, info in tool_info.get("best_moves", {}).items():
            lines.append(f"RL-best ({goal}): #{info['recommended_index']} ({info['recommended']}, s={info['score']:.3f})")
        return "\n".join(lines)


def run_eval(name, agent, rl_model, vf, scorer, num_games, seed):
    opponent_class = WeightedRandomPlayer
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    results = []
    t_start = time.time()

    for i in range(num_games):
        gs = seed + i * 100
        random.seed(gs)
        shuffled = list(colors)
        random.shuffle(shuffled)
        ac = shuffled[0]
        player = HybridForEval(ac, agent, rl_model, vf, scorer)
        opponents = [opponent_class(c) for c in shuffled[1:]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        logger.info(f"[{name}] Game {i+1}/{num_games} (seed={gs})...")
        gt = time.time()
        try:
            game = Game(all_players, vps_to_win=10)
            winner = game.play()
            outcome = "WIN" if winner == ac else "LOSS"
        except Exception as e:
            logger.warning(f"Error: {e}")
            outcome = "ERROR"

        turns = game.state.num_turns if hasattr(game, 'state') else 0
        game_time = time.time() - gt
        torch.cuda.empty_cache()
        results.append({"outcome": outcome, "turns": turns, "game_time_s": game_time})
        wins = sum(1 for r in results if r["outcome"] == "WIN")
        elapsed = time.time() - t_start
        logger.info(f"[{name}] Game {i+1}/{num_games} | {wins}W/{i+1-wins}L | {turns}t/{game_time:.0f}s | {elapsed:.0f}s total")

    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")
    wr = wins / max(completed, 1)
    total_time = time.time() - t_start
    logger.info(f"[{name}] FINAL: {wins}/{completed} ({wr:.1%}) | {total_time:.0f}s ({total_time/60:.1f}min)")
    return {"name": name, "win_rate": wr, "wins": wins, "games": completed, "time_s": total_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", type=str, default="hybrid_vf,hybrid_rl,hybrid_none")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]

    logger.info("Loading shared resources...")
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS)
    rl_model = CatanValueNetwork.load(RL_MODEL_PATH)
    rl_model.eval()
    agent = None

    all_results = []
    for config in configs:
        if agent is None:
            agent = QwenCatanAgent.from_pretrained(
                model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
                device="cuda", load_in_4bit=True, lora_path=AB_SFT_PATH, prompt_version="v1",
            )
            agent.max_new_tokens = 16
            agent.temperature = 0.1
            agent.do_sample = True

        logger.info(f"\n{'='*40}\n  {config}\n{'='*40}")

        if config == "hybrid_vf":
            scorer = "vf"
        elif config == "hybrid_rl":
            scorer = "rl"
        elif config == "hybrid_none":
            scorer = "none"
        else:
            logger.warning(f"Unknown config: {config}")
            continue

        result = run_eval(config, agent, rl_model, vf, scorer, args.games, args.seed)
        all_results.append(result)

    logger.info("\n" + "=" * 60)
    logger.info("  HYBRID AGENT ABLATION SUMMARY")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(f"  {r['name']:20s}: {r['win_rate']:.1%} ({r['wins']}/{r['games']})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
