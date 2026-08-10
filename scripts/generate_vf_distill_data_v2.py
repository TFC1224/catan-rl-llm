#!/usr/bin/env python3
"""
Generate VF Distillation data v2 — Catanatron AlphaBeta-supervised methodology.

Key improvements over v1:
1. Records VF scores for ALL valid actions at each decision (not just final choice)
2. Records game outcome for outcome-weighted label blending
3. Encodes VF score comparison in observation text
4. Filters to decisions where VF and LLM disagree (strongest learning signal)

Catanatron reference: train_r1_alphabeta_supervised.py — records (features, ab_score)
pairs at every decision, normalizes per-episode, blends 80% AB score + 20% outcome.

Usage:
    python scripts/generate_vf_distill_data_v2.py --games 150 --output data/vf_distill_v2/
"""
import argparse, json, logging, os, random, sys, time
from collections import Counter
from typing import List

import numpy as np

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
for _p in [_FORK_CORE, _FORK_EXP]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, 'src'))

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import player_key
from catanatron.players.minimax import get_value_fn
from catan_rl.rl.value import CONTENDER_WEIGHTS
from catan_rl.agent.observation import format_catan_observation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DistillDataCollectorV2(RandomPlayer):
    """
    Plays full games using VF-Guard: LLM proposes action type, VF scores all candidates.

    Records at EVERY decision:
    - Full observation text
    - VF scores for ALL valid actions (for comparison context)
    - VF-best action (target for distillation)
    - LLM proposed action
    - Whether VF overrode LLM
    - Action type distribution

    After game ends: records outcome (WIN/LOSS) for outcome-weighted blending.
    """

    def __init__(self, color, agent, vf, opponent_class=WeightedRandomPlayer, num_players=4):
        super().__init__(color)
        self.agent = agent
        self.vf = vf
        self.opponent_class = opponent_class
        self.num_players = num_players
        self.records: List[dict] = []
        self.total_decisions = 0
        self.vf_overrides = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            self.total_decisions += 1
            return actions[0]

        self.total_decisions += 1

        # 1. LLM proposal
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

        # 2. VF scores ALL actions (Catanatron-style: evaluate every candidate)
        vf_scores = []
        best_idx, best_score = 0, float('-inf')
        for i, action in enumerate(actions):
            gc = game.copy()
            gc.execute(action)
            score = self.vf(gc, self.color)
            vf_scores.append({
                "idx": i,
                "action_type": action.action_type.name,
                "action_str": str(action),
                "vf_score": round(score, 2),
            })
            if score > best_score:
                best_score = score
                best_idx = i

        # Sort by score descending for observation enrichment
        vf_scores_sorted = sorted(vf_scores, key=lambda x: x["vf_score"], reverse=True)

        llm_score = vf_scores[llm_idx]["vf_score"]
        was_override = (best_idx != llm_idx and best_score > llm_score)

        if was_override:
            self.vf_overrides += 1

        # 3. Build observation using agent's OWN formatter (MATCHES INFERENCE FORMAT)
        # IMPORTANT: VF analysis is stored SEPARATELY (not in observation)
        # This avoids distribution shift — model sees standard format at both train & inference
        # Catanatron approach: VF scores are labels, not input features
        obs = format_catan_observation(game.state, actions, 0)

        # 4. Record (Catanatron-style: always record best action as target)
        best_action = actions[best_idx]
        vf_analysis = self._build_vf_analysis(vf_scores_sorted, best_idx)
        self.records.append({
            "system_prompt": self.agent.get_system_prompt(),
            "observation": obs,
            "vf_analysis": vf_analysis,  # separate field for analysis/debugging
            "action": json.dumps({"action_number": best_idx}),
            "llm_action_idx": llm_idx,
            "vf_best_idx": best_idx,
            "vf_best_score": round(best_score, 2),
            "llm_score": round(llm_score, 2),
            "was_override": was_override,
            "num_actions": len(actions),
            "chosen_action_type": best_action.action_type.name,
            "vf_scores": vf_scores_sorted[:5],  # top 5 for reference
        })

        # 5. Return VF-best action (ensuring 100% WR data quality)
        return best_action

    def _build_vf_analysis(self, vf_scores_sorted, best_idx):
        """Build VF analysis block — teaches model to recognize good vs bad moves."""
        lines = ["\n## VF Score Analysis (for training reference)"]
        lines.append(f"Best action: #{best_idx} (score={vf_scores_sorted[0]['vf_score']})")
        lines.append("Score comparison (top actions):")
        for s in vf_scores_sorted[:5]:
            marker = " ★BEST" if s["idx"] == best_idx else ""
            lines.append(f"  Action #{s['idx']}: {s['action_type']} — VF score={s['vf_score']}{marker}")
        return "\n".join(lines)

    @property
    def override_rate(self):
        return self.vf_overrides / max(self.total_decisions, 1)


def load_agent(checkpoint_path: str, device: str = "cuda"):
    from src.catan_rl.agent.qwen_agent import QwenCatanAgent
    logger.info(f"Loading agent from: {checkpoint_path}")
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        device=device,
        load_in_4bit=True,
        lora_path=checkpoint_path,
        prompt_version="v1",
    )
    agent.temperature = 0.1
    return agent


def play_game_with_collector(agent, vf, colors, seed, opponent_class, num_players):
    """Play one game and return collector with all records."""
    random.seed(seed)
    np.random.seed(seed)

    collector_color = colors[0]
    collector = DistillDataCollectorV2(
        color=collector_color, agent=agent, vf=vf,
        opponent_class=opponent_class, num_players=num_players,
    )

    opponents = [opponent_class(c) for c in colors[1:num_players]]
    all_players = [collector] + opponents
    random.shuffle(all_players)

    try:
        game_obj = Game(all_players, vps_to_win=10)
        winner = game_obj.play()
    except Exception as e:
        logger.warning(f"Game error (seed={seed}): {e}")
        return collector, None

    outcome = "WIN" if winner == collector.color else "LOSS"
    return collector, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--output", type=str, default="data/vf_distill_v2/")
    parser.add_argument("--model", type=str, default="checkpoints/ab_sft/checkpoint-200/")
    parser.add_argument("--opponent", type=str, default="weighted_random")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load agent and VF
    agent = load_agent(args.model, args.device)
    vf = get_value_fn("contender_fn", CONTENDER_WEIGHTS)
    logger.info("VF loaded (contender_fn)")

    opponent_class = WeightedRandomPlayer if args.opponent == "weighted_random" else RandomPlayer
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    all_train = []
    all_val = []
    wins = losses = errors = 0
    total_overrides = 0
    total_decisions = 0
    action_types = []
    t_start = time.time()

    for i in range(1, args.games + 1):
        game_seed = args.seed + i * 100
        shuffled = list(colors)
        random.seed(game_seed)
        random.shuffle(shuffled)

        collector, outcome = play_game_with_collector(
            agent, vf, shuffled, game_seed, opponent_class, args.num_players
        )

        total_decisions += collector.total_decisions
        total_overrides += collector.vf_overrides
        action_types.extend([r["chosen_action_type"] for r in collector.records])

        if outcome is None:
            errors += 1
            continue

        # Annotate records with outcome (Catanatron-style: outcome-aware labels)
        if outcome == "WIN":
            wins += 1
        else:
            losses += 1

        for r in collector.records:
            r["outcome"] = outcome

        # 90/10 train/val split
        target = all_val if i % 10 == 0 else all_train
        target.extend(collector.records)

        if i % 20 == 0:
            elapsed = time.time() - t_start
            wr = wins / max(i - errors, 1) * 100
            logger.info(
                f"Game {i}/{args.games} | WR: {wr:.0f}% | "
                f"Decisions: {total_decisions} | "
                f"Override rate: {total_overrides/max(total_decisions,1)*100:.0f}% | "
                f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)"
            )

    # Save
    def save_jsonl(data, path):
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_jsonl(all_train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(all_val, os.path.join(args.output, "val.jsonl"))

    total_time = time.time() - t_start
    completed = args.games - errors
    logger.info("=" * 60)
    logger.info("  VF Distill Data Generation v2 — Complete")
    logger.info("=" * 60)
    logger.info(f"  Games: {args.games} ({completed} completed, {errors} errors)")
    logger.info(f"  Win rate: {wins}/{completed} ({wins/max(completed,1)*100:.1f}%)")
    logger.info(f"  Total decisions: {total_decisions}")
    logger.info(f"  Override rate: {total_overrides/max(total_decisions,1)*100:.1f}%")
    logger.info(f"  Train records: {len(all_train)} | Val records: {len(all_val)}")
    logger.info(f"  Action distribution: {dict(Counter(action_types).most_common())}")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
