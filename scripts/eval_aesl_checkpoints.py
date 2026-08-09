#!/usr/bin/env python3
"""
Evaluate AESL checkpoints on Catan games.

Compares:
- best_loss: Checkpoint with lowest validation perplexity
- best_entropy: Checkpoint with highest token entropy (AESL method)
- baseline: WeightedRandom (for reference)

Each model plays N games vs WeightedRandom opponents.

Usage:
    python scripts/eval_aesl_checkpoints.py \
        --aesl_dir checkpoints/aesl \
        --games 10 --seed 42
"""

import argparse, json, logging, os, random, sys, time
import numpy as np
import torch

_FORK_CORE = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron'
_FORK_EXP = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/catanatron_experimental'
_CAT_ROOT = '/root/autodl-tmp/catan-rl-llm/Catanatron-main/'
_PROJ = '/root/autodl-tmp/catan-rl-llm/catan-rl-llm'
for _p in [_FORK_CORE, _FORK_EXP, _CAT_ROOT, _PROJ,
           os.path.join(_PROJ, 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catan_rl.agent.qwen_agent import QwenCatanAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StandalonePlayer(Player):
    """Pure LLM decision — no guardrails, tests model's own decision quality."""

    def __init__(self, color, agent):
        super().__init__(color)
        self.agent = agent
        self.total = 0

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) <= 1:
            self.total += 1
            return actions[0] if actions else None
        self.total += 1
        try:
            r = self.agent.act(
                observation=game.state,
                valid_actions=actions,
                player_index=0,
            )
            idx = r.action_index
            if not (0 <= idx < len(actions)):
                idx = 0
        except Exception as e:
            logger.debug(f"Agent error: {e}")
            idx = 0
        return actions[idx]


def load_agent(lora_path, device="cuda"):
    """Load a QwenCatanAgent with a specific LoRA checkpoint."""
    agent = QwenCatanAgent.from_pretrained(
        model_name="/root/autodl-tmp/Qwen/Qwen3-8B/",
        device=device,
        load_in_4bit=True,
        lora_path=lora_path,
        prompt_version="v1",
    )
    agent.max_new_tokens = 16
    agent.temperature = 0.1
    agent.do_sample = True
    return agent


def evaluate_checkpoint(lora_path, name, num_games=10, seed=42):
    """Evaluate a checkpoint by playing games vs WeightedRandom opponents."""
    logger.info(f"\n{'='*50}\n  Evaluating: {name}\n  Path: {lora_path}\n{'='*50}")

    if not os.path.exists(lora_path):
        logger.error(f"Checkpoint not found: {lora_path}")
        return {"name": name, "win_rate": 0.0, "wins": 0, "games": 0,
                "error": "checkpoint_missing"}

    try:
        agent = load_agent(lora_path)
    except Exception as e:
        logger.error(f"Failed to load agent: {e}")
        return {"name": name, "win_rate": 0.0, "wins": 0, "games": 0,
                "error": str(e)}

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    results = []
    t_start = time.time()

    for i in range(num_games):
        gs = seed + i * 100
        random.seed(gs)
        shuffled = list(colors)
        random.shuffle(shuffled)
        agent_color = shuffled[0]
        player = StandalonePlayer(agent_color, agent)
        opponents = [WeightedRandomPlayer(c) for c in shuffled[1:]]
        all_players = [player] + opponents
        random.shuffle(all_players)

        logger.info(f"[{name}] Game {i+1}/{num_games} (seed={gs})...")
        gt = time.time()
        try:
            game = Game(all_players, vps_to_win=10)
            winner = game.play()
            outcome = "WIN" if winner == agent_color else "LOSS"
        except Exception as e:
            logger.warning(f"Game error: {e}")
            outcome = "ERROR"

        turns = game.state.num_turns if hasattr(game, 'state') else 0
        game_time = time.time() - gt
        torch.cuda.empty_cache()

        results.append({
            "game": i + 1,
            "outcome": outcome,
            "turns": turns,
            "game_time_s": game_time,
            "seed": gs,
        })

        wins = sum(1 for r in results if r["outcome"] == "WIN")
        elapsed = time.time() - t_start
        logger.info(f"  Game {i+1}: {outcome} | {turns}t/{game_time:.0f}s | "
                    f"Running: {wins}W/{i+1-wins}L | {elapsed:.0f}s elapsed")

    wins = sum(1 for r in results if r["outcome"] == "WIN")
    completed = sum(1 for r in results if r["outcome"] != "ERROR")
    wr = wins / max(completed, 1)
    total_time = time.time() - t_start

    logger.info(f"[{name}] RESULT: {wins}/{completed} ({wr:.1%}) | {total_time:.0f}s ({total_time/60:.1f}min)")

    return {
        "name": name,
        "lora_path": lora_path,
        "win_rate": wr,
        "wins": wins,
        "games": completed,
        "errors": len(results) - completed,
        "total_time_s": total_time,
        "avg_turns": np.mean([r["turns"] for r in results if r["outcome"] != "ERROR"]),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AESL checkpoints on Catan games")
    parser.add_argument("--aesl_dir", type=str, default="checkpoints/aesl",
                       help="Directory containing AESL experiment outputs")
    parser.add_argument("--games", type=int, default=10,
                       help="Number of games per checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results JSON")
    args = parser.parse_args()

    # Load AESL summary
    summary_path = os.path.join(args.aesl_dir, "aesl_summary.json")
    if not os.path.exists(summary_path):
        logger.error(f"AESL summary not found: {summary_path}")
        logger.error("Run train_aesl.py first to generate checkpoints.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    # Load metrics log
    metrics_path = os.path.join(args.aesl_dir, "aesl_metrics.json")
    metrics_log = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_log = json.load(f)

    logger.info("=" * 60)
    logger.info("  AESL Checkpoint Evaluation")
    logger.info("=" * 60)
    logger.info(f"  AESL dir:       {args.aesl_dir}")
    logger.info(f"  Games per model: {args.games}")
    logger.info(f"  Seed:            {args.seed}")
    logger.info(f"")
    logger.info(f"  Best Loss:      step={summary['best_loss_step']}, "
                f"ppl={summary.get('best_loss_perplexity', 'N/A'):.2f}")
    logger.info(f"  Peak Entropy:   step={summary['peak_entropy_step']}, "
                f"entropy={summary['peak_entropy']:.4f}")
    logger.info(f"  Total checkpoints: {len(metrics_log)}")

    # Determine checkpoints to evaluate
    checkpoint_configs = []

    # Best loss checkpoint
    best_loss_path = os.path.join(args.aesl_dir, "best_loss")
    if os.path.exists(best_loss_path):
        checkpoint_configs.append((
            best_loss_path,
            f"Best-Loss (step={summary['best_loss_step']})",
        ))

    # Entropy peak checkpoint
    best_entropy_path = os.path.join(args.aesl_dir, "best_entropy")
    if os.path.exists(best_entropy_path):
        checkpoint_configs.append((
            best_entropy_path,
            f"Entropy-Peak (step={summary['peak_entropy_step']})",
        ))

    # If best_loss and best_entropy are the same step, also evaluate a few others
    if summary['best_loss_step'] == summary['peak_entropy_step']:
        logger.info("  Note: best_loss == peak_entropy step. Evaluating all checkpoints.")
        for m in metrics_log:
            ckpt_path = os.path.join(args.aesl_dir, f"checkpoint-{m['step']}")
            if os.path.exists(ckpt_path) and ckpt_path not in [p for p, _ in checkpoint_configs]:
                checkpoint_configs.append((
                    ckpt_path,
                    f"Step-{m['step']} (ent={m['token_entropy']:.4f}, ppl={m['perplexity']:.2f})",
                ))

    if not checkpoint_configs:
        logger.error("No checkpoints found to evaluate!")
        sys.exit(1)

    logger.info(f"\n  Evaluating {len(checkpoint_configs)} checkpoint(s):")
    for path, name in checkpoint_configs:
        logger.info(f"    - {name}")

    # Evaluate each checkpoint
    all_results = []
    for ckpt_path, ckpt_name in checkpoint_configs:
        result = evaluate_checkpoint(ckpt_path, ckpt_name, args.games, args.seed)
        all_results.append(result)
        torch.cuda.empty_cache()
        time.sleep(2)  # Brief cooldown between evaluations

    # ---- Final Report ----
    logger.info("\n" + "=" * 70)
    logger.info("  AESL EXPERIMENT RESULTS")
    logger.info("=" * 70)
    logger.info(f"  {'Method':40s} {'Win Rate':>8s} {'Avg Turns':>10s} {'Games':>8s}")
    logger.info(f"  {'-'*40} {'-'*8} {'-'*10} {'-'*8}")

    # Baselines for reference
    baselines = [
        ("WeightedRandom (baseline)", "~25%", "~200", "—"),
        ("AB-SFT (existing, step=200)", "25%", "~200", "5"),
        ("VF-Distill v2", "40%", "~200", "5"),
        ("Hybrid Agent (tools+VF)", "100%", "~90", "6"),
    ]
    for name, wr, turns, games in baselines:
        logger.info(f"  {name:40s} {wr:>8s} {turns:>10s} {games:>8s}")

    logger.info(f"  {'-'*40} {'-'*8} {'-'*10} {'-'*8}")

    for r in all_results:
        wr_str = f"{r['win_rate']:.1%}"
        turns_str = f"{r.get('avg_turns', 0):.0f}"
        games_str = f"{r['games']}"
        logger.info(f"  {r['name']:40s} {wr_str:>8s} {turns_str:>10s} {games_str:>8s}")

    logger.info("=" * 70)

    # AESL hypothesis test
    if len(all_results) >= 2:
        best_loss = all_results[0]
        best_ent = all_results[1]
        hypothesis = (
            "SUPPORTED" if best_ent['win_rate'] > best_loss['win_rate']
            else "REJECTED"
        )
        logger.info(f"\n  AESL Hypothesis ({hypothesis}):")
        logger.info(f"    Entropy-peak WR ({best_ent['win_rate']:.1%}) "
                    f"{'>' if best_ent['win_rate'] > best_loss['win_rate'] else '<='} "
                    f"Best-loss WR ({best_loss['win_rate']:.1%})")
        logger.info(f"    Entropy peak at step {summary['peak_entropy_step']}, "
                    f"best loss at step {summary['best_loss_step']}")

    # Save results
    output_data = {
        "aesl_summary": summary,
        "metrics_log": metrics_log,
        "evaluation_results": all_results,
        "config": {
            "games_per_model": args.games,
            "seed": args.seed,
            "aesl_dir": args.aesl_dir,
        },
    }

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.aesl_dir, "eval_results.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
