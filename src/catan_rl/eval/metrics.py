"""
Evaluation metrics for Catan agent performance.

Computes win rates, ELO estimates, action statistics, and
resource efficiency metrics from tournament results.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(tournament_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive metrics from tournament results.

    Args:
        tournament_results: Output from run_tournament()

    Returns:
        Dict of computed metrics
    """
    games = tournament_results.get("games", [])

    if not games:
        return {"error": "No games to analyze"}

    # Basic win/loss/draw
    outcomes = [g.get("outcome", "UNKNOWN") for g in games]
    wins = outcomes.count("WIN")
    losses = outcomes.count("LOSS")
    draws = outcomes.count("DRAW") + outcomes.count("TIMEOUT") + outcomes.count("UNKNOWN")
    total = len(outcomes)

    metrics = {
        "num_games": total,
        "win_rate": wins / total if total > 0 else 0,
        "loss_rate": losses / total if total > 0 else 0,
        "draw_rate": draws / total if total > 0 else 0,
    }

    # VP metrics
    vp_margins = [g.get("vp_margin", 0) for g in games]
    metrics["avg_vp_margin"] = float(np.mean(vp_margins))
    metrics["std_vp_margin"] = float(np.std(vp_margins))
    metrics["max_vp_margin"] = int(np.max(vp_margins))
    metrics["min_vp_margin"] = int(np.min(vp_margins))

    # Turn metrics
    turns = [g.get("turns", 0) for g in games]
    metrics["avg_turns"] = float(np.mean(turns))
    metrics["std_turns"] = float(np.std(turns))

    # Action validity
    valid_counts = [g.get("valid_actions", 0) for g in games]
    total_counts = [g.get("total_actions", 0) for g in games]
    total_valid = sum(valid_counts)
    total_actions = sum(total_counts)
    metrics["action_validity_rate"] = total_valid / max(total_actions, 1)

    # ELO estimate (simplified)
    metrics["elo_estimate"] = estimate_elo(metrics["win_rate"])

    return metrics


def estimate_elo(
    win_rate: float,
    opponent_elo: float = 1500,
    k_factor: float = 32,
) -> float:
    """
    Estimate ELO rating from win rate.

    Uses the standard ELO formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Solving for R_A:
    R_A = R_B - 400 * log10(1/win_rate - 1)

    Args:
        win_rate: Agent's win rate (0.0 to 1.0)
        opponent_elo: Estimated opponent ELO
        k_factor: K-factor (unused in estimation, included for API compatibility)

    Returns:
        Estimated ELO rating
    """
    win_rate = max(0.01, min(0.99, win_rate))

    try:
        rating_diff = -400 * math.log10(1.0 / win_rate - 1.0)
        return opponent_elo + rating_diff
    except (ValueError, ZeroDivisionError):
        return opponent_elo


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as a human-readable table.

    Args:
        metrics: Dict from compute_metrics()

    Returns:
        Formatted string table
    """
    rows = []

    # Handle special case of missing opponent_vp_mean
    opp_vp_mean = metrics.get("opponent_vp_mean", "N/A")
    if isinstance(opp_vp_mean, float):
        opp_vp_mean = f"{opp_vp_mean:.1f}"

    table = [
        ("Metric", "Value"),
        ("-" * 30, "-" * 15),
        ("Games", str(metrics.get("num_games", 0))),
        ("Win Rate", f"{metrics.get('win_rate', 0):.1%}"),
        ("Loss Rate", f"{metrics.get('loss_rate', 0):.1%}"),
        ("Draw Rate", f"{metrics.get('draw_rate', 0):.1%}"),
        ("Avg VP Margin", f"{metrics.get('avg_vp_margin', 0):.1f}"),
        ("Avg Turns", f"{metrics.get('avg_turns', 0):.1f}"),
        ("Action Validity", f"{metrics.get('action_validity_rate', 0):.1%}"),
        ("ELO Estimate", f"{metrics.get('elo_estimate', 0):.0f}"),
    ]

    for metric, value in table:
        rows.append(f"  {metric:<30} {value:>15}")

    return "\n".join(rows)
