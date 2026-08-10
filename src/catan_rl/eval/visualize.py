"""
Visualization utilities for evaluation results.

Generates plots for:
- Win rate bar charts (by opponent, by map)
- Learning curves (SFT loss, GRPO reward over steps)
- Action distribution by game phase
"""

import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_win_rates(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Catan Agent Win Rates",
) -> str:
    """
    Plot win rates as a grouped bar chart.

    Args:
        results: Dict mapping label -> metrics dict (from compute_metrics)
        output_path: Path to save the plot (PNG)
        title: Plot title

    Returns:
        Path to the saved plot
    """
    labels = list(results.keys())
    win_rates = [results[l]["win_rate"] * 100 for l in labels]
    loss_rates = [results[l]["loss_rate"] * 100 for l in labels]
    draw_rates = [results[l]["draw_rate"] * 100 for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, win_rates, width, label="Win", color="#4CAF50", edgecolor="white")
    bars2 = ax.bar(x, loss_rates, width, label="Loss", color="#F44336", edgecolor="white")
    bars3 = ax.bar(x + width, draw_rates, width, label="Draw", color="#FFC107", edgecolor="white")

    ax.set_ylabel("Rate (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.5,
                    f"{height:.0f}%",
                    ha="center", va="bottom", fontsize=8,
                )

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to: {output_path}")
    else:
        output_path = "results/plots/win_rates.png"
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close()
    return output_path


def plot_learning_curve(
    steps: List[int],
    values: List[float],
    label: str = "Loss",
    output_path: Optional[str] = None,
    title: str = "Training Learning Curve",
    ylabel: str = "Value",
) -> str:
    """
    Plot a learning curve (loss, reward, etc.) over training steps.

    Args:
        steps: List of step numbers
        values: List of metric values
        label: Metric label
        output_path: Path to save plot
        title: Plot title
        ylabel: Y-axis label

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, values, label=label, color="#2196F3", linewidth=2)
    ax.fill_between(steps, values, alpha=0.1, color="#2196F3")

    # Rolling average (window = 10)
    if len(values) > 10:
        window = min(10, len(values) // 5)
        rolling = np.convolve(values, np.ones(window) / window, mode="valid")
        smooth_steps = steps[window - 1:]
        ax.plot(smooth_steps, rolling, label=f"MA({window})", color="#FF5722", linewidth=1, linestyle="--")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        output_path = "results/plots/learning_curve.png"
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close()
    return output_path
