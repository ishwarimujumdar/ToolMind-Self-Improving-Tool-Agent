"""
Generate static figures (PNG/SVG) for blog / README — ToolMind eval summary.

Run: python submission/generate_submission_charts.py
Outputs: submission/charts/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Eval on 40 held-out scenarios (HF Jobs full run, Qwen2.5-3B, bf16 LoRA)
STAGES = ("Baseline", "GRPO round 1", "GRPO round 2")
AVG_REWARD = (0.565, 0.734, 0.768)
ACC_PCT = (47.5, 70.0, 72.5)

OUT_DIR = Path(__file__).resolve().parent / "charts"
COL_REWARD = ("#4a6fa5", "#2d8f6f", "#1a5c4a")
COL_ACC = ("#6b4a8f", "#8f2d6b", "#5c1a3d")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def fig_eval_summary() -> plt.Figure:
    _set_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=120)
    x = np.arange(len(STAGES))
    w = 0.6

    bars0 = ax0.bar(x, AVG_REWARD, width=w, color=COL_REWARD, edgecolor="white", linewidth=0.8)
    ax0.set_xticks(x, STAGES, rotation=15, ha="right")
    ax0.set_ylabel("Average reward (eval)")
    ax0.set_ylim(0, 1.0)
    ax0.set_title("ToolMind — 40-scenario eval")
    for b, v in zip(bars0, AVG_REWARD):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    bars1 = ax1.bar(x, ACC_PCT, width=w, color=COL_ACC, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x, STAGES, rotation=15, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Same held-out eval split")
    for b, v in zip(bars1, ACC_PCT):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Qwen2.5-3B · GRPO (R1) → memory → GRPO (R2)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def fig_delta_over_baseline() -> plt.Figure:
    _set_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)
    labels = ("Round 1 vs\nbaseline", "Round 2 vs\nbaseline", "Round 2 vs\nround 1")
    values = (0.169, 0.203, 0.034)
    colors = ("#2d8f6f", "#1a5c4a", "#6b4a8f")
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Δ average reward")
    ax.set_title("Marginal lift on the same 40 eval scenarios")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"+{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1 = fig_eval_summary()
    f1.savefig(OUT_DIR / "eval_summary.png", bbox_inches="tight", facecolor="white")
    f1.savefig(OUT_DIR / "eval_summary.svg", bbox_inches="tight", facecolor="white")
    plt.close(f1)

    f2 = fig_delta_over_baseline()
    f2.savefig(OUT_DIR / "reward_deltas.png", bbox_inches="tight", facecolor="white")
    f2.savefig(OUT_DIR / "reward_deltas.svg", bbox_inches="tight", facecolor="white")
    plt.close(f2)

    print(f"Wrote PNG + SVG to {OUT_DIR}")


if __name__ == "__main__":
    main()
