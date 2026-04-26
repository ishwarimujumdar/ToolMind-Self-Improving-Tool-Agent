#!/usr/bin/env python3
"""
Generate impactful charts from HF Jobs training logs for ToolMind presentation.

Extracts reward trajectories, entropy curves, scenario-level comparisons, and
training dynamics from hf_job_full_logs.txt.

Run:  python submission/generate_story_charts.py
Outputs: submission/charts/

Requires: matplotlib, numpy
"""
from __future__ import annotations

import re
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "hf_job_full_logs.txt"
OUT_DIR = ROOT / "submission" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

ACCENT_1 = "#ff6b6b"
ACCENT_2 = "#feca57"
ACCENT_3 = "#48dbfb"
ACCENT_4 = "#2d8f6f"


def parse_training_metrics(log_text: str) -> tuple[list[dict], list[dict]]:
    """Parse logged training metrics into Round 1 and Round 2 lists."""
    pattern = re.compile(
        r"\{" r"'loss': '([^']*)'.*?"
        r"'rewards/reward_fn/mean': '([^']*)'.*?"
        r"'rewards/reward_fn/std': '([^']*)'.*?"
        r"'reward': '([^']*)'.*?"
        r"'entropy': '([^']*)'.*?"
        r"'epoch': '([^']*)'"
        r"\}"
    )

    all_metrics = []
    for m in pattern.finditer(log_text):
        all_metrics.append({
            "loss": float(m.group(1)),
            "reward_mean": float(m.group(2)),
            "reward_std": float(m.group(3)),
            "reward": float(m.group(4)),
            "entropy": float(m.group(5)),
            "epoch": float(m.group(6)),
        })

    r1_end = next(
        (i for i, x in enumerate(all_metrics) if x["epoch"] >= 3.99 and i > 10),
        len(all_metrics) // 2,
    )
    for i in range(r1_end, len(all_metrics)):
        if all_metrics[i]["epoch"] < 1.0:
            r1_end = i
            break
    else:
        r1_end = len(all_metrics) // 2

    r1 = all_metrics[:r1_end]
    r2 = all_metrics[r1_end:]
    return r1, r2


def parse_eval_scenarios(log_text: str, label: str) -> list[tuple[int, float]]:
    """Parse per-scenario eval results for a given label section."""
    section_pat = re.compile(rf"{re.escape(label)}\s*\(40 scenarios\)", re.IGNORECASE)
    section_match = section_pat.search(log_text)
    if not section_match:
        return []
    start = section_match.end()
    end = log_text.find("avg_reward=", start)
    if end == -1:
        end = start + 5000
    block = log_text[start:end]
    scenario_pat = re.compile(r"Scenario\s+(\d+)\s*\|.*?\|\s*reward=([\d.]+)")
    return [(int(m.group(1)), float(m.group(2))) for m in scenario_pat.finditer(block)]


def chart_1_reward_trajectory(r1: list[dict], r2: list[dict]) -> None:
    """Training reward over optimizer steps — both rounds overlaid with baseline reference."""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    steps_r1 = np.linspace(5, 400, len(r1))
    steps_r2 = np.linspace(5, 400, len(r2))

    ax.plot(steps_r1, [m["reward_mean"] for m in r1],
            color=ACCENT_2, linewidth=1.5, alpha=0.9, label="Round 1 (base prompts)")
    ax.fill_between(steps_r1, [m["reward_mean"] for m in r1], alpha=0.12, color=ACCENT_2)

    ax.plot(steps_r2, [m["reward_mean"] for m in r2],
            color=ACCENT_3, linewidth=1.5, alpha=0.9, label="Round 2 (memory-enriched)")
    ax.fill_between(steps_r2, [m["reward_mean"] for m in r2], alpha=0.12, color=ACCENT_3)

    # Baseline avg reward reference line
    BASELINE_AVG = 0.565
    ax.axhline(BASELINE_AVG, color=ACCENT_1, linewidth=2, linestyle="--", alpha=0.85,
               label=f"Baseline avg reward ({BASELINE_AVG})")

    # Round 1 eval avg reference line
    R1_EVAL_AVG = 0.734
    ax.axhline(R1_EVAL_AVG, color=ACCENT_2, linewidth=1.5, linestyle=":", alpha=0.7,
               label=f"Round 1 eval avg ({R1_EVAL_AVG})")

    # Round 2 eval avg reference line
    R2_EVAL_AVG = 0.768
    ax.axhline(R2_EVAL_AVG, color=ACCENT_4, linewidth=1.5, linestyle=":", alpha=0.7,
               label=f"Round 2 eval avg ({R2_EVAL_AVG})")

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward Trajectory — Round 1 vs Round 2 (with eval baselines)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reward_trajectory.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "reward_trajectory.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  reward_trajectory (.png/.svg)")


def chart_2_entropy_curve(r1: list[dict], r2: list[dict]) -> None:
    """Entropy over training — shows exploration/exploitation trade-off."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    steps_r1 = np.linspace(5, 400, len(r1))
    steps_r2 = np.linspace(5, 400, len(r2))

    ax.plot(steps_r1, [m["entropy"] for m in r1],
            color=ACCENT_2, linewidth=1.3, alpha=0.9, label="Round 1")
    ax.plot(steps_r2, [m["entropy"] for m in r2],
            color=ACCENT_3, linewidth=1.3, alpha=0.9, label="Round 2")

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Entropy")
    ax.set_title("Generation Entropy During Training (exploration vs exploitation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "entropy_curve.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "entropy_curve.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  entropy_curve (.png/.svg)")


def chart_3_scenario_heatmap(
    baseline: list[tuple[int, float]],
    round1: list[tuple[int, float]],
    round2: list[tuple[int, float]],
) -> None:
    """Heatmap of per-scenario rewards across three stages."""
    n = min(len(baseline), len(round1), len(round2), 40)
    if n == 0:
        print("  scenario_heatmap SKIPPED (no eval data)")
        return
    data = np.array([
        [r for _, r in baseline[:n]],
        [r for _, r in round1[:n]],
        [r for _, r in round2[:n]],
    ])
    fig, ax = plt.subplots(figsize=(16, 3), dpi=150)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2], ["Baseline", "Round 1", "Round 2"])
    ax.set_xticks(range(n), [f"S{i+1}" for i in range(n)], fontsize=7, rotation=45)
    ax.set_title("Per-Scenario Reward Heatmap (green = high, red = low)")
    plt.colorbar(im, ax=ax, label="Reward", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scenario_heatmap.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "scenario_heatmap.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  scenario_heatmap (.png/.svg)")


def chart_4_improvement_waterfall(
    baseline: list[tuple[int, float]],
    round1: list[tuple[int, float]],
    round2: list[tuple[int, float]],
) -> None:
    """Show which scenarios improved/regressed between stages."""
    n = min(len(baseline), len(round1), len(round2), 40)
    if n == 0:
        print("  improvement_waterfall SKIPPED")
        return
    r1_deltas = [round1[i][1] - baseline[i][1] for i in range(n)]
    r2_deltas = [round2[i][1] - round1[i][1] for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), dpi=150, sharex=True)
    x = np.arange(n)

    colors_r1 = [ACCENT_4 if d > 0 else ACCENT_1 for d in r1_deltas]
    ax1.bar(x, r1_deltas, color=colors_r1, edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Δ Reward")
    ax1.set_title("Scenario-level Improvement: Baseline → Round 1")

    colors_r2 = [ACCENT_4 if d > 0 else ACCENT_1 for d in r2_deltas]
    ax2.bar(x, r2_deltas, color=colors_r2, edgecolor="white", linewidth=0.4)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Δ Reward")
    ax2.set_xlabel("Scenario")
    ax2.set_title("Scenario-level Improvement: Round 1 → Round 2 (memory gain)")
    ax2.set_xticks(x, [f"S{i+1}" for i in range(n)], fontsize=7, rotation=45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "improvement_waterfall.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "improvement_waterfall.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  improvement_waterfall (.png/.svg)")


def chart_5_difficulty_breakdown(
    baseline: list[tuple[int, float]],
    round1: list[tuple[int, float]],
    round2: list[tuple[int, float]],
) -> None:
    """Group scenarios by complexity (1-tool, 2-tool, 3+tool, abstention) and
    show avg reward per group across stages."""
    if not baseline or not round1 or not round2:
        print("  difficulty_breakdown SKIPPED")
        return

    abstention_ids = {5, 12, 14, 16, 22, 28, 30, 34, 35, 37, 38}
    single_tool_ids = {1, 2, 3, 4, 8, 15, 19, 23, 25, 36}
    two_tool_ids = {6, 7, 10, 13, 17, 18, 32, 40}
    multi_tool_ids = {9, 11, 20, 21, 24, 26, 27, 29, 31, 33, 39}

    def avg_for_ids(data: list[tuple[int, float]], ids: set) -> float:
        vals = [r for sid, r in data if sid in ids]
        return sum(vals) / len(vals) if vals else 0

    categories = ["Single-tool", "Two-tool chain", "3+ tool chain", "Abstention/safety"]
    id_groups = [single_tool_ids, two_tool_ids, multi_tool_ids, abstention_ids]

    bl_avgs = [avg_for_ids(baseline, g) for g in id_groups]
    r1_avgs = [avg_for_ids(round1, g) for g in id_groups]
    r2_avgs = [avg_for_ids(round2, g) for g in id_groups]

    x = np.arange(len(categories))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(x - w, bl_avgs, w, label="Baseline", color=ACCENT_1, edgecolor="white")
    ax.bar(x, r1_avgs, w, label="Round 1", color=ACCENT_2, edgecolor="white")
    ax.bar(x + w, r2_avgs, w, label="Round 2", color=ACCENT_3, edgecolor="white")

    for i, (b, r1, r2) in enumerate(zip(bl_avgs, r1_avgs, r2_avgs)):
        ax.text(i - w, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i, r1 + 0.02, f"{r1:.2f}", ha="center", fontsize=8)
        ax.text(i + w, r2 + 0.02, f"{r2:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x, categories)
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward by Task Complexity — Training Impact per Category")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "difficulty_breakdown.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "difficulty_breakdown.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  difficulty_breakdown (.png/.svg)")


def chart_6_summary_dashboard(avg_b: float, avg_r1: float, avg_r2: float,
                               acc_b: float, acc_r1: float, acc_r2: float) -> None:
    """Single-image executive summary with reward bars + accuracy bars + delta annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

    stages = ["Baseline", "GRPO R1", "GRPO R2\n(+Memory)"]
    colors = [ACCENT_1, ACCENT_2, ACCENT_3]

    bars = ax1.bar(stages, [avg_b, avg_r1, avg_r2], color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, [avg_b, avg_r1, avg_r2]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax1.annotate("", xy=(1, avg_r1 + 0.06), xytext=(0, avg_b + 0.06),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_4, lw=1.5))
    ax1.text(0.5, max(avg_b, avg_r1) + 0.09, f"+{avg_r1 - avg_b:.3f}",
             ha="center", fontsize=9, color=ACCENT_4, fontweight="bold")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Eval Reward")
    ax1.set_ylim(0, 1.05)

    bars2 = ax2.bar(stages, [acc_b * 100, acc_r1 * 100, acc_r2 * 100],
                    color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars2, [acc_b, acc_r1, acc_r2]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 100 + 1.5,
                 f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Eval Accuracy")
    ax2.set_ylim(0, 100)

    fig.suptitle("ToolMind — Qwen2.5-3B · Two-Round GRPO · 40-Scenario Held-Out Eval",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_dashboard.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "summary_dashboard.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  summary_dashboard (.png/.svg)")


def main() -> None:
    print("Reading logs...")
    log_text = LOG_FILE.read_text()

    print("Parsing training metrics...")
    r1, r2 = parse_training_metrics(log_text)
    print(f"  Found {len(r1)} R1 metric entries, {len(r2)} R2 entries")

    print("Parsing eval results...")
    baseline = parse_eval_scenarios(log_text, "BASELINE")
    round1_eval = parse_eval_scenarios(log_text, "ROUND 1")
    round2_eval = parse_eval_scenarios(log_text, "ROUND 2")
    print(f"  Baseline: {len(baseline)}, R1: {len(round1_eval)}, R2: {len(round2_eval)} scenarios")

    print("\nGenerating charts:")
    chart_1_reward_trajectory(r1, r2)
    chart_2_entropy_curve(r1, r2)
    chart_3_scenario_heatmap(baseline, round1_eval, round2_eval)
    chart_4_improvement_waterfall(baseline, round1_eval, round2_eval)
    chart_5_difficulty_breakdown(baseline, round1_eval, round2_eval)
    chart_6_summary_dashboard(0.565, 0.734, 0.768, 0.475, 0.700, 0.725)

    print(f"\nAll charts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
