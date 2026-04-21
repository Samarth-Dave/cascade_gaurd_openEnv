"""
plot_curves.py — Generate reward improvement curves from training logs
========================================================================
Reads reward_log.csv produced by grpo_train.py and generates
publication-quality matplotlib figures for the HuggingFace blog.

Usage:
    python training/plot_curves.py --log training/reward_log.csv --out figs/

Output:
    figs/reward_curve.png   — main training progress plot
    figs/per_task.png       — per-task score breakdown (if curriculum data present)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


# ── Data loading ───────────────────────────────────────────────────────────────

def load_log(path: str) -> List[dict]:
    """Read reward_log.csv into a list of row dicts."""
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step":             int(row["step"]),
                "task_id":          row["task_id"],
                "mean_score":       float(row["mean_score"]),
                "min_score":        float(row["min_score"]),
                "max_score":        float(row["max_score"]),
                "curriculum_stage": int(row.get("curriculum_stage", 0)),
            })
    return rows


# ── Rolling average ────────────────────────────────────────────────────────────

def _rolling(values: List[float], window: int = 10) -> List[float]:
    result = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


# ── Main reward curve plot ─────────────────────────────────────────────────────

def plot_reward_curve(rows: List[dict], output_dir: str = "figs") -> None:
    """
    Generate the main training progress plot with:
    - Shaded min/max score band
    - Raw per-step mean (faint)
    - 10-step rolling mean (prominent)
    - Curriculum stage background shading
    - Horizontal baseline reference line
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. pip install matplotlib")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps  = [r["step"]       for r in rows]
    scores = [r["mean_score"] for r in rows]
    mins   = [r["min_score"]  for r in rows]
    maxs   = [r["max_score"]  for r in rows]
    smoothed = _rolling(scores, window=10)

    # Detect curriculum stage boundaries
    stage_changes: List[Tuple[int, str]] = []
    for i, r in enumerate(rows):
        if i == 0 or r["task_id"] != rows[i - 1]["task_id"]:
            stage_changes.append((r["step"], r["task_id"]))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    # Curriculum stage shading
    stage_colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#FCE4EC", "#EDE7F6"]
    for i, (start_step, task_id) in enumerate(stage_changes):
        end_step = (
            stage_changes[i + 1][0] if i + 1 < len(stage_changes) else max(steps) + 1
        )
        ax.axvspan(
            start_step, end_step,
            alpha=0.5,
            color=stage_colors[i % len(stage_colors)],
            zorder=0,
        )
        # Label the stage at the top of the band
        ax.text(
            (start_step + end_step) / 2, 0.965,
            task_id.replace("task_", ""),
            ha="center", va="top", fontsize=7.5, color="#555",
            transform=ax.get_xaxis_transform(),
        )

    # Score band (min–max)
    ax.fill_between(steps, mins, maxs, alpha=0.12, color="#1976D2", zorder=1,
                    label="Score range (min–max)")

    # Raw per-step mean (faint)
    ax.plot(steps, scores, color="#90CAF9", alpha=0.35, linewidth=0.9,
            zorder=2, label="Per-step mean score")

    # Smoothed 10-step rolling mean
    ax.plot(steps, smoothed, color="#1565C0", linewidth=2.2, zorder=3,
            label="10-step rolling mean")

    # Baseline reference
    ax.axhline(y=0.46, color="#E53935", linestyle="--", alpha=0.7, linewidth=1.2,
               label="Random policy baseline (~0.46)", zorder=4)

    # Decoration
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Grader score", fontsize=12)
    ax.set_title(
        "CascadeGuard — GRPO Training Progress",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0, max(steps) + 1)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = Path(output_dir) / "reward_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Saved: {out_path}")
    plt.close(fig)


# ── Per-task breakdown plot ────────────────────────────────────────────────────

def plot_per_task(rows: List[dict], output_dir: str = "figs") -> None:
    """
    Bar chart: final 20-step mean score per task.
    Useful for showing curriculum progression in the blog.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from collections import defaultdict

    task_scores: Dict[str, List[float]] = defaultdict(list)
    for r in rows[-min(len(rows), 100):]:  # last 100 steps
        task_scores[r["task_id"]].append(r["mean_score"])

    tasks = list(task_scores.keys())
    means = [sum(task_scores[t]) / len(task_scores[t]) for t in tasks]

    bar_colors = ["#1565C0", "#2E7D32", "#F57F17", "#B71C1C", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(tasks, means, color=bar_colors[: len(tasks)], edgecolor="white",
                  linewidth=1.2, zorder=2)
    ax.axhline(y=0.46, color="#E53935", linestyle="--", alpha=0.8, linewidth=1.2,
               label="Random baseline (~0.46)")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.015,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylabel("Mean grader score", fontsize=11)
    ax.set_title("CascadeGuard — Score per Task (Trained Policy)", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=18, ha="right", fontsize=9)

    out_path = Path(output_dir) / "per_task.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Saved: {out_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CascadeGuard reward curve visualiser"
    )
    parser.add_argument(
        "--log", default="training/reward_log.csv",
        help="Path to reward_log.csv from grpo_train.py"
    )
    parser.add_argument(
        "--out", default="figs",
        help="Output directory for PNG figures"
    )
    args = parser.parse_args()

    rows = load_log(args.log)
    print(f"[✓] Loaded {len(rows)} training steps from {args.log}")

    plot_reward_curve(rows, args.out)
    plot_per_task(rows, args.out)
    print(f"[✓] All figures saved to {args.out}/")
