"""
cli_display.py — Clean terminal output for CascadeGuard demo mode
=================================================================
Renders step-by-step environment state in a readable, judge-friendly
format using only built-in terminal characters (no external deps).

Usage:
    from cli_display import render_step, render_episode_start, render_episode_end

    render_episode_start(task_id, seed)
    for each step:
        render_step(obs, action_type, target, reward, reasoning, metadata)
    render_episode_end(score, grader_breakdown)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional


# ── ANSI colour codes (safe on most terminals) ─────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_BLUE   = "\033[94m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"
_ORANGE = "\033[38;5;208m"

# Disable colour if not a tty (e.g. log files, CI)
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


# ── Health bar ──────────────────────────────────────────────────────────────────

def _health_bar(value: float, width: int = 10) -> str:
    """Render a Unicode block bar for a 0.0–1.0 health value."""
    filled = round(value * width)
    empty  = width - filled
    bar    = "█" * filled + "░" * empty
    if value >= 0.7:
        colour = _GREEN
    elif value >= 0.4:
        colour = _YELLOW
    else:
        colour = _RED
    return _c(colour, bar)


def _health_str(value: float) -> str:
    s = f"{value:.2f}"
    if value >= 0.7:
        return _c(_GREEN, s)
    elif value >= 0.4:
        return _c(_YELLOW, s)
    else:
        return _c(_RED, s)


# ── Narrative phase banner ──────────────────────────────────────────────────────

PHASE_COLOURS = {
    "pre_storm":     _BLUE,
    "storm_onset":   _YELLOW,
    "crisis_peak":   _RED,
    "recovery":      _CYAN,
    "stabilization": _GREEN,
}

def _phase_banner(phase: str) -> str:
    icons = {
        "pre_storm":     "🌤",
        "storm_onset":   "⚡",
        "crisis_peak":   "🚨",
        "recovery":      "🔧",
        "stabilization": "✅",
    }
    icon   = icons.get(phase, "•")
    colour = PHASE_COLOURS.get(phase, _WHITE)
    label  = phase.replace("_", " ").upper()
    return _c(colour, f"[{icon}  {label}]")


# ── Weather indicator ───────────────────────────────────────────────────────────

def _weather_str(weather: str) -> str:
    mapping = {
        "clear":        _c(_GREEN, "☀  CLEAR"),
        "storm_warning": _c(_YELLOW, "⛈  STORM WARNING"),
        "extreme_storm": _c(_RED, "🌪  EXTREME STORM"),
    }
    return mapping.get(weather, weather)


# ── Budget bar ──────────────────────────────────────────────────────────────────

def _budget_bar(remaining: float, total: float, width: int = 12) -> str:
    ratio  = remaining / max(total, 0.01)
    filled = round(ratio * width)
    bar    = "▰" * filled + "▱" * (width - filled)
    colour = _GREEN if ratio > 0.5 else (_YELLOW if ratio > 0.25 else _RED)
    return _c(colour, bar) + f" {remaining:.1f}/{total:.0f}"


# ── Action display ──────────────────────────────────────────────────────────────

ACTION_ICONS = {
    "harden":     "🛡",
    "recover":    "♻",
    "shed_load":  "↓",
    "coordinate": "📡",
    "wait":       "⏸",
}

def _action_str(action_type: str, target: Optional[str], valid: bool) -> str:
    icon = ACTION_ICONS.get(action_type, "?")
    tgt  = f"({target})" if target else "()"
    text = f"{icon}  {action_type.upper()}{tgt}"
    if not valid:
        return _c(_RED, text + "  ✗ INVALID")
    if action_type == "wait":
        return _c(_DIM, text)
    return _c(_BOLD, _c(_CYAN, text))


# ── Main step renderer ──────────────────────────────────────────────────────────

def render_episode_start(task_id: str, seed: int, description: str = "") -> None:
    """Print episode header."""
    print()
    print(_c(_BOLD, "=" * 65))
    print(_c(_BOLD, f"  CascadeGuard  —  {task_id}  (seed={seed})"))
    if description:
        # Wrap description to 60 chars
        words = description.split()
        line  = "  "
        for w in words:
            if len(line) + len(w) > 63:
                print(_c(_DIM, line))
                line = "  " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(_c(_DIM, line))
    print(_c(_BOLD, "=" * 65))


def render_step(
    obs,
    action_type: str,
    target: Optional[str],
    reward: float,
    reasoning: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    budget_total: float = 10.0,
) -> None:
    """
    Render one step of the episode to stdout.

    Keeps output ≤ 35 lines per step (judge-friendly).
    """
    metadata = metadata or {}
    action_valid   = metadata.get("action_valid", True)
    invalid_reason = metadata.get("invalid_reason", "")
    narrative      = getattr(obs, "narrative_phase", "pre_storm")
    consec_waits   = getattr(obs, "consecutive_waits", 0)
    coverage_evts  = getattr(obs, "radius_coverage_events", [])

    # ── Step header ─────────────────────────────────────────────────
    print()
    print(
        f"{_phase_banner(narrative)}  "
        f"{_c(_BOLD, f'Step {obs.step}/{obs.max_steps}')}  "
        f"{_weather_str(obs.weather_forecast)}"
    )
    print(
        f"  Budget: {_budget_bar(obs.budget_remaining, budget_total)}  "
        f"Reward: {_c(_GREEN if reward > 0 else _RED, f'{reward:+.3f}')}"
    )

    # ── Action taken ─────────────────────────────────────────────────
    print(f"  Action: {_action_str(action_type, target, action_valid)}", end="")
    if not action_valid and invalid_reason:
        print(f"  [{_c(_YELLOW, invalid_reason)}]")
    else:
        print()

    # ── LLM reasoning (truncated) ─────────────────────────────────────
    if reasoning:
        snippet = reasoning[:120].replace("\n", " ").strip()
        if len(reasoning) > 120:
            snippet += "…"
        print(f"  {_c(_DIM, f'Think: {snippet}')}")

    # ── Consecutive waits warning ─────────────────────────────────────
    if consec_waits >= 3:
        print(f"  {_c(_ORANGE, f'⚠  {consec_waits} consecutive waits — pressure escalating!')}")

    # ── Node status table (condensed) ─────────────────────────────────
    print(f"\n  {'Node':<22} {'Sector':<9} {'Health':>8}  {'Load':>5}  {'Status':<12}")
    print(f"  {'─'*22} {'─'*9} {'─'*8}  {'─'*5}  {'─'*12}")
    for n in sorted(obs.nodes, key=lambda x: x.node_id):
        status = "OK"
        if not n.is_operational:
            status = _c(_RED, "FAILED")
        elif n.is_hardened:
            status = _c(_GREEN, "hardened")
        elif n.observation_delayed:
            status = _c(_YELLOW, "delayed")

        conf_str = ""
        if hasattr(n, "observation_confidence") and n.observation_confidence < 0.8:
            conf_str = _c(_DIM, f" conf={n.observation_confidence:.2f}")

        print(
            f"  {n.node_id:<22} {n.sector:<9} "
            f"{_health_bar(n.health, 6)} {_health_str(n.health)}  "
            f"{n.load:.2f}  {status}{conf_str}"
        )

    # ── Active failures / pending recoveries ──────────────────────────
    if obs.active_failures:
        print(f"\n  {_c(_RED, '✗ Failures:')} {', '.join(obs.active_failures)}")
    if obs.pending_recoveries:
        print(f"  {_c(_CYAN, '♻ Recovering:')} {', '.join(obs.pending_recoveries)}")

    # ── Radius coverage events ─────────────────────────────────────────
    if coverage_evts:
        print(f"  {_c(_BLUE, '📡 Radius coverage:')}")
        for evt in coverage_evts[:3]:
            print(f"     {_c(_DIM, evt)}")

    # ── Sector health summary ──────────────────────────────────────────
    print()
    sectors = sorted(obs.sector_summary.items())
    sec_str = "  "
    for sector, h in sectors:
        sec_str += f"{sector[:6]}: {_health_bar(h, 5)} {h:.2f}   "
    print(sec_str)


def render_episode_end(
    score: float,
    steps: int,
    grader_breakdown: Optional[Dict[str, float]] = None,
    action_counts: Optional[Dict[str, int]] = None,
) -> None:
    """Print final score summary."""
    print()
    print(_c(_BOLD, "─" * 65))
    colour = _GREEN if score >= 0.70 else (_YELLOW if score >= 0.55 else _RED)
    print(_c(_BOLD, f"  EPISODE COMPLETE  —  Score: {_c(colour, f'{score:.4f}')}  ({steps} steps)"))

    if grader_breakdown:
        print(f"\n  {_c(_BOLD, 'Grader Components:')}")
        for k, v in grader_breakdown.items():
            bar = _health_bar(v, 8)
            print(f"    {k:<30} {bar}  {v:.3f}")

    if action_counts:
        total = max(sum(action_counts.values()), 1)
        print(f"\n  {_c(_BOLD, 'Action Distribution:')}")
        for a, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            icon = ACTION_ICONS.get(a, "?")
            print(f"    {icon}  {a:<14} {cnt:>3}x  ({pct:.0f}%)")

    print(_c(_BOLD, "─" * 65))
    print()
