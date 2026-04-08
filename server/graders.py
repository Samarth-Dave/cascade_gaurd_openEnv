"""
graders.py — CascadeGuard episode scoring
==========================================
INVARIANT enforced at every exit point:
    Every public function returns a float strictly in (SCORE_EPS, 1.0 - SCORE_EPS).
    No function may return exactly 0.0 or 1.0 under any input — including NaN,
    ±inf, empty lists, all-zero lists, division by zero, or any exception path.

Design contract
---------------
* _clamp(v)          — first line of defence; maps any float → [SCORE_EPS, 1-SCORE_EPS]
* _safe(v)           — final gate; round-FIRST then boundary-check (never round-after-check)
* Every sub-scorer   — returns via _clamp so its output is already safe
* grade()            — wraps the dispatch in a broad try/except; applies _safe() twice
                       (once on the raw grader result, once at the very end)
"""

from __future__ import annotations

import math
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORE_EPS: float = 1e-4          # smallest legal score  (exclusive lower bound is 0)
_SCORE_HI: float = 1.0 - SCORE_EPS   # largest  legal score  (exclusive upper bound is 1)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _clamp(value: float,
           lo: float = SCORE_EPS,
           hi: float = _SCORE_HI) -> float:
    """
    Map any finite float → [lo, hi].
    Handles NaN, ±inf, non-numeric inputs.
    Default range: [SCORE_EPS, 1-SCORE_EPS].
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))


def _safe(value: float) -> float:
    
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SCORE_EPS
    if not math.isfinite(v):
        return SCORE_EPS


    # ---- THEN check boundary -----------------------------------------------
    if v <= 0.0:
        return SCORE_EPS
    if v >= 1.0:
        return _SCORE_HI

    return v


# ---------------------------------------------------------------------------
# Sub-scorers  (each returns via _clamp → always in [SCORE_EPS, 1-SCORE_EPS])
# ---------------------------------------------------------------------------

def _hospital_maintained_score(hospital_health_log: List[float]) -> float:
    """
    Fraction of steps where hospital health stayed ≥ 0.4.
    Empty log → perfect score (no data, no penalty).
    All steps violated → SCORE_EPS (worst, never 0.0).
    """
    if not hospital_health_log:
        return _SCORE_HI

    total_steps = len(hospital_health_log)
    violations = sum(1 for h in hospital_health_log
                     if not math.isfinite(float(h) if isinstance(h, (int, float)) else float('nan'))
                     or float(h) < 0.4)
    if violations == 0:
        return _SCORE_HI

    ratio = violations / max(total_steps, 1)
    return _clamp(1.0 - ratio)


def _cascade_contained_score(failure_history: List[Set[str]], total_nodes: int) -> float:
    """
    1 minus (worst single-step failure fraction).
    Empty history → perfect. All nodes failed → SCORE_EPS.
    """
    if not failure_history:
        return _SCORE_HI

    all_seen: Set[str] = set()
    for s in failure_history:
        all_seen.update(s)

    denom = max(total_nodes or len(all_seen), 1)
    worst_fail_ratio = max(len(s) for s in failure_history) / denom
    return _clamp(1.0 - worst_fail_ratio)


def _dependency_order_score(dependency_order_log: List[int]) -> float:
    """
    Fraction of recoveries performed in correct dependency order.
    Empty log → neutral 0.5 (no data, no opinion).
    """
    if not dependency_order_log:
        return _safe(0.5)

    return _clamp(sum(dependency_order_log) / max(len(dependency_order_log), 1))


def _cascade_depth_score(cascade_depth_log: List[int], total_nodes: int) -> float:
    """
    1 minus (average cascade depth ratio per step).
    Empty log → perfect. Depth exceeds total_nodes → SCORE_EPS.
    """
    if not cascade_depth_log:
        return _SCORE_HI

    denom = max(total_nodes, 1)
    depth_ratio = sum(cascade_depth_log) / (len(cascade_depth_log) * denom)
    return _clamp(1.0 - depth_ratio)


def _intervention_efficiency_score(action_history: List[str],
                                   budget_spent: float,
                                   budget_total: float) -> float:
    """
    Blend of proactive-action rate and budget efficiency.
    Empty history → neutral 0.5.
    """
    if not action_history:
        return _safe(0.5)

    active_actions = sum(1 for a in action_history if a != "wait")
    proactive_rate = active_actions / max(len(action_history), 1)   # in [0, 1]

    budget_eff = _clamp(1.0 - (budget_spent / max(float(budget_total), 1e-9)))

    # proactive_rate is already in [0,1]; budget_eff in [SCORE_EPS, 1-SCORE_EPS]
    return _clamp(0.6 * proactive_rate + 0.4 * budget_eff)


# ---------------------------------------------------------------------------
# Per-task graders
# ---------------------------------------------------------------------------

def grade_easy(
    failure_history: List[Set[str]],
    hospital_health_log: List[float],
    final_sector_summary: Dict[str, float],
    budget_spent: float,
    budget_total: float,
    total_nodes: int = 0,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
) -> float:
    """
    task_easy grader.
    Weights: avg_health=0.35, no_blackout=0.35, cascade=0.20, depth=0.10   (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg = _clamp(
        sum(final_sector_summary.values()) / len(final_sector_summary)
    )
    no_blackout = _clamp(
        1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    )
    cascade = _cascade_contained_score(failure_history, total_nodes)
    depth   = _cascade_depth_score(cascade_depth_log or [], total_nodes)

    raw = 0.35 * avg + 0.35 * no_blackout + 0.20 * cascade + 0.10 * depth
    return _safe(raw)


def grade_medium(
    failure_history: List[Set[str]],
    hospital_health_log: List[float],
    final_sector_summary: Dict[str, float],
    budget_spent: float,
    budget_total: float,
    total_nodes: int = 0,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
) -> float:
    """
    task_medium grader.
    Weights: avg=0.25, hosp=0.30, no_blackout=0.20, cascade=0.10, depth=0.10, dep_order=0.05   (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg       = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp      = _hospital_maintained_score(hospital_health_log)
    no_blackout = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade   = _cascade_contained_score(failure_history, total_nodes)
    depth     = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order = _dependency_order_score(dependency_order_log or [])

    raw = (0.25 * avg
           + 0.30 * hosp
           + 0.20 * no_blackout
           + 0.10 * cascade
           + 0.10 * depth
           + 0.05 * dep_order)
    return _safe(raw)


def grade_hard(
    failure_history: List[Set[str]],
    hospital_health_log: List[float],
    final_sector_summary: Dict[str, float],
    budget_spent: float,
    budget_total: float,
    total_nodes: int = 0,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
) -> float:
    """
    task_hard grader.
    Weights: avg=0.20, hosp=0.25, no_blackout=0.15, cascade=0.10,
             budget_eff=0.10, depth=0.10, dep_order=0.05, intervention=0.05   (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg              = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp             = _hospital_maintained_score(hospital_health_log)
    no_blackout      = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade          = _cascade_contained_score(failure_history, total_nodes)
    budget_efficiency = _clamp(1.0 - (budget_spent / max(float(budget_total), 1e-9)))
    depth            = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order        = _dependency_order_score(dependency_order_log or [])
    intervention     = _intervention_efficiency_score(action_history or [], budget_spent, budget_total)

    raw = (0.20 * avg
           + 0.25 * hosp
           + 0.15 * no_blackout
           + 0.10 * cascade
           + 0.10 * budget_efficiency
           + 0.10 * depth
           + 0.05 * dep_order
           + 0.05 * intervention)
    return _safe(raw)


def grade_gen_blackout(
    failure_history: List[Set[str]],
    hospital_health_log: List[float],
    final_sector_summary: Dict[str, float],
    budget_spent: float,
    budget_total: float,
    total_nodes: int = 0,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
) -> float:
    """
    task_gen_blackout grader.
    Root-generator cascade to hospitals — high hospital weight.
    Weights: avg=0.20, hosp=0.35, no_blackout=0.20, cascade=0.10, depth=0.10, dep_order=0.05  (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg         = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp        = _hospital_maintained_score(hospital_health_log)
    no_blackout = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade     = _cascade_contained_score(failure_history, total_nodes)
    depth       = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order   = _dependency_order_score(dependency_order_log or [])

    raw = (0.20 * avg
           + 0.35 * hosp
           + 0.20 * no_blackout
           + 0.10 * cascade
           + 0.10 * depth
           + 0.05 * dep_order)
    return _safe(raw)


def grade_cyberattack(
    failure_history: List[Set[str]],
    hospital_health_log: List[float],
    final_sector_summary: Dict[str, float],
    budget_spent: float,
    budget_total: float,
    total_nodes: int = 0,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
) -> float:
    """
    task_cyberattack grader.
    Sustained SCADA + compound faults + storm — hardest scenario.
    Weights: avg=0.18, hosp=0.28, no_blackout=0.16, cascade=0.10, budget_eff=0.10,
             depth=0.08, dep_order=0.05, intervention=0.05   (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg              = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp             = _hospital_maintained_score(hospital_health_log)
    no_blackout      = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade          = _cascade_contained_score(failure_history, total_nodes)
    budget_efficiency = _clamp(1.0 - (budget_spent / max(float(budget_total), 1e-9)))
    depth            = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order        = _dependency_order_score(dependency_order_log or [])
    intervention     = _intervention_efficiency_score(action_history or [], budget_spent, budget_total)

    raw = (0.18 * avg
           + 0.28 * hosp
           + 0.16 * no_blackout
           + 0.10 * cascade
           + 0.10 * budget_efficiency
           + 0.08 * depth
           + 0.05 * dep_order
           + 0.05 * intervention)
    return _safe(raw)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GRADERS = {
    "task_easy":         grade_easy,
    "task_medium":       grade_medium,
    "task_hard":         grade_hard,
    "task_gen_blackout": grade_gen_blackout,
    "task_cyberattack":  grade_cyberattack,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(task_id: str, **kwargs) -> float:
    """
    Dispatch to the appropriate grader and guarantee the result is in (0, 1).

    Two-layer defence:
    1. The individual grader already returns via _safe() → in (0,1).
    2. grade() applies _safe() once more on the final value so that any
       future change to a grader cannot accidentally leak 0.0 or 1.0.

    Exception handling: catch ALL exceptions (not just TypeError/ValueError)
    so that an unexpected crash in a grader never propagates to the pipeline.
    """
    if task_id not in GRADERS:
        # Unknown task — return neutral rather than crashing.
        return _safe(0.5)

    try:
        raw = GRADERS[task_id](**kwargs)
    except Exception:
        # Any grader crash → neutral fallback, never 0.0 or 1.0.
        return _safe(0.5)

    # Belt-and-suspenders: apply _safe even though graders already do so.
    return _safe(raw)