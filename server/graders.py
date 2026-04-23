"""
graders.py — CascadeGuard episode scoring
==========================================
INVARIANT enforced at every exit point:
    Every public function returns a float strictly in (0.01, 0.99).
    No function may return exactly 0.0 or 1.0 under any input — including NaN,
    ±inf, empty lists, all-zero lists, division by zero, or any exception path.

Design contract
---------------
* _clamp(v)          — first line of defence; maps any float → [0.01, 0.99]
* _safe(v)           — final gate; round-FIRST then boundary-check (never round-after-check)
* Every sub-scorer   — returns via _clamp so its output is already safe
* grade()            — wraps the dispatch in a broad try/except; applies _safe() twice
                       (once on the raw grader result, once at the very end)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORE_EPS: float = 0.01          # smallest legal score  (exclusive lower bound is 0)
_SCORE_HI: float = 0.99          # largest  legal score  (exclusive upper bound is 1)


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
    """
    CRITICAL: ROUND FIRST → THEN CLAMP
    This order is essential to prevent float precision edge cases.
    Ex: 0.99996 would round to 1.0000, escape boundary check if checked first.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SCORE_EPS
    if not math.isfinite(v):
        return SCORE_EPS

    # ROUND FIRST (to 4 decimals)
    v = round(v, 4)

    # THEN CLAMP to safe boundaries
    return min(max(v, SCORE_EPS), _SCORE_HI)


# ---------------------------------------------------------------------------
# Sub-scorers  (each returns via _clamp → always in [SCORE_EPS, 1-SCORE_EPS])
# ---------------------------------------------------------------------------

def _hospital_maintained_score(hospital_health_log: List[float]) -> float:
    """
    Fraction of steps where hospital health stayed ≥ 0.4.
    Empty log → neutral 0.5 (no data, no opinion — agent neither helped nor hurt).
    All steps violated → SCORE_EPS (worst, never 0.0).

    NOTE: Changed from returning _SCORE_HI on empty log. A passive agent that
    never acts produces an empty hospital log — we should not reward that with
    a perfect score. Neutral 0.5 is correct here.
    """
    if not hospital_health_log:
        return _safe(0.5)  # neutral: no data means we don't know — don't reward

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
    avg_fail_ratio = sum(len(s) for s in failure_history) / (len(failure_history) * denom)
    combined = 0.6 * worst_fail_ratio + 0.4 * avg_fail_ratio
    return _clamp(1.0 - combined)


def _dependency_order_score(dependency_order_log: List[int]) -> float:
    """
    Fraction of recoveries performed in correct dependency order.
    Empty log → slightly-below-neutral 0.35 (agent did zero recoveries — no credit).
    """
    if not dependency_order_log:
        return _safe(0.35)  # no recoveries attempted → below neutral, not penalised fully

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


def _proactive_rate_score(action_history: List[str]) -> float:
    """
    Fraction of steps where the agent took a non-wait action.
    Empty history → 0.0 (worst — no actions taken at all means pure passivity).
    100% wait → SCORE_EPS (near zero — pure passivity is the floor).

    NOTE: Changed from neutral 0.5 on empty. A 0% active rate should score 0,
    not 0.5. This is the primary lever that breaks the wait-attractor floor.
    """
    if not action_history:
        return SCORE_EPS  # no history at all → worst possible proactive score

    active_actions = sum(1 for a in action_history if a != "wait")
    proactive_rate = active_actions / max(len(action_history), 1)   # in [0, 1]

    return _clamp(proactive_rate)


def _budget_conservation_score(budget_spent: float, budget_total: float) -> float:
    """Allow prudent spending; penalize only heavy spending above 70%."""
    if budget_total <= 0:
        return _safe(0.5)
    spend_ratio = budget_spent / budget_total
    if spend_ratio <= 0.70:
        return _SCORE_HI
    return _clamp(1.0 - (spend_ratio - 0.70) * 2.0)


def _budget_efficiency_score(budget_spent: float, budget_total: float) -> float:
    """
    Reward useful budget usage to break the zero-spend fixed point.

    Full credit is reached at 50% spend; 0 spend gets SCORE_EPS.
    """
    if budget_total <= 0:
        return SCORE_EPS
    if budget_spent <= 0:
        return SCORE_EPS
    ratio = min(float(budget_spent) / (0.5 * float(budget_total)), 1.0)
    return _clamp(ratio)


def _centrality_protection_score(
    action_history: List[str],
    high_centrality_nodes: List[str],
    failure_history: List[Set[str]],
) -> float:
    """
    Rewards the agent for protecting high-centrality (backbone) nodes.
    Penalises if any high-centrality node appeared in the failure history.

    high_centrality_nodes: list of node IDs with degree >= 3 (from env).
    Returns _safe(0.5) when data is unavailable (neutral, never penalises).
    """
    if not high_centrality_nodes or not failure_history:
        return _safe(0.5)

    all_failed: Set[str] = set()
    for s in failure_history:
        all_failed.update(s)

    central_failed = sum(1 for n in high_centrality_nodes if n in all_failed)
    protected_ratio = 1.0 - (central_failed / max(len(high_centrality_nodes), 1))
    return _clamp(protected_ratio)


def _sacrifice_penalty(sacrifice_count: int) -> float:
    """
    Grader-side penalty for controlled_cascade actions.
    Each sacrifice node recorded in episode metadata reduces the health
    component by 0.1, capped at 0.5 total reduction.
    Returns a non-negative deduction (caller subtracts it).
    """
    return min(0.5, 0.1 * max(0, sacrifice_count))


def _critical_failure_penalty(total_critical_failures: int) -> float:
    """
    Grader-side penalty for accumulated critical failures across the episode.
    Each critical failure reduces the raw composite by 0.04, unbounded below.
    Returns a non-negative deduction value (caller subtracts it).

    This directly links CF accumulation (which step rewards penalise) to the
    final grader score, eliminating the reward-score decoupling that caused
    GRPO to optimise the wrong objective.

    Example: 13 CFs (gen_blackout worst case) → -0.52 deduction.
    A wait-only policy on gen_blackout (baseline 0.5 raw) → final ~0.01.
    """
    return 0.04 * max(0, total_critical_failures)


def _cascade_contained_score_lockdown(
    failure_history: List[Set[str]],
    total_nodes: int,
    lockdown_steps: List[int],
) -> float:
    """
    Variant of _cascade_contained_score that excludes lockdown freeze-window
    steps from the measurement so the agent is not penalised for cascade
    propagation that was deliberately paused.
    """
    if not failure_history:
        return _SCORE_HI

    skip = set(lockdown_steps)
    filtered = [
        s for i, s in enumerate(failure_history) if i not in skip
    ]
    if not filtered:
        return _SCORE_HI

    all_seen: Set[str] = set()
    for s in filtered:
        all_seen.update(s)

    denom = max(total_nodes or len(all_seen), 1)
    worst_fail_ratio = max(len(s) for s in filtered) / denom
    avg_fail_ratio = sum(len(s) for s in filtered) / (len(filtered) * denom)
    combined = 0.6 * worst_fail_ratio + 0.4 * avg_fail_ratio
    return _clamp(1.0 - combined)


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
    **kwargs,
) -> float:
    """
    task_easy grader.
    Weights: avg_health=0.28, hosp=0.10, no_blackout=0.28, cascade=0.20, depth=0.14   (sum=1.00)
    Proactive-action component added to penalise 100% wait policies.
    """
    if not final_sector_summary:
        return _safe(0.15)  # no data → low score, not neutral 0.5

    avg = _clamp(
        sum(final_sector_summary.values()) / len(final_sector_summary)
    )
    hosp = _hospital_maintained_score(hospital_health_log)
    no_blackout = _clamp(
        1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    )
    cascade = kwargs.get("_adjusted_cascade_score") or _cascade_contained_score(failure_history, total_nodes)
    depth   = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    proactive = _proactive_rate_score(action_history or [])
    budget_efficiency = _budget_efficiency_score(budget_spent, budget_total)

    raw = (
        0.24 * avg
        + 0.10 * hosp
        + 0.20 * no_blackout
        + 0.18 * cascade
        + 0.10 * depth
        + 0.08 * proactive
        + 0.10 * budget_efficiency
    )
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
    **kwargs,
) -> float:
    """
    task_medium grader.
    Weights: avg=0.20, hosp=0.25, no_blackout=0.16, cascade=0.10,
             depth=0.07, dep_order=0.05, proactive=0.09, budget_eff=0.08 (sum=1.00)
    Proactive rate now a first-class component to break the wait-attractor floor.
    """
    if not final_sector_summary:
        return _safe(0.15)  # no data → low score, not neutral 0.5

    avg         = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp        = _hospital_maintained_score(hospital_health_log)
    no_blackout = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade     = kwargs.get("_adjusted_cascade_score") or _cascade_contained_score(failure_history, total_nodes)
    depth       = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order   = _dependency_order_score(dependency_order_log or [])
    proactive   = _proactive_rate_score(action_history or [])
    budget_efficiency = _budget_efficiency_score(budget_spent, budget_total)

    raw = (0.20 * avg
            + 0.25 * hosp
            + 0.16 * no_blackout
            + 0.10 * cascade
            + 0.07 * depth
            + 0.05 * dep_order
            + 0.09 * proactive
            + 0.08 * budget_efficiency)
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
    high_centrality_nodes: List[str] | None = None,   # P2: centrality-aware bonus
    **kwargs,
) -> float:
    """
    task_hard grader.
    Weights: avg=0.20, hosp=0.25, no_blackout=0.15, cascade=0.10,
             budget_eff=0.08, depth=0.08, dep_order=0.05, proactive=0.04,
             centrality=0.05   (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg               = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp              = _hospital_maintained_score(hospital_health_log)
    no_blackout       = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade           = kwargs.get("_adjusted_cascade_score") or _cascade_contained_score(failure_history, total_nodes)
    budget_efficiency = _budget_conservation_score(budget_spent, budget_total)
    depth             = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order         = _dependency_order_score(dependency_order_log or [])
    proactive         = _proactive_rate_score(action_history or [])
    # P2: centrality protection (optional — defaults to neutral 0.5 when not passed)
    centrality        = _centrality_protection_score(
        action_history or [], high_centrality_nodes or [], failure_history
    )

    raw = (0.20 * avg
           + 0.25 * hosp
           + 0.15 * no_blackout
           + 0.10 * cascade
           + 0.08 * budget_efficiency   # slightly reduced to make room
           + 0.08 * depth               # slightly reduced to make room
           + 0.05 * dep_order
           + 0.04 * proactive
           + 0.05 * centrality)         # NEW weight  → sum = 1.00
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
    **kwargs,
) -> float:
    """
    task_gen_blackout grader.
    Root-generator cascade to hospitals — high hospital weight.
    Weights: avg=0.16, hosp=0.30, no_blackout=0.16, cascade=0.10,
             depth=0.09, dep_order=0.05, proactive=0.08, budget_eff=0.06 (sum=1.00)
    Proactive component added so a generator-blackout passive agent scores near zero.
    """
    if not final_sector_summary:
        return _safe(0.15)  # no data → low score, not neutral 0.5

    avg         = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp        = _hospital_maintained_score(hospital_health_log)
    no_blackout = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade     = kwargs.get("_adjusted_cascade_score") or _cascade_contained_score(failure_history, total_nodes)
    depth       = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order   = _dependency_order_score(dependency_order_log or [])
    proactive   = _proactive_rate_score(action_history or [])
    budget_efficiency = _budget_efficiency_score(budget_spent, budget_total)

    raw = (0.16 * avg
            + 0.30 * hosp
            + 0.16 * no_blackout
            + 0.10 * cascade
            + 0.09 * depth
            + 0.05 * dep_order
            + 0.08 * proactive
            + 0.06 * budget_efficiency)
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
    high_centrality_nodes: List[str] | None = None,   # P2: centrality-aware bonus
    **kwargs,
) -> float:
    """
    task_cyberattack grader.
    Sustained SCADA + compound faults + storm — hardest scenario.
    Weights: avg=0.18, hosp=0.28, no_blackout=0.16, cascade=0.10, budget_eff=0.08,
             depth=0.06, dep_order=0.05, proactive=0.04, centrality=0.05  (sum=1.00)
    """
    if not final_sector_summary:
        return _safe(0.5)

    avg               = _clamp(sum(final_sector_summary.values()) / len(final_sector_summary))
    hosp              = _hospital_maintained_score(hospital_health_log)
    no_blackout       = _clamp(1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0)
    cascade           = kwargs.get("_adjusted_cascade_score") or _cascade_contained_score(failure_history, total_nodes)
    budget_efficiency = _budget_conservation_score(budget_spent, budget_total)
    depth             = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order         = _dependency_order_score(dependency_order_log or [])
    proactive         = _proactive_rate_score(action_history or [])
    # P2: centrality protection (optional — defaults to neutral 0.5 when not passed)
    centrality        = _centrality_protection_score(
        action_history or [], high_centrality_nodes or [], failure_history
    )

    raw = (0.18 * avg
           + 0.28 * hosp
           + 0.16 * no_blackout
           + 0.10 * cascade
           + 0.08 * budget_efficiency   # slightly reduced to make room
           + 0.06 * depth               # slightly reduced to make room
           + 0.05 * dep_order
           + 0.04 * proactive
           + 0.05 * centrality)         # NEW weight  → sum = 1.00
    return _safe(raw)


def grade_surge_demand(
    failure_history: List[Set[str]],
    hospital_health_log: List[float],
    final_sector_summary: Dict[str, float],
    budget_spent: float,
    budget_total: float,
    total_nodes: int = 0,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
    **kwargs,
) -> float:
    """
    task_surge_demand grader.

    Load-spike task with no direct equipment faults.
    Weights: hospital_safety=0.45, cascade_contained=0.40, budget_efficiency=0.15
    """
    if not final_sector_summary:
        return _safe(0.15)

    hospital_safety = _hospital_maintained_score(hospital_health_log)
    cascade_contained = kwargs.get("_adjusted_cascade_score") or _cascade_contained_score(
        failure_history,
        total_nodes,
    )
    budget_efficiency = _budget_efficiency_score(budget_spent, budget_total)

    raw = (
        0.45 * hospital_safety
        + 0.40 * cascade_contained
        + 0.15 * budget_efficiency
    )
    return _safe(raw)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

def grade_real_city(
    failure_history: List[List[str]] | None = None,
    hospital_health_log: List[float] | None = None,
    cascade_depth_log: List[int] | None = None,
    dependency_order_log: List[int] | None = None,
    action_history: List[str] | None = None,
    final_sector_summary: Dict[str, float] | None = None,
    budget_spent: float = 0.0,
    budget_total: float = 15.0,
    total_nodes: int = 18,
    **kwargs,
) -> float:
    """
    Metro City (18-node, 35-step) grader.

    Components:
      35%  Hospital survival  — min hospital health across all 35 steps
      20%  Average infrastructure health across all active failures
      20%  No-blackout preservation  — fraction of steps with 0 critical failures
      15%  Radius coverage utilized  — action diversity (agent used non-wait actions)
      10%  Cascade containment  — low average cascade depth
    """
    failure_history = failure_history or []
    hospital_health_log = hospital_health_log or []
    cascade_depth_log = cascade_depth_log or []
    dependency_order_log = dependency_order_log or []
    action_history = action_history or []
    final_sector_summary = final_sector_summary or {}
    steps = max(len(failure_history), 1)

    # 1. Hospital survival
    avg_hospital_health = sum(hospital_health_log) / max(len(hospital_health_log), 1)
    hospital_score = _clamp(avg_hospital_health)

    # 2. Average infrastructure health (from failure history)
    total_failures = sum(len(f) for f in failure_history)
    failure_rate = total_failures / max(steps * total_nodes, 1)
    health_score = _clamp(1.0 - failure_rate)

    # 3. No-blackout preservation (steps with zero critical failures)
    zero_failure_steps = sum(1 for f in failure_history if len(f) == 0)
    blackout_score = _clamp(zero_failure_steps / steps)

    # 4. Action diversity (proxy for radius coverage utilization)
    non_wait  = sum(1 for a in action_history if a != "wait")
    diversity = _clamp(non_wait / max(len(action_history), 1))

    # 5. Cascade containment
    avg_depth = sum(cascade_depth_log) / max(len(cascade_depth_log), 1)
    cascade_score = _clamp(1.0 - avg_depth / 5.0)

    composite = (
        0.35 * hospital_score
        + 0.20 * health_score
        + 0.20 * blackout_score
        + 0.15 * diversity
        + 0.10 * cascade_score
    )
    return _safe(composite)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GRADERS = {
    "task_easy":         grade_easy,
    "task_medium":       grade_medium,
    "task_hard":         grade_hard,
    "task_gen_blackout": grade_gen_blackout,
    "task_cyberattack":  grade_cyberattack,
    "task_surge_demand": grade_surge_demand,
    "task_real_city":    grade_real_city,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(task_id: str, uncapped: bool = False, **kwargs) -> float:
    """
    Dispatch to the appropriate grader and guarantee the result is in (0, 1).

    Two-layer defence:
    1. The individual grader already returns via _safe() → in (0,1).
    2. grade() applies _safe() once more on the final value so that any
       future change to a grader cannot accidentally leak 0.0 or 1.0.

    Args:
        task_id:  One of the registered CascadeGuard task IDs.
        uncapped: If True, returns the raw grader composite BEFORE the final
                  _safe() boundary clamp. This is the Mercor bonus sub-theme
                  (uncapped reward environments). The value is still passed
                  through each sub-scorer's _clamp(), so it is numerically
                  stable but NOT guaranteed to be in (0.01, 0.99).
                  Use only for training signals — never for OpenEnv submission.
        **kwargs: All grader arguments (failure_history, hospital_health_log,
                  final_sector_summary, budget_spent, budget_total, etc.)
                  v2.2 optional kwargs:
                    controlled_cascade_sacrifices (int): number of sacrifice
                      nodes used in the episode; each deducts 0.1 from score.
                    lockdown_steps (List[int]): step indices during which a
                      multi_sector_lockdown was active; these are excluded from
                      the cascade containment measurement window.

    Exception handling: catches ALL exceptions so grader crashes never
    propagate to the OpenEnv pipeline.
    """
    if task_id not in GRADERS:
        # Unknown task — return neutral rather than crashing.
        return _safe(0.15)  # no longer 0.5 — unknown task should not floor at neutral

    # v2.2: extract and strip v2.2-specific kwargs before passing to grader
    sacrifice_count: int = int(kwargs.pop("controlled_cascade_sacrifices", 0) or 0)
    lockdown_steps: List[int] = list(kwargs.pop("lockdown_steps", []) or [])

    # Extract critical failure count for the CF penalty (passed in kwargs,
    # does NOT get forwarded to the per-task grader functions).
    # The caller (cascade_environment.py state property) accumulates this from
    # the failure_history. We compute it here if not directly provided.
    explicit_cf: Optional[int] = kwargs.pop("total_critical_failures", None)

    # v2.2: if lockdown steps exist, override the failure_history view for
    # cascade containment by replacing it with a lockdown-aware score injected
    # via high_centrality_nodes pathway — we store the adjusted cascade score
    # and pass it through the grader unchanged via the kwargs.
    if lockdown_steps:
        failure_history = kwargs.get("failure_history", [])
        total_nodes = kwargs.get("total_nodes", 0)
        adjusted_cascade = _cascade_contained_score_lockdown(
            failure_history, total_nodes, lockdown_steps
        )
        kwargs["_adjusted_cascade_score"] = adjusted_cascade

    try:
        grader_func = GRADERS[task_id]
        raw = grader_func(**kwargs)
    except Exception:
        # Any grader crash → low fallback, not neutral 0.5 (hidden attractor).
        return _safe(0.10)

    # v2.2: apply controlled_cascade sacrifice penalty
    if sacrifice_count > 0:
        raw = _clamp(raw - _sacrifice_penalty(sacrifice_count))

    # ── CRITICAL FIX: Apply accumulated critical-failure penalty ──────────────
    # This closes the reward-score decoupling gap: the step reward already fires
    # a per-CF penalty, but the final grader score previously ignored CF count
    # entirely. Now both signals move in the same direction.
    #
    # If explicit_cf was passed, use it. Otherwise derive from failure_history.
    if explicit_cf is None:
        fh = kwargs.get("failure_history", [])
        # Count total critical-node failures across the episode
        # (approximate: we count each step a node was failed, summed)
        explicit_cf = sum(len(s) for s in fh) if fh else 0
    if explicit_cf > 0:
        cf_deduction = _critical_failure_penalty(explicit_cf)
        raw = max(SCORE_EPS, raw - cf_deduction)

    if uncapped:
        # Return raw composite (already through sub-scorer _clamp(), but NOT _safe())
        # Caller is responsible for interpreting values outside (0.01, 0.99).
        return raw

    # Belt-and-suspenders: apply _safe even though graders already do so.
    return _safe(raw)