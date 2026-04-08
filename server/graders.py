from __future__ import annotations

from typing import Dict, List, Set


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _hospital_maintained_score(hospital_health_log: List[float]) -> float:
    total_steps = len(hospital_health_log)
    violations = sum(1 for h in hospital_health_log if h < 0.4)
    if violations == 0:
        return 1.0
    return max(0.0, 1.0 - violations / max(total_steps, 1))


def _cascade_contained_score(failure_history: List[Set[str]], total_nodes: int) -> float:
    if not failure_history:
        return 1.0
    all_seen: Set[str] = set()
    for s in failure_history:
        all_seen.update(s)
    denom = max(total_nodes or len(all_seen), 1)
    worst_fail_ratio = max(len(s) for s in failure_history) / denom
    return _clamp(1.0 - worst_fail_ratio)


def _dependency_order_score(dependency_order_log: List[int]) -> float:
    if not dependency_order_log:
        return 0.5
    return _clamp(sum(dependency_order_log) / max(len(dependency_order_log), 1))


def _cascade_depth_score(cascade_depth_log: List[int], total_nodes: int) -> float:
    if not cascade_depth_log:
        return 1.0
    denom = max(total_nodes, 1)
    depth_ratio = sum(cascade_depth_log) / (len(cascade_depth_log) * denom)
    return _clamp(1.0 - depth_ratio)


def _intervention_efficiency_score(action_history: List[str], budget_spent: float, budget_total: float) -> float:
    if not action_history:
        return 0.5
    active_actions = sum(1 for a in action_history if a != "wait")
    proactive_rate = active_actions / max(len(action_history), 1)
    budget_eff = _clamp(1.0 - (budget_spent / max(budget_total, 1e-9)))
    return _clamp(0.6 * proactive_rate + 0.4 * budget_eff)


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
    Grader for task_easy.
    Weights: avg_health=0.40, no_blackout=0.40, cascade_contained=0.20
    """
    avg = sum(final_sector_summary.values()) / len(final_sector_summary) if final_sector_summary else 0.0
    no_blackout = 1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    cascade = _cascade_contained_score(failure_history, total_nodes)
    depth = _cascade_depth_score(cascade_depth_log or [], total_nodes)

    raw = 0.35 * avg + 0.35 * no_blackout + 0.20 * cascade + 0.10 * depth
    return round(_clamp(raw), 4)


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
    Grader for task_medium.
    Weights: avg=0.30, hospital_maintained=0.35, no_blackout=0.25, cascade=0.10
    """
    avg = sum(final_sector_summary.values()) / len(final_sector_summary) if final_sector_summary else 0.0
    hosp = _hospital_maintained_score(hospital_health_log)
    no_blackout = 1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    cascade = _cascade_contained_score(failure_history, total_nodes)
    depth = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order = _dependency_order_score(dependency_order_log or [])

    raw = 0.25 * avg + 0.30 * hosp + 0.20 * no_blackout + 0.10 * cascade + 0.10 * depth + 0.05 * dep_order
    return round(_clamp(raw), 4)


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
    Grader for task_hard.
    Weights: avg=0.25, hospital=0.35, no_blackout=0.20, cascade=0.10, budget_eff=0.10
    """
    avg = sum(final_sector_summary.values()) / len(final_sector_summary) if final_sector_summary else 0.0
    hosp = _hospital_maintained_score(hospital_health_log)
    no_blackout = 1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    cascade = _cascade_contained_score(failure_history, total_nodes)
    budget_efficiency = _clamp(1.0 - (budget_spent / max(budget_total, 1e-9)))
    depth = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order = _dependency_order_score(dependency_order_log or [])
    intervention = _intervention_efficiency_score(action_history or [], budget_spent, budget_total)

    raw = (
        0.20 * avg
        + 0.25 * hosp
        + 0.15 * no_blackout
        + 0.10 * cascade
        + 0.10 * budget_efficiency
        + 0.10 * depth
        + 0.05 * dep_order
        + 0.05 * intervention
    )
    return round(_clamp(raw), 4)


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
    Grader for task_gen_blackout.
    Root generator cascades to hospitals - hard cascade containment test.
    Weights: avg=0.25, hospital_maintained=0.40, no_blackout=0.25, cascade=0.10
    The unusually high hospital weight reflects that the whole scenario's
    point is protecting hospitals from a catastrophic root failure.
    """
    avg = sum(final_sector_summary.values()) / len(final_sector_summary) if final_sector_summary else 0.0
    hosp = _hospital_maintained_score(hospital_health_log)
    no_blackout = 1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    cascade = _cascade_contained_score(failure_history, total_nodes)
    depth = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order = _dependency_order_score(dependency_order_log or [])

    raw = 0.20 * avg + 0.35 * hosp + 0.20 * no_blackout + 0.10 * cascade + 0.10 * depth + 0.05 * dep_order
    return round(_clamp(raw), 4)


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
    Grader for task_cyberattack.
    Sustained SCADA + compound faults + storm - the hardest scenario.
    Weights: avg=0.20, hospital=0.40, no_blackout=0.20, cascade=0.10, budget_eff=0.10
    """
    avg = sum(final_sector_summary.values()) / len(final_sector_summary) if final_sector_summary else 0.0
    hosp = _hospital_maintained_score(hospital_health_log)
    no_blackout = 1.0 if all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    cascade = _cascade_contained_score(failure_history, total_nodes)
    budget_efficiency = _clamp(1.0 - (budget_spent / max(budget_total, 1e-9)))
    depth = _cascade_depth_score(cascade_depth_log or [], total_nodes)
    dep_order = _dependency_order_score(dependency_order_log or [])
    intervention = _intervention_efficiency_score(action_history or [], budget_spent, budget_total)

    raw = (
        0.18 * avg
        + 0.28 * hosp
        + 0.16 * no_blackout
        + 0.10 * cascade
        + 0.10 * budget_efficiency
        + 0.08 * depth
        + 0.05 * dep_order
        + 0.05 * intervention
    )
    return round(_clamp(raw), 4)


GRADERS = {
    "task_easy":         grade_easy,
    "task_medium":       grade_medium,
    "task_hard":         grade_hard,
    "task_gen_blackout": grade_gen_blackout,
    "task_cyberattack":  grade_cyberattack,
}


def grade(task_id: str, **kwargs) -> float:
    """Dispatch to the appropriate grader for the given task_id."""
    if task_id not in GRADERS:
        raise KeyError(f"Unknown task_id {task_id!r}. Valid: {list(GRADERS)}")
    return GRADERS[task_id](**kwargs)
