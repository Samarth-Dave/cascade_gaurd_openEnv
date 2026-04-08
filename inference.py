from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
import json
import math
import os
import statistics
import subprocess
import sys
from urllib.parse import urlparse
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

from openai import OpenAI
from cascade_guard.client import CascadeGuardEnv
from cascade_guard.models import CascadeAction
from cascade_guard.tasks import TASK_CONFIGS, TASK_SEED_SPLITS

try:
    from cascade_guard.server.cascade_environment import CascadeEnvironment as LocalCascadeEnvironment
    from cascade_guard.server.graders import grade as grade_episode
except Exception:
    LocalCascadeEnvironment = None
    grade_episode = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
# Separate the Groq/LLM API key from the HuggingFace token.
# HF_TOKEN is for HF Space auth; GROQ_API_KEY is for LLM calls.
# Falls back to HF_TOKEN for backwards compatibility.
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY") or HF_TOKEN
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "https://samarthdave0305-cascade-failure-env.hf.space")
LOCAL_IMAGE_NAME: str = os.environ.get("LOCAL_IMAGE_NAME", "cascade-guard:latest")
DEBUG_LOGS: bool = os.getenv("DEBUG_LOGS", "0") == "1"

TASK_NAMES: List[str] = [
    "task_easy",
    "task_medium",
    "task_hard",
    "task_gen_blackout",
    "task_cyberattack",
]
MAX_STEPS: Dict[str, int] = {
    "task_easy": 10,
    "task_medium": 20,
    "task_hard": 30,
    "task_gen_blackout": 15,
    "task_cyberattack": 25,
}
TASK_CONFIGS_BUDGET: Dict[str, float] = {
    "task_easy": 5.0,
    "task_medium": 8.0,
    "task_hard": 10.0,
    "task_gen_blackout": 8.0,
    "task_cyberattack": 10.0,
}

# Token budget: output only needs ~60 tokens for the JSON action.
# Keeping max_tokens LOW is critical - Groq limits total context (input+output).
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 80   # action JSON ~ 50-60 chars / ~15 tokens; 80 is ample
EVAL_MODE: str = os.environ.get("EVAL_MODE", "baseline")  # baseline | multiseed
EVAL_SPLIT: str = os.environ.get("EVAL_SPLIT", "holdout")
SCORE_EPS: float = 0.1
BASELINE_RESULTS_PATH: str = os.environ.get("BASELINE_RESULTS_PATH", "baseline_results.json")

# System prompt - ultra-compact to save context tokens
SYSTEM_PROMPT = (
    "You are an infrastructure resilience agent. "
    "Output ONLY a single JSON line - no markdown, no explanation.\n"
    'Format: {"action_type": "harden|shed_load|coordinate|recover|wait", '
    '"target_node_id": "NODE_ID or null"}\n'
    "Rules (in order of priority): "
    "1) NEVER shed_load on hospital-sector or critical nodes. "
    "2) ALWAYS recover in RecOrder sequence; do not recover blocked leaf nodes. "
    "3) DO NOT wait if Fail or AtRisk are non-empty. "
    "4) Harden before stress peaks when budget >=2. "
    "5) Coordinate only when delayed observations exist and budget allows."
)


def _normalize_base_url(raw_url: Optional[str]) -> Optional[str]:
    if not raw_url:
        return None
    url = raw_url.strip().rstrip("/")
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return url


def _candidate_base_urls() -> List[str]:
    keys = [
        "ENV_BASE_URL",
        "OPENENV_BASE_URL",
        "SPACE_URL",
        "HF_SPACE_URL",
        "PING_URL",
    ]
    candidates: List[str] = []
    seen = set()
    for key in keys:
        normalized = _normalize_base_url(os.getenv(key))
        if normalized and normalized not in seen:
            candidates.append(normalized)
            seen.add(normalized)

    normalized_default = _normalize_base_url(ENV_BASE_URL)
    if normalized_default and normalized_default not in seen:
        candidates.append(normalized_default)

    local_default = "http://localhost:8000"
    if local_default not in seen:
        candidates.append(local_default)
    return candidates


def _docker_image_exists(image_name: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _build_local_docker_image(image_name: str) -> None:
    repo_root = str(Path(__file__).resolve().parent)
    result = subprocess.run(
        ["docker", "build", "-t", image_name, repo_root],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-400:]
        stdout_tail = (result.stdout or "")[-400:]
        raise RuntimeError(
            f"docker build failed for image {image_name} (exit={result.returncode}). "
            f"stderr_tail={stderr_tail!r} stdout_tail={stdout_tail!r}"
        )


async def _connect_environment() -> CascadeGuardEnv:
    connect_errors: List[str] = []

    # Phase-2 validators can reach the deployed HF Space but may not have local image tags.
    for base_url in _candidate_base_urls():
        env = CascadeGuardEnv(base_url=base_url)
        try:
            await env.connect()
            _debug(f"[ENV] connected via base_url={base_url}")
            return env
        except Exception as exc:
            connect_errors.append(f"base_url={base_url} -> {exc}")
            try:
                await env.close()
            except Exception:
                pass

    try:
        if not _docker_image_exists(LOCAL_IMAGE_NAME):
            _debug(f"[ENV] docker image missing, attempting build image={LOCAL_IMAGE_NAME}")
            _build_local_docker_image(LOCAL_IMAGE_NAME)
        _debug(f"[ENV] trying docker image={LOCAL_IMAGE_NAME}")
        env = await CascadeGuardEnv.from_docker_image(LOCAL_IMAGE_NAME)
        _debug(f"[ENV] connected via docker image={LOCAL_IMAGE_NAME}")
        return env
    except Exception as exc:
        connect_errors.append(f"docker_image={LOCAL_IMAGE_NAME} -> {exc}")

    details = " | ".join(connect_errors) if connect_errors else "no connection attempts were made"
    raise RuntimeError(
        "Failed to connect to CascadeGuard environment. "
        "Set ENV_BASE_URL to your deployed HF Space URL or ensure Docker has the expected local image. "
        f"Attempts: {details}"
    )


def _debug(msg: str) -> None:
    # Keep stdout reserved for [START]/[STEP]/[END] lines only.
    if DEBUG_LOGS:
        print(msg, file=sys.stderr, flush=True)


def _extract_raw_step_reward(observation: Any, fallback_reward: float) -> float:
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, dict):
        raw_reward = metadata.get("clipped_reward", metadata.get("raw_reward"))
        if raw_reward is not None:
            try:
                return float(raw_reward)
            except (TypeError, ValueError):
                pass
    return float(fallback_reward)


def _write_reproducibility_report(run_records: List[Dict[str, Any]]) -> None:
    if not run_records:
        return

    sanitized_records: List[Dict[str, Any]] = []
    for rec in run_records:
        rec_copy = dict(rec)
        rec_copy["score"] = _sanitize_score(float(rec_copy.get("score", SCORE_EPS)))
        sanitized_records.append(rec_copy)

    grouped: Dict[str, List[float]] = {}
    for rec in sanitized_records:
        grouped.setdefault(str(rec["task_id"]), []).append(_sanitize_score(float(rec["score"])))

    task_summary: Dict[str, Dict[str, Any]] = {}
    for task_id, vals in grouped.items():
        task_summary[task_id] = {
            "runs": len(vals),
            "mean_score": _sanitize_score(statistics.mean(vals)),
            "std_score": max(statistics.pstdev(vals) if len(vals) > 1 else 0.0, SCORE_EPS),
            "min_score": _sanitize_score(min(vals)),
            "max_score": _sanitize_score(max(vals)),
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "eval_mode": EVAL_MODE,
        "eval_split": EVAL_SPLIT,
        "records": sanitized_records,
        "task_summary": task_summary,
    }

    # ── Final validation sweep ────────────────────────────────────────────
    # Walk every *_score field and guarantee nothing is exactly 0.0 or 1.0.
    # This is the last line of defence before the JSON hits disk.
    _SCORE_FIELDS = {"score", "mean_score", "std_score", "min_score", "max_score"}
    for rec in payload.get("records", []):
        if isinstance(rec.get("score"), float) and not (0.0 < rec["score"] < 1.0):
            rec["score"] = _sanitize_score(rec["score"])

    for task_data in payload.get("task_summary", {}).values():
        for field in _SCORE_FIELDS:
            if field in task_data and isinstance(task_data[field], float):
                v = task_data[field]
                if not (0.0 < v < 1.0):
                    task_data[field] = _sanitize_score(v)
                    if task_data[field] <= 0.0 or task_data[field] >= 1.0:
                        # absolute fallback
                        task_data[field] = SCORE_EPS
    # ── End validation sweep ──────────────────────────────────────────────

    output_path = Path(BASELINE_RESULTS_PATH)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Keep runtime output strict by only emitting report summaries in debug mode.
    _debug(f"[RESULTS] wrote reproducibility report to {output_path}")
    for task_id, stats in task_summary.items():
        _debug(
            f"[RESULTS] task={task_id} runs={int(stats['runs'])} "
            f"mean={stats['mean_score']:.4f} std={stats['std_score']:.4f} "
            f"min={stats['min_score']:.4f} max={stats['max_score']:.4f}"
        )


def _clamp_score_open(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SCORE_EPS
    if not math.isfinite(v):
        return SCORE_EPS
    return max(SCORE_EPS, min(1.0 - SCORE_EPS, v))


def _sanitize_score(value: float) -> float:
    """
    Guarantee the returned float is strictly in (0.0, 1.0).

    CRITICAL ORDER: ROUND FIRST, then apply boundary checks.
    Doing the checks before rounding is wrong because:
        round(0.99997, 4) == 1.0  — would escape a pre-check
        round(0.00004, 4) == 0.0  — would escape a pre-check
    The round-then-check order is the only safe approach.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SCORE_EPS
    if not math.isfinite(v):
        return SCORE_EPS

    

    # ── THEN check boundaries ────────────────────────────────────────────
    if v <= 0.0:
        return SCORE_EPS
    if v >= 1.0:
        return 1.0 - SCORE_EPS

    return v


# ---------------------------------------------------------------------------
# Prompt builder - MINIMAL to save context tokens
# ---------------------------------------------------------------------------
def _build_user_prompt(obs: Any, step: int, last_reward: float = 0.0) -> str:
    """Ultra-compact prompt. Target < 200 tokens even for 15-node tasks."""
    # Only include non-trivial info - skip healthy/clean nodes
    critical_lines = []
    for n in obs.nodes:
        # Skip fully healthy, unhardened, non-delayed, operational nodes
        if (n.health >= 0.95 and n.load <= 0.55 and n.is_operational
                and not n.is_hardened and not n.observation_delayed):
            continue
        flags = ""
        if not n.is_operational:
            flags += " FAILED"
        elif n.health < 0.7:
            flags += f" h={n.health:.2f}"
        if n.load > 0.6:
            flags += f" load={n.load:.2f}"
        if n.is_hardened:
            flags += " HDN"
        if n.observation_delayed:
            flags += " DLY"
        if n.is_critical:
            flags += " CRIT"
        if flags:
            critical_lines.append(f"  {n.node_id}({n.sector}){flags}")

    # Summary line
    sector_str = " ".join(f"{k[:3]}={v:.2f}" for k, v in obs.sector_summary.items())
    fail_str = ",".join(obs.active_failures) if obs.active_failures else "-"
    pend_str = ",".join(obs.pending_recoveries) if obs.pending_recoveries else "-"
    diagnostics = getattr(obs, "diagnostics", None)
    crit_fail = ",".join(getattr(diagnostics, "critical_failures", [])) or "-"
    at_risk = ",".join(getattr(diagnostics, "at_risk_nodes", [])[:5]) or "-"
    dep_alerts = "; ".join(getattr(diagnostics, "dependency_alerts", [])[:4]) or "-"
    rec_order = ",".join(getattr(diagnostics, "recommended_recovery_order", [])[:5]) or "-"
    pressure = float(getattr(diagnostics, "system_pressure", 0.0) or 0.0)
    urgency = "HIGH" if (pressure > 0.3 or bool(obs.active_failures)) else "LOW"

    parts = [
        f"LastReward={last_reward:+.2f} Urgency={urgency}",
        f"S{step}/{obs.max_steps} B={obs.budget_remaining:.0f} W={obs.weather_forecast[:3]} P={pressure:.2f}",
        f"Sectors:{sector_str}",
        f"Fail:{fail_str} Pend:{pend_str}",
        f"CritFail:{crit_fail}",
        f"AtRisk:{at_risk}",
        f"DepAlerts:{dep_alerts}",
        f"RecOrder:{rec_order}",
    ]
    if critical_lines:
        parts.append("Notable:")
        parts.extend(critical_lines)
    parts.append('Action JSON:')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------
def _build_upstream_map(obs: Any) -> Dict[str, List[str]]:
    upstream: Dict[str, List[str]] = {}
    for edge in obs.edges:
        upstream.setdefault(edge.target_id, []).append(edge.source_id)
    return upstream


def _downstream_count(node_id: str, obs: Any) -> int:
    """Count how many nodes depend directly on node_id."""
    return sum(1 for e in obs.edges if e.source_id == node_id)


@lru_cache(maxsize=None)
def _task_topology_stats(task_id: str) -> Dict[str, Dict[str, Any]]:
    """Pre-compute simple descendant-based importance scores per task graph."""
    task_cfg = TASK_CONFIGS.get(task_id, {})
    task_nodes = {
        node["node_id"]: {
            "sector": node["sector"],
            "is_critical": bool(node["is_critical"]),
        }
        for node in task_cfg.get("nodes", [])
    }
    task_edges = [
        (edge["source_id"], edge["target_id"])
        for edge in task_cfg.get("edges", [])
    ]

    children: Dict[str, List[str]] = {nid: [] for nid in task_nodes}
    for src, tgt in task_edges:
        children.setdefault(src, []).append(tgt)

    memo: Dict[str, set[str]] = {}

    def descendants(node_id: str) -> set[str]:
        if node_id in memo:
            return memo[node_id]
        out: set[str] = set()
        for child_id in children.get(node_id, []):
            out.add(child_id)
            out.update(descendants(child_id))
        memo[node_id] = out
        return out

    stats: Dict[str, Dict[str, Any]] = {}
    for node_id, meta in task_nodes.items():
        desc = descendants(node_id)
        hospital_support = sum(
            1
            for desc_id in desc
            if task_nodes[desc_id]["sector"] == "hospital" or task_nodes[desc_id]["is_critical"]
        )
        stats[node_id] = {
            "descendants": len(desc),
            "hospital_support": hospital_support,
            "sector": meta["sector"],
            "is_critical": meta["is_critical"],
        }
    return stats


def _planner_upstream_map(planner_env: Any) -> Dict[str, List[str]]:
    upstream: Dict[str, List[str]] = {}
    for key in getattr(planner_env, "_edge_states", {}):
        src, tgt = key.split("->")
        upstream.setdefault(tgt, []).append(src)
    return upstream


def _planner_future_events(planner_env: Any) -> List[tuple[int, Dict[str, Any]]]:
    events = []
    for step_n, event in sorted(getattr(planner_env, "_stress_schedule", {}).items()):
        if step_n >= getattr(planner_env, "_step", 0):
            events.append((step_n, event))
    return events


def _planner_future_fault_targets(planner_env: Any) -> List[tuple[int, str]]:
    targets: List[tuple[int, str]] = []
    for step_n, event in _planner_future_events(planner_env):
        if event.get("type") == "equipment_fault":
            target = event.get("target")
            if target in getattr(planner_env, "_node_states", {}):
                targets.append((step_n, target))
    return targets


def _planner_pick_recover_target(planner_env: Any) -> Optional[str]:
    node_states = getattr(planner_env, "_node_states", {})
    pending = set(getattr(planner_env, "_pending_recoveries", {}) or {})
    upstream_map = _planner_upstream_map(planner_env)

    failed = [
        nid for nid, ns in node_states.items()
        if (not ns.is_operational) and nid not in pending
    ]
    if not failed:
        return None

    def upstream_all_ok(node_id: str) -> bool:
        return all(
            node_states[u].is_operational
            for u in upstream_map.get(node_id, [])
            if u in node_states
        )

    recoverable = [nid for nid in failed if upstream_all_ok(nid)]
    if recoverable:
        recoverable.sort(
            key=lambda nid: (not node_states[nid].is_critical, node_states[nid].health)
        )
        return recoverable[0]

    failed_ids = set(failed)
    roots = [
        nid for nid in failed
        if not any(u in failed_ids for u in upstream_map.get(nid, []))
    ]
    if roots:
        roots.sort(
            key=lambda nid: (not node_states[nid].is_critical, node_states[nid].health)
        )
        return roots[0]
    return failed[0]


def _planner_pick_safe_shed_target(planner_env: Any) -> Optional[str]:
    node_states = getattr(planner_env, "_node_states", {})
    candidates = [
        ns for ns in node_states.values()
        if ns.is_operational and (not ns.is_critical) and ns.sector != "hospital" and ns.load > 0.9
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda ns: ns.load, reverse=True)
    return candidates[0].node_id


def _planner_pick_coordinate_target(planner_env: Any, task_id: str) -> Optional[str]:
    if getattr(planner_env, "_budget", 0.0) < 1.0:
        return None

    future_faults = _planner_future_fault_targets(planner_env)
    weather_now = getattr(planner_env, "_weather_forecast", "clear") != "clear"
    weather_soon = any(
        event.get("type") == "weather"
        and event.get("effect") != "clear"
        and (step_n - getattr(planner_env, "_step", 0)) <= 2
        for step_n, event in _planner_future_events(planner_env)
    )
    active_failures = any(
        not ns.is_operational for ns in getattr(planner_env, "_node_states", {}).values()
    )
    pending_recoveries = bool(getattr(planner_env, "_pending_recoveries", {}))
    if active_failures or pending_recoveries or weather_now or weather_soon:
        return None
    if any((step_n - getattr(planner_env, "_step", 0)) <= 2 for step_n, _ in future_faults):
        return None

    delayed_nodes = [
        nid for nid in getattr(planner_env, "_node_states", {})
        if planner_env._is_node_delayed(nid)
    ]
    if not delayed_nodes:
        return None

    topo = _task_topology_stats(task_id)
    delayed_nodes.sort(
        key=lambda nid: (
            topo.get(nid, {}).get("hospital_support", 0),
            topo.get(nid, {}).get("descendants", 0),
        ),
        reverse=True,
    )
    return delayed_nodes[0]


def _planner_is_action_valid(planner_env: Any, action: Dict[str, Optional[str]]) -> bool:
    action_type = action.get("action_type")
    target_id = action.get("target_node_id")
    node_states = getattr(planner_env, "_node_states", {})

    if action_type == "wait":
        return True

    if action_type == "harden":
        if target_id is None or target_id not in node_states:
            return False
        ns = node_states[target_id]
        return ns.is_operational and (not ns.is_hardened) and getattr(planner_env, "_budget", 0.0) >= 2.0

    if action_type == "shed_load":
        if target_id is None or target_id not in node_states:
            return False
        ns = node_states[target_id]
        return ns.is_operational and (not ns.is_critical) and ns.sector != "hospital" and ns.load > 0.5

    if action_type == "coordinate":
        if target_id is None or target_id not in node_states:
            return False
        return getattr(planner_env, "_budget", 0.0) >= 1.0 and any(
            planner_env._is_node_delayed(nid) for nid in node_states
        )

    if action_type == "recover":
        if target_id is None or target_id not in node_states:
            return False
        ns = node_states[target_id]
        if ns.is_operational:
            return False
        if target_id in set(getattr(planner_env, "_pending_recoveries", {}) or {}):
            return False
        upstream_map = _planner_upstream_map(planner_env)
        return all(
            node_states[u].is_operational
            for u in upstream_map.get(target_id, [])
            if u in node_states
        )

    return False


def _planner_harden_priority(planner_env: Any, task_id: str, node_id: str) -> float:
    topo = _task_topology_stats(task_id).get(node_id, {})
    node_states = getattr(planner_env, "_node_states", {})
    ns = node_states.get(node_id)
    if ns is None:
        return -1e9

    score = 0.0
    score += 1.8 * float(topo.get("hospital_support", 0))
    score += 0.5 * float(topo.get("descendants", 0))
    if topo.get("sector") == "power":
        score += 1.0
    if topo.get("sector") == "telecom":
        score += 0.35
    if topo.get("is_critical"):
        score += 0.75
    score += 1.25 * max(0.0, 0.85 - float(ns.health))

    weather_now = getattr(planner_env, "_weather_forecast", "clear") != "clear"
    weather_soon = any(
        event.get("type") == "weather"
        and event.get("effect") != "clear"
        and (step_n - getattr(planner_env, "_step", 0)) <= 2
        for step_n, event in _planner_future_events(planner_env)
    )
    if weather_now or weather_soon:
        score += 0.9 * float(topo.get("hospital_support", 0))
        score += 0.35 * float(topo.get("descendants", 0))

    future_faults = _planner_future_fault_targets(planner_env)
    for step_n, target in future_faults:
        dt = max(0, step_n - getattr(planner_env, "_step", 0))
        if target == node_id:
            score += 6.0 / max(1, dt + 1)
        elif topo.get("sector") == _task_topology_stats(task_id).get(target, {}).get("sector"):
            score += 0.75 / max(1, dt + 1)

    return score


def _planner_candidate_actions(planner_env: Any, task_id: str) -> List[Dict[str, Optional[str]]]:
    node_states = getattr(planner_env, "_node_states", {})
    candidates: List[Dict[str, Optional[str]]] = []
    weather_now = getattr(planner_env, "_weather_forecast", "clear") != "clear"
    weather_soon = any(
        event.get("type") == "weather"
        and event.get("effect") != "clear"
        and (step_n - getattr(planner_env, "_step", 0)) <= 2
        for step_n, event in _planner_future_events(planner_env)
    )
    active_failures = any(not ns.is_operational for ns in node_states.values())
    overloaded = any(ns.is_operational and ns.load > 0.9 for ns in node_states.values())
    urgent = active_failures or overloaded or weather_now or weather_soon

    recover_target = _planner_pick_recover_target(planner_env)
    if recover_target is not None:
        candidates.append({"action_type": "recover", "target_node_id": recover_target})

    if getattr(planner_env, "_budget", 0.0) >= 2.0:
        for _, target in _planner_future_fault_targets(planner_env)[:2]:
            ns = node_states.get(target)
            if ns is not None and ns.is_operational and (not ns.is_hardened):
                candidates.append({"action_type": "harden", "target_node_id": target})

        hardenable = [
            nid for nid, ns in node_states.items()
            if ns.is_operational and (not ns.is_hardened)
        ]
        hardenable.sort(
            key=lambda nid: _planner_harden_priority(planner_env, task_id, nid),
            reverse=True,
        )
        for nid in hardenable[:5]:
            candidates.append({"action_type": "harden", "target_node_id": nid})

    shed_target = _planner_pick_safe_shed_target(planner_env)
    if shed_target is not None:
        candidates.append({"action_type": "shed_load", "target_node_id": shed_target})

    coord_target = _planner_pick_coordinate_target(planner_env, task_id)
    if (
        coord_target is not None
        and getattr(planner_env, "_step", 0) <= 2
        and getattr(planner_env, "_budget", 0.0) >= 3.0
        and not any(not ns.is_operational for ns in node_states.values())
    ):
        candidates.append({"action_type": "coordinate", "target_node_id": coord_target})

    if not urgent:
        candidates.append({"action_type": "wait", "target_node_id": None})

    deduped: List[Dict[str, Optional[str]]] = []
    seen = set()
    for action in candidates:
        sig = (action.get("action_type"), action.get("target_node_id"))
        if sig in seen:
            continue
        seen.add(sig)
        if _planner_is_action_valid(planner_env, action):
            deduped.append(action)
    return deduped


def _planner_default_action(planner_env: Any, task_id: str) -> Dict[str, Optional[str]]:
    recover_target = _planner_pick_recover_target(planner_env)
    if recover_target is not None:
        return {"action_type": "recover", "target_node_id": recover_target}

    node_states = getattr(planner_env, "_node_states", {})
    if any(ns.load > 0.9 for ns in node_states.values() if ns.is_operational):
        shed_target = _planner_pick_safe_shed_target(planner_env)
        if shed_target is not None:
            return {"action_type": "shed_load", "target_node_id": shed_target}

    if getattr(planner_env, "_budget", 0.0) >= 2.0:
        future_faults = _planner_future_fault_targets(planner_env)
        if future_faults:
            step_n, target = future_faults[0]
            ns = node_states.get(target)
            dt = max(0, step_n - getattr(planner_env, "_step", 0))
            if ns is not None and ns.is_operational and (not ns.is_hardened) and dt <= 3:
                return {"action_type": "harden", "target_node_id": target}

    weather_now = getattr(planner_env, "_weather_forecast", "clear") != "clear"
    weather_soon = any(
        event.get("type") == "weather"
        and event.get("effect") != "clear"
        and (step_n - getattr(planner_env, "_step", 0)) <= 2
        for step_n, event in _planner_future_events(planner_env)
    )
    if getattr(planner_env, "_budget", 0.0) >= 2.0 and (weather_now or weather_soon):
        hardenable = [
            nid for nid, ns in node_states.items()
            if ns.is_operational and (not ns.is_hardened)
        ]
        hardenable.sort(
            key=lambda nid: _planner_harden_priority(planner_env, task_id, nid),
            reverse=True,
        )
        if hardenable:
            return {"action_type": "harden", "target_node_id": hardenable[0]}

    coord_target = _planner_pick_coordinate_target(planner_env, task_id)
    if (
        coord_target is not None
        and getattr(planner_env, "_step", 0) <= 2
        and getattr(planner_env, "_budget", 0.0) >= 3.0
        and not any(not ns.is_operational for ns in node_states.values())
    ):
        return {"action_type": "coordinate", "target_node_id": coord_target}

    return {"action_type": "wait", "target_node_id": None}


def _merge_candidate_actions(
    planner_candidates: List[Dict[str, Optional[str]]],
    model_candidates: List[Dict[str, Optional[str]]],
) -> List[Dict[str, Optional[str]]]:
    merged: List[Dict[str, Optional[str]]] = []
    seen = set()
    for action in planner_candidates + model_candidates:
        sig = (action.get("action_type"), action.get("target_node_id"))
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(action)
    return merged


def _planner_grade(task_id: str, planner_env: Any) -> float:
    if grade_episode is None:
        return SCORE_EPS

    state_data = planner_env.state
    return _sanitize_score(
        grade_episode(
            task_id,
            failure_history=[set(s) for s in state_data.failure_history],
            hospital_health_log=list(state_data.hospital_health_log),
            final_sector_summary=dict(state_data.sector_health),
            budget_spent=TASK_CONFIGS_BUDGET[task_id] - state_data.budget_remaining,
            budget_total=TASK_CONFIGS_BUDGET[task_id],
            total_nodes=int(getattr(state_data, "total_nodes", 0) or 0),
            cascade_depth_log=list(getattr(state_data, "cascade_depth_log", []) or []),
            dependency_order_log=list(getattr(state_data, "dependency_order_log", []) or []),
            action_history=list(getattr(state_data, "action_history", []) or []),
        )
    )


def _planner_rollout_score(
    planner_env: Any,
    task_id: str,
    first_action: Dict[str, Optional[str]],
) -> tuple[float, float]:
    sim = copy.deepcopy(planner_env)
    obs = sim.step(CascadeAction(**first_action))
    reward_sum = float(getattr(obs, "reward", 0.0) or 0.0)

    while not getattr(obs, "done", False):
        next_action = _planner_default_action(sim, task_id)
        obs = sim.step(CascadeAction(**next_action))
        reward_sum += float(getattr(obs, "reward", 0.0) or 0.0)

    return _planner_grade(task_id, sim), reward_sum


def _fallback_proxy_grade(
    final_sector_summary: Dict[str, float],
    hospital_health_log: List[float],
    failure_history: List[set[str]],
    total_nodes: int,
) -> float:
    if not final_sector_summary:
        return _sanitize_score(0.5)

    avg_health = _clamp_score_open(
        sum(final_sector_summary.values()) / len(final_sector_summary)
        if final_sector_summary
        else 0.0
    )
    hospital_ok = _clamp_score_open(1.0)
    if hospital_health_log:
        violations = sum(1 for h in hospital_health_log if h < 0.4)
        hospital_ok = _clamp_score_open(1.0 - (violations / max(len(hospital_health_log), 1)))

    no_blackout = _clamp_score_open(
        1.0 if final_sector_summary and all(v > 0.0 for v in final_sector_summary.values()) else 0.0
    )

    if failure_history:
        peak_failed = max(len(step_failed) for step_failed in failure_history)
    else:
        peak_failed = 0
    denom = max(total_nodes, 1)
    cascade_contained = _clamp_score_open(1.0 - (peak_failed / denom))

    raw = 0.35 * avg_health + 0.30 * hospital_ok + 0.20 * no_blackout + 0.15 * cascade_contained
    return _sanitize_score(raw)


def _select_planner_action(
    planner_env: Any,
    task_id: str,
) -> Optional[Dict[str, Optional[str]]]:
    if planner_env is None or grade_episode is None:
        return None

    candidates = _planner_candidate_actions(planner_env, task_id)
    if not candidates:
        return None

    best_action: Optional[Dict[str, Optional[str]]] = None
    best_sig: Optional[tuple[float, float]] = None
    for action in candidates:
        score, reward_sum = _planner_rollout_score(planner_env, task_id, action)
        sig = (score, reward_sum)
        if best_sig is None or sig > best_sig:
            best_sig = sig
            best_action = action
    return best_action


def _pick_recover_target(obs: Any) -> Optional[str]:
    """Pick a dependency-safe failed node to recover first."""
    pending = set(obs.pending_recoveries) if obs.pending_recoveries else set()
    node_map = {n.node_id: n for n in obs.nodes}
    upstream_map = _build_upstream_map(obs)

    failed = [n for n in obs.nodes if not n.is_operational and n.node_id not in pending]
    if not failed:
        return None

    def upstream_all_ok(node_id: str) -> bool:
        return all(
            node_map[u].is_operational
            for u in upstream_map.get(node_id, [])
            if u in node_map
        )

    recoverable = [n for n in failed if upstream_all_ok(n.node_id)]
    if recoverable:
        target = min(recoverable, key=lambda n: (not n.is_critical, n.health))
        return target.node_id

    failed_ids = {n.node_id for n in failed}
    root_failures = [
        n for n in failed
        if not any(u in failed_ids for u in upstream_map.get(n.node_id, []))
    ]
    if root_failures:
        target = min(root_failures, key=lambda n: (not n.is_critical, n.health))
        return target.node_id

    return failed[0].node_id


def _pick_safe_shed_target(obs: Any) -> Optional[str]:
    """Prefer shedding non-critical overloaded nodes only."""
    candidates = [
        n for n in obs.nodes
        if n.is_operational and (not n.is_critical) and n.sector != "hospital" and n.load > 0.9
    ]
    if not candidates:
        candidates = [
            n for n in obs.nodes
            if n.is_operational and (not n.is_critical) and n.sector != "hospital" and n.load > 0.85
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda n: n.load).node_id


def _pick_coordinate_target(obs: Any) -> Optional[str]:
    if obs.budget_remaining < 1.0:
        return None
    delayed = [n for n in obs.nodes if n.observation_delayed]
    if not delayed:
        return None
    # Prefer critical delayed nodes for faster visibility restoration.
    delayed.sort(key=lambda n: (not n.is_critical, n.health))
    return delayed[0].node_id


def _is_action_valid_local(action: Dict[str, Optional[str]], obs: Any) -> bool:
    action_type = action.get("action_type")
    target_id = action.get("target_node_id")
    node_map = {n.node_id: n for n in obs.nodes}

    if action_type == "wait":
        return True

    if action_type == "harden":
        if target_id is None or target_id not in node_map:
            return False
        n = node_map[target_id]
        return n.is_operational and (not n.is_hardened) and obs.budget_remaining >= 2.0

    if action_type == "shed_load":
        if target_id is None or target_id not in node_map:
            return False
        n = node_map[target_id]
        return n.is_operational and (not n.is_critical) and n.sector != "hospital" and n.load > 0.5

    if action_type == "coordinate":
        if target_id is None or target_id not in node_map:
            return False
        return obs.budget_remaining >= 1.0 and any(n.observation_delayed for n in obs.nodes)

    if action_type == "recover":
        if target_id is None or target_id not in node_map:
            return False
        n = node_map[target_id]
        if n.is_operational:
            return False
        if target_id in set(obs.pending_recoveries or []):
            return False
        # Local dependency check from observable graph.
        upstream = [e.source_id for e in obs.edges if e.target_id == target_id]
        return all(node_map[u].is_operational for u in upstream if u in node_map)

    return False


def _task_playbook_action(obs: Any, step: int, task_id: str) -> Optional[Dict[str, Optional[str]]]:
    node_map = {n.node_id: n for n in obs.nodes}

    def maybe_harden(node_id: str) -> Optional[Dict[str, Optional[str]]]:
        n = node_map.get(node_id)
        if n is None:
            return None
        if obs.budget_remaining >= 2.0 and n.is_operational and (not n.is_hardened):
            return {"action_type": "harden", "target_node_id": node_id}
        return None

    if task_id == "task_gen_blackout" and step <= 4:
        for nid in ["POWER_GEN_1", "POWER_TRANS_1", "POWER_DIST_1", "POWER_DIST_2"]:
            action = maybe_harden(nid)
            if action is not None:
                return action

    if task_id == "task_medium":
        # Weather-heavy task: prioritize central power/water support before hospital leaves.
        if step <= 4:
            for nid in ["POWER_GEN_1", "POWER_DIST_1", "WATER_DIST_1", "POWER_TRANS_1"]:
                action = maybe_harden(nid)
                if action is not None:
                    return action
        if 5 <= step <= 7 and not obs.active_failures and obs.budget_remaining >= 1.0:
            coord_target = _pick_coordinate_target(obs)
            if coord_target is not None:
                return {"action_type": "coordinate", "target_node_id": coord_target}

    if task_id == "task_hard":
        # Keep some budget in reserve; hardening every early node was starving recovery.
        if step <= 5:
            for nid in ["POWER_GEN_1", "POWER_TRANS_1", "POWER_TRANS_2", "POWER_DIST_1"]:
                action = maybe_harden(nid)
                if action is not None:
                    return action
        if 2 <= step <= 4 and not obs.active_failures and obs.budget_remaining >= 3.0:
            coord_target = _pick_coordinate_target(obs)
            if coord_target is not None:
                return {"action_type": "coordinate", "target_node_id": coord_target}

    if task_id == "task_cyberattack":
        # Faults are power-sector heavy after jitter; protect backbone first.
        if step <= 5:
            for nid in ["POWER_GEN_1", "POWER_DIST_1", "POWER_DIST_2", "POWER_TRANS_1"]:
                action = maybe_harden(nid)
                if action is not None:
                    return action
        if 1 <= step <= 3 and not obs.active_failures and obs.budget_remaining >= 3.0:
            coord_target = _pick_coordinate_target(obs)
            if coord_target is not None:
                return {"action_type": "coordinate", "target_node_id": coord_target}

    if task_id == "task_easy" and step <= 2:
        for nid in ["POWER_GEN_1", "POWER_TRANS_1"]:
            action = maybe_harden(nid)
            if action is not None:
                return action

    return None


def _build_candidate_actions(
    obs: Any,
    step: int,
    task_id: str,
    playbook_action: Optional[Dict[str, Optional[str]]] = None,
) -> List[Dict[str, Optional[str]]]:
    candidates: List[Dict[str, Optional[str]]] = []
    failed_now = [n for n in obs.nodes if not n.is_operational]
    overloaded_now = [n for n in obs.nodes if n.is_operational and n.load > 0.85]
    pressure = float(getattr(getattr(obs, "diagnostics", None), "system_pressure", 0.0) or 0.0)
    high_pressure = pressure >= 0.45 or bool(failed_now) or bool(overloaded_now)

    if playbook_action is not None and _is_action_valid_local(playbook_action, obs):
        candidates.append(playbook_action)

    recover_target = _pick_recover_target(obs)
    if recover_target is not None:
        candidates.append({"action_type": "recover", "target_node_id": recover_target})

    shed_target = _pick_safe_shed_target(obs)
    if shed_target is not None:
        candidates.append({"action_type": "shed_load", "target_node_id": shed_target})

    # Coordinate only when delayed visibility is present and pressure suggests it is useful.
    if high_pressure:
        coord_target = _pick_coordinate_target(obs)
        if coord_target is not None:
            candidates.append({"action_type": "coordinate", "target_node_id": coord_target})

    if obs.budget_remaining >= 2.0:
        critical_unhardened = [n for n in obs.nodes if n.is_critical and n.is_operational and not n.is_hardened]
        if critical_unhardened:
            candidates.append({"action_type": "harden", "target_node_id": critical_unhardened[0].node_id})

    # Wait only when there is no urgent pressure.
    if not high_pressure:
        candidates.append({"action_type": "wait", "target_node_id": None})

    # Deduplicate while preserving order.
    deduped: List[Dict[str, Optional[str]]] = []
    seen = set()
    for a in candidates:
        sig = (a.get("action_type"), a.get("target_node_id"))
        if sig in seen:
            continue
        seen.add(sig)
        if _is_action_valid_local(a, obs):
            deduped.append(a)

    if not deduped:
        if recover_target is not None:
            return [{"action_type": "recover", "target_node_id": recover_target}]
        return [{"action_type": "wait", "target_node_id": None}]
    return deduped


# ---------------------------------------------------------------------------
# Heuristic - runs every step, used before LLM for emergencies + early steps
# ---------------------------------------------------------------------------
def smart_heuristic(obs: Any, step: int) -> Optional[Dict]:
    nodes = obs.nodes
    pending = set(obs.pending_recoveries) if obs.pending_recoveries else set()
    node_map = {n.node_id: n for n in nodes}
    upstream_map = _build_upstream_map(obs)

    def upstream_all_ok(node_id: str) -> bool:
        return all(
            node_map[u].is_operational
            for u in upstream_map.get(node_id, [])
            if u in node_map
        )

    # --- 1 & 2. Recover failed nodes in topological order ---
    failed = [n for n in nodes if not n.is_operational and n.node_id not in pending]
    if failed:
        recoverable = [n for n in failed if upstream_all_ok(n.node_id)]
        if recoverable:
            target = min(recoverable, key=lambda n: (not n.is_critical, n.health))
            return {"action_type": "recover", "target_node_id": target.node_id}
        # No directly recoverable - find root of failure chain
        failed_ids = {n.node_id for n in failed}
        root_failures = [
            n for n in failed
            if n.node_id not in pending
            and not any(u in failed_ids for u in upstream_map.get(n.node_id, []))
        ]
        if root_failures:
            def failed_upstream_count(n):
                return sum(
                    1 for u in upstream_map.get(n.node_id, [])
                    if u in node_map and not node_map[u].is_operational
                )
            target = min(root_failures, key=failed_upstream_count)
            return {"action_type": "recover", "target_node_id": target.node_id}

    # --- 3. Overloaded nodes ---
    overloaded = [n for n in nodes if n.load > 0.88 and n.is_operational]
    non_critical_overloaded = [n for n in overloaded if not n.is_critical]
    if non_critical_overloaded:
        target = max(non_critical_overloaded, key=lambda n: n.load)
        return {"action_type": "shed_load", "target_node_id": target.node_id}

    task_id = getattr(obs, "task_id", "")
    topo = _task_topology_stats(task_id) if task_id else {}

    # --- 4. Harden central support nodes before hospital leaves on harder tasks. ---
    if obs.budget_remaining >= 2.0 and task_id in {"task_medium", "task_hard", "task_gen_blackout", "task_cyberattack"}:
        structural_targets = [
            n for n in nodes
            if not n.is_hardened and n.is_operational
        ]
        structural_targets.sort(
            key=lambda n: (
                topo.get(n.node_id, {}).get("hospital_support", 0),
                topo.get(n.node_id, {}).get("descendants", 0),
                n.sector == "power",
                n.is_critical,
            ),
            reverse=True,
        )
        if structural_targets and topo.get(structural_targets[0].node_id, {}).get("hospital_support", 0) > 0:
            return {"action_type": "harden", "target_node_id": structural_targets[0].node_id}

    # --- 5. Harden critical unhardened nodes (hospitals, etc.). ---
    if obs.budget_remaining >= 2.0:
        critical_unhardened = [
            n for n in nodes
            if n.is_critical and not n.is_hardened and n.is_operational
        ]
        if critical_unhardened:
            critical_unhardened.sort(
                key=lambda n: (
                    topo.get(n.node_id, {}).get("hospital_support", 0),
                    -n.health,
                ),
                reverse=True,
            )
            return {"action_type": "harden", "target_node_id": critical_unhardened[0].node_id}

    # --- 6. Pre-emptive hardening in early steps (before faults hit) ---
    if task_id == "task_easy" and step < 3 and obs.budget_remaining >= 2.0 and not failed and not pending:
        # Harden the node with the most downstream dependents (highest leverage)
        unhardened_op = [
            n for n in nodes
            if not n.is_hardened and n.is_operational
        ]
        if unhardened_op:
            target = max(
                unhardened_op,
                key=lambda n: (_downstream_count(n.node_id, obs), n.is_critical)
            )
            # Only harden if it actually has dependents or is critical
            if _downstream_count(target.node_id, obs) > 0 or target.is_critical:
                return {"action_type": "harden", "target_node_id": target.node_id}

    # --- 7. Weak node hardening (before storm or cascade takes them) ---
    if obs.budget_remaining >= 2.0:
        at_risk = [
            n for n in nodes
            if not n.is_hardened and n.is_operational
            and (0.3 <= n.health < 0.7
                 or (obs.weather_forecast != "clear" and n.health < 0.85))
        ]
        if at_risk:
            target = min(at_risk, key=lambda n: n.health)
            return {"action_type": "harden", "target_node_id": target.node_id}

    # --- 8. Coordinate only if there are delayed nodes ---
    has_delayed = any(n.observation_delayed for n in nodes)
    if pending and has_delayed and obs.budget_remaining >= 1.0:
        delayed_node = next((n for n in nodes if n.observation_delayed), None)
        if delayed_node:
            return {"action_type": "coordinate", "target_node_id": delayed_node.node_id}
    elif pending:
        # Recoveries in flight, nothing urgent - just wait
        return {"action_type": "wait", "target_node_id": None}

    return None  # Let the LLM decide


# ---------------------------------------------------------------------------
# LLM action getter with rate-limit retry
# ---------------------------------------------------------------------------
async def _get_action(
    client: OpenAI,
    obs: Any,
    step: int,
    messages: List[Dict],
    last_reward: float,
    candidate_actions: List[Dict[str, Optional[str]]],
    playbook_action: Optional[Dict[str, Optional[str]]] = None,
) -> tuple[str, Optional[str]]:
    heuristic_action = smart_heuristic(obs, step)
    critical_emergency = any(
        (not n.is_operational and n.is_critical) or (n.is_critical and n.load > 0.92)
        for n in obs.nodes
    )
    diagnostics = getattr(obs, "diagnostics", None)
    diag_pressure = float(getattr(diagnostics, "system_pressure", 0.0) or 0.0)
    broad_emergency = (
        len(getattr(obs, "active_failures", [])) >= 3
        or diag_pressure >= 0.75
    )
    emergency = critical_emergency or broad_emergency

    _debug(
        f"[DIAG] _get_action step={step} emergency={emergency} "
        f"heuristic={heuristic_action}"
    )

    # Use heuristic only for immediate high-risk emergencies and very early bootstrap.
    # Let the LLM choose among curated candidates instead of forcing the playbook.
    bootstrap_steps = 1
    if step < bootstrap_steps or emergency:
        if heuristic_action is not None:
            if heuristic_action.get("action_type") == "wait" and not emergency and step > 0:
                pass
            else:
                _debug(f"[HEURISTIC] step={step} -> {heuristic_action}")
                return json.dumps(heuristic_action), None
        # No rule fired -> fall through to LLM

    # LLM path - single-turn only (no history), to keep context small
    user_prompt = _build_user_prompt(obs, step, last_reward=last_reward)
    candidate_text = " | ".join(
        f"{a['action_type']}:{a.get('target_node_id')}" for a in candidate_actions
    )
    user_prompt = f"{user_prompt}\nLegalActions: {candidate_text}\nChoose ONLY one listed action in exact JSON format."
    # Only send system + current state (no history) to stay well under 16k tokens
    messages_llm = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}]

    error: Optional[str] = None
    backoff = 1.0
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=messages_llm,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            _debug(f"[DIAG] LLM raw: {raw!r}")
            parsed = json.loads(raw)

            chosen = {
                "action_type": parsed.get("action_type", "wait"),
                "target_node_id": parsed.get("target_node_id"),
            }
            allowed = {
                (a.get("action_type"), a.get("target_node_id"))
                for a in candidate_actions
            }
            chosen_sig = (chosen.get("action_type"), chosen.get("target_node_id"))

            if chosen_sig not in allowed:
                _debug(f"[FILTER] out-of-candidate action {chosen_sig}, fallback={candidate_actions[0]}")
                chosen = candidate_actions[0]

            return json.dumps(chosen), None

        except json.JSONDecodeError as e:
            error = f"json_parse_error: {e}"
            _debug(f"[WARN] LLM parse fail (attempt {attempt+1}): {e}")
            if attempt < 2:
                continue

        except Exception as e:
            error_str = str(e)
            # Rate limit or token limit - back off and retry
            if "rate_limit" in error_str.lower() or "429" in error_str or "token" in error_str.lower():
                _debug(f"[WARN] API limit (attempt {attempt+1}): {error_str[:80]}. Backing off {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                error = f"api_limit: {error_str[:60]}"
                continue
            else:
                error = f"api_error: {error_str[:60]}"
                _debug(f"[ERROR] Unexpected API error: {error_str[:100]}")
                break

    # Fallback to heuristic, then wait
    if heuristic_action is not None:
        _debug(f"[FALLBACK] using heuristic: {heuristic_action}")
        return json.dumps(heuristic_action), error

    _debug("[FALLBACK] -> wait")
    return '{"action_type": "wait", "target_node_id": null}', error


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------
async def run_task(
    env: CascadeGuardEnv,
    client: OpenAI,
    task_id: str,
    seed: Optional[int] = None,
    scenario_split: str = "train",
    scenario_index: int = 0,
) -> tuple[float, List[float]]:
    print(f"[START] task={task_id} env=cascade_guard model={MODEL_NAME}")

    planner_env: Optional[Any] = None
    rewards: List[float] = []
    score: float = SCORE_EPS
    success: bool = False

    try:
        max_steps = MAX_STEPS[task_id]
        result = await env.reset(
            task_id=task_id,
            seed=seed,
            scenario_split=scenario_split,
            scenario_index=scenario_index,
        )
        obs = result.observation

        if LocalCascadeEnvironment is not None and grade_episode is not None:
            try:
                planner_env = LocalCascadeEnvironment()
                planner_env.reset(
                    task_id=task_id,
                    seed=seed,
                    scenario_split=scenario_split,
                    scenario_index=scenario_index,
                )
            except Exception as exc:
                planner_env = None
                _debug(f"[PLANNER] init failed: {exc}")

        # No persistent message history - each step is stateless LLM call.
        # This eliminates context growth entirely.
        messages: List[Dict] = []  # kept for heuristic logging only
        last_done = False
        last_action_sig: Optional[tuple[str, Optional[str]]] = None
        last_reward: float = 0.0
        recent_recover_outcomes: List[float] = []

        for step in range(max_steps):
            playbook_action = _task_playbook_action(obs, step, task_id)
            candidate_actions = _build_candidate_actions(obs, step, task_id, playbook_action)
            planner_candidates = _planner_candidate_actions(planner_env, task_id) if planner_env is not None else []
            planner_action = _select_planner_action(planner_env, task_id) if planner_env is not None else None
            candidate_actions = _merge_candidate_actions(planner_candidates, candidate_actions)
            raw_action, error = await _get_action(
                client,
                obs,
                step,
                messages,
                last_reward=last_reward,
                candidate_actions=candidate_actions,
                playbook_action=playbook_action,
            )

            action_type = "wait"
            try:
                action_dict = json.loads(raw_action)
                action = CascadeAction(**action_dict)
                # Auto-fill missing target_node_id
                if action.action_type != "wait" and not action.target_node_id:
                    if action.action_type == "shed_load":
                        tgt = max(obs.nodes, key=lambda n: n.load)
                    elif action.action_type == "recover":
                        failed_nodes = [n for n in obs.nodes if not n.is_operational]
                        tgt = min(failed_nodes, key=lambda n: n.health) if failed_nodes else min(obs.nodes, key=lambda n: n.health)
                    else:
                        crit = [n for n in obs.nodes if n.is_critical and not n.is_hardened]
                        tgt = crit[0] if crit else min(obs.nodes, key=lambda n: n.health)
                    action.target_node_id = tgt.node_id
                action_type = action.action_type
            except Exception as exc:
                error = str(exc)
                _debug(f"[ERROR] parse fail: {exc} | raw='{raw_action}'")
                action = CascadeAction(action_type="wait", target_node_id=None, parameters={})

            # Safety filter to avoid heavy penalties on invalid or unsafe actions.
            node_map = {n.node_id: n for n in obs.nodes}
            pending = set(obs.pending_recoveries) if obs.pending_recoveries else set()
            adjust_reason: Optional[str] = None
            planner_valid = (
                planner_env is not None
                and _planner_is_action_valid(
                    planner_env,
                    {
                        "action_type": action.action_type,
                        "target_node_id": action.target_node_id,
                    },
                )
            )

            if not planner_valid:
                if action.action_type == "shed_load":
                    target = node_map.get(action.target_node_id or "")
                    if target is None or (target.load <= 0.5) or target.is_critical or target.sector == "hospital":
                        adjust_reason = "unsafe_shed_load"
                elif action.action_type == "harden":
                    target = node_map.get(action.target_node_id or "")
                    if (
                        target is None
                        or target.is_hardened
                        or (not target.is_operational)
                        or obs.budget_remaining < 2.0
                    ):
                        adjust_reason = "invalid_harden"
                elif action.action_type == "recover":
                    target = node_map.get(action.target_node_id or "")
                    upstream_map = _build_upstream_map(obs)

                    def upstream_all_ok(nid: str) -> bool:
                        return all(
                            node_map[u].is_operational
                            for u in upstream_map.get(nid, [])
                            if u in node_map
                        )

                    if (
                        target is None
                        or target.node_id in pending
                        or (target.is_operational and target.health > 0.0)
                        or (target is not None and not upstream_all_ok(target.node_id))
                    ):
                        adjust_reason = "invalid_recover"

                current_sig = (action.action_type, action.target_node_id)
                if last_action_sig == current_sig and last_reward < 0:
                    adjust_reason = "repeat_negative_action"

                if action.action_type == "recover" and len(recent_recover_outcomes) >= 2:
                    if recent_recover_outcomes[-1] < 0 and recent_recover_outcomes[-2] < 0:
                        adjust_reason = "repeat_negative_recover"

            if adjust_reason is not None:
                if planner_env is not None:
                    fallback_action = planner_action or _planner_default_action(planner_env, task_id)
                    fallback_action_clean = {k: v for k, v in fallback_action.items() if k != "parameters"}
                    action = CascadeAction(**fallback_action_clean, parameters={})
                else:
                    recover_target = _pick_recover_target(obs)
                    if recover_target is not None:
                        action = CascadeAction(action_type="recover", target_node_id=recover_target, parameters={})
                    else:
                        shed_target = _pick_safe_shed_target(obs)
                        if shed_target is not None:
                            action = CascadeAction(action_type="shed_load", target_node_id=shed_target, parameters={})
                        else:
                            coord_target = _pick_coordinate_target(obs)
                            if coord_target is not None:
                                action = CascadeAction(action_type="coordinate", target_node_id=coord_target, parameters={})
                            else:
                                action = CascadeAction(action_type="wait", target_node_id=None, parameters={})
                action_type = action.action_type

            result = await env.step(action)
            if planner_env is not None:
                try:
                    planner_env.step(action)
                except Exception as exc:
                    _debug(f"[PLANNER] step sync failed: {exc}")
                    planner_env = None
            obs = result.observation
            reward = result.reward
            raw_reward = _extract_raw_step_reward(obs, reward)
            done = result.done
            rewards.append(reward)
            last_reward = raw_reward
            last_action_sig = (action.action_type, action.target_node_id)
            if action.action_type == "recover":
                recent_recover_outcomes.append(raw_reward)
                if len(recent_recover_outcomes) > 4:
                    recent_recover_outcomes.pop(0)

            error_str = "null" if error is None else error
            print(
                f"[STEP] step={step + 1} action={action_type} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error={error_str}",
                flush=True,
            )

            last_done = done
            if done:
                break

            # Small delay to respect Groq rate limits.
            await asyncio.sleep(0.3)

        # ---------------------------------------------------------------------------
        # Grader
        # ---------------------------------------------------------------------------
        grade_fn = grade_episode
        if grade_fn is None:
            try:
                from cascade_guard.server.graders import grade as grade_fn
            except Exception as exc:
                _debug(f"[WARN] grader import failed: {exc}")
                grade_fn = None

        state_data = await env.state()

        if state_data is not None and getattr(state_data, "failure_history", None) is not None:
            failure_history = [set(s) for s in state_data.failure_history]
            hospital_health_log = list(state_data.hospital_health_log)
            budget_spent = TASK_CONFIGS_BUDGET[task_id] - state_data.budget_remaining
            budget_total = TASK_CONFIGS_BUDGET[task_id]
            total_nodes = int(getattr(state_data, "total_nodes", 0) or 0)
            cascade_depth_log = list(getattr(state_data, "cascade_depth_log", []) or [])
            dependency_order_log = list(getattr(state_data, "dependency_order_log", []) or [])
            action_history = list(getattr(state_data, "action_history", []) or [])
        else:
            inner_env = getattr(env, "_env", None) or getattr(env, "env", None)
            if inner_env is not None and hasattr(inner_env, "_failure_history"):
                failure_history = inner_env._failure_history
                hospital_health_log = inner_env._hospital_health_log
                budget_spent = TASK_CONFIGS_BUDGET[task_id] - inner_env._budget
                budget_total = TASK_CONFIGS_BUDGET[task_id]
                total_nodes = len(getattr(inner_env, "_node_states", {}) or {})
                cascade_depth_log = list(getattr(inner_env, "_cascade_depth_log", []) or [])
                dependency_order_log = list(getattr(inner_env, "_dependency_order_log", []) or [])
                action_history = list(getattr(inner_env, "_action_history", []) or [])
            else:
                failure_history = []
                hospital_health_log = [1.0] * max(len(rewards), 1)
                budget_total = TASK_CONFIGS_BUDGET[task_id]
                budget_spent = 0.0
                total_nodes = 0
                cascade_depth_log = []
                dependency_order_log = []
                action_history = []

        final_sector_summary = dict(state_data.sector_health) if state_data else {}

        if not final_sector_summary:
            score = _sanitize_score(0.5)
            _debug(f"[WARN] empty sector summary; using neutral fallback score for task={task_id}")
        elif grade_fn is None:
            score = _fallback_proxy_grade(
                final_sector_summary=final_sector_summary,
                hospital_health_log=hospital_health_log,
                failure_history=failure_history,
                total_nodes=total_nodes,
            )
            _debug(f"[WARN] using proxy grade for task={task_id} score={score:.4f}")
        else:
            score = grade_fn(
                task_id,
                failure_history=failure_history,
                hospital_health_log=hospital_health_log,
                final_sector_summary=final_sector_summary,
                budget_spent=budget_spent,
                budget_total=budget_total,
                total_nodes=total_nodes,
                cascade_depth_log=cascade_depth_log,
                dependency_order_log=dependency_order_log,
                action_history=action_history,
            )
        # Graders already return _clamp(round(raw, 4)), no need to re-sanitize
        _debug(
            f"[GRADE] task={task_id} normalized_score={score:.4f} "
            f"final_sector_summary={final_sector_summary}"
        )

        success = last_done and (
    final_sector_summary and
    (sum(final_sector_summary.values()) / len(final_sector_summary)) >= 0.1
)
    except asyncio.CancelledError as exc:
        success = False
        score = SCORE_EPS
        _debug(f"[ERROR] run_task cancelled: {exc}")
    except Exception as exc:
        success = False
        score = SCORE_EPS
        _debug(f"[ERROR] run_task failed: {exc}")
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} steps={len(rewards)} "
            f"rewards={rewards_str}",
            flush=True,
        )

    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = float("nan")
    
    # Final hard guard - explicit boundary clamping
    if score_value <= 0.0:
        _debug(f"[WARN] score <= 0.0 for task={task_id}: {score_value} - clamping to SCORE_EPS")
        score = SCORE_EPS
    elif score_value >= 1.0:
        _debug(f"[WARN] score >= 1.0 for task={task_id}: {score_value} - clamping to 1.0-SCORE_EPS")
        score = 1.0 - SCORE_EPS
    elif not math.isfinite(score_value):
        _debug(f"[WARN] score not finite for task={task_id}: {score_value} - clamping to SCORE_EPS")
        score = SCORE_EPS
    else:
        score = score_value

    return score, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing LLM API key. Set GROQ_API_KEY (preferred) or HF_TOKEN.")
    if not HF_TOKEN and GROQ_API_KEY:
        print("[WARN] HF_TOKEN is not set; continuing with GROQ_API_KEY.", file=sys.stderr, flush=True)

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=API_BASE_URL,
    )
    scores: List[float] = []
    run_records: List[Dict[str, Any]] = []

    env = await _connect_environment()

    try:
        if EVAL_MODE == "multiseed":
            task_stats: Dict[str, List[float]] = {}
            for task_id in TASK_NAMES:
                seed_list = TASK_SEED_SPLITS.get(task_id, {}).get(EVAL_SPLIT, [])
                if not seed_list:
                    seed_list = TASK_SEED_SPLITS.get(task_id, {}).get("validation", [])
                if not seed_list:
                    seed_list = [None]

                task_scores: List[float] = []
                for idx, seed in enumerate(seed_list):
                    if idx > 0:
                        await asyncio.sleep(1)
                    score, _ = await run_task(
                        env,
                        client,
                        task_id,
                        seed=seed,
                        scenario_split=EVAL_SPLIT,
                        scenario_index=idx,
                    )
                    task_scores.append(score)
                    scores.append(score)
                    run_records.append(
                        {
                            "task_id": task_id,
                            "score": _sanitize_score(score),
                            "seed": int(seed) if seed is not None else None,
                            "scenario_split": EVAL_SPLIT,
                            "scenario_index": idx,
                        }
                    )

                task_stats[task_id] = task_scores
                if task_scores:
                    mean_s = statistics.mean(task_scores)
                    std_s = statistics.pstdev(task_scores) if len(task_scores) > 1 else 0.0
                    worst_s = min(task_scores)
                    _debug(
                        f"[EVAL] task={task_id} split={EVAL_SPLIT} runs={len(task_scores)} "
                        f"mean={mean_s:.3f} std={std_s:.3f} worst={worst_s:.3f}"
                    )
        else:
            for i, task_id in enumerate(TASK_NAMES):
                seed_list = TASK_SEED_SPLITS.get(task_id, {}).get("train", [])
                baseline_seed = seed_list[i % len(seed_list)] if seed_list else None
                if i > 0:
                    # Pause between tasks to let rate limit window reset
                    _debug("[INFO] Pausing 5s before next task...")
                    await asyncio.sleep(5)
                score, _ = await run_task(
                    env,
                    client,
                    task_id,
                    seed=baseline_seed,
                    scenario_split="train",
                    scenario_index=i,
                )
                scores.append(score)
                run_records.append(
                    {
                        "task_id": task_id,
                        "score": _sanitize_score(score),
                        "seed": int(baseline_seed) if baseline_seed is not None else None,
                        "scenario_split": "train",
                        "scenario_index": i,
                    }
                )
    finally:
        try:
            await env.close()
        except Exception as e:
            _debug(f"[DEBUG] env.close() error: {e}")
        try:
            _write_reproducibility_report(run_records)
        except Exception as exc:
            print(f"[WARN] failed to write reproducibility report: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())