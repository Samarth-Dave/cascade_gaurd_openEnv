"""
reward.py — CascadeGuard standalone reward module + GRPO verifier
=================================================================
Extracted from CascadeEnvironment so training scripts and external
evaluators can import the reward logic without instantiating the full env.

Usage:
    from reward import compute_reward, grpo_verifier
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Weights (must match CascadeEnvironment._compute_reward)
# ---------------------------------------------------------------------------
W_HOSPITAL      = 0.30   # was 0.35 — still highest priority
W_CASCADE       = 0.20   # was 0.25
W_BUDGET        = 0.15   # was 0.20
W_CENTRALITY    = 0.15   # was 0.20
W_ACTION        = 0.20   # NEW — weight for action-quality score (creates GRPO variance)

INIT_BUDGET     = 10_000  # dollar-equivalent label (env uses float units internally)

# ── Validity penalty magnitudes ───────────────────────────────────────────────
# These are subtracted BEFORE clamping to [-1, 1].
# Sized so that within a GRPO group, parse-fail gets a notably lower reward
# than a syntactically valid action even if the env state is identical.
PENALTY_PARSE_FAIL    = 0.30   # output couldn't be parsed at all → wait
PENALTY_UNKNOWN_TGT   = 0.20   # action type valid but target_node_id is null/unknown
PENALTY_ALREADY_DONE  = 0.12   # harden on already-hardened, recover on healthy node
PENALTY_COOLDOWN      = 0.10   # action on cooldown
PENALTY_BUDGET        = 0.08   # insufficient budget
BONUS_PARSE_OK        = 0.05   # clean parse bonus — rewards good format even if env unchanged
BONUS_VALID_ACTION    = 0.05   # env accepted the action without error


def compute_reward(
    sector_health: Dict[str, float],
    cascade_depth: int,
    budget_remaining: float,
    init_budget: float,
    node_states: Dict[str, Dict],
    centrality: Dict[str, float],
    base_action_reward: float = 0.0,
    action_quality: float = 0.0,   # NEW — per-completion validity/quality score
) -> float:
    """
    Multi-component reward.

    action_quality is a [-1, 0] penalty or [0, +0.1] bonus injected by
    grpo_verifier based on parse success and action validity. This is the
    component that creates within-group reward VARIANCE for GRPO to learn from.
    """
    hospital_score = sector_health.get("hospital", 1.0)
    cascade_score  = 1.0 - min(1.0, cascade_depth / 5.0)
    budget_score   = budget_remaining / max(1.0, init_budget)

    total_cent     = sum(centrality.values()) or 1.0
    defended_cent  = sum(
        c for nid, c in centrality.items()
        if node_states.get(nid, {}).get("status") not in ("critical", "isolated")
    )
    centrality_score = defended_cent / total_cent

    composite = (
        W_HOSPITAL   * hospital_score    +
        W_CASCADE    * cascade_score     +
        W_BUDGET     * budget_score      +
        W_CENTRALITY * centrality_score  +
        W_ACTION     * action_quality    +   # NEW — per-completion variance
        base_action_reward
    )
    return round(max(-1.0, min(1.0, composite)), 4)


def compute_episode_score(reward_history: List[float]) -> float:
    """Mean reward over episode — the hackathon metric."""
    if not reward_history:
        return 0.0
    return round(sum(reward_history) / len(reward_history), 4)


def grpo_verifier(
    completion: str,
    obs: Dict,
    action_taken: Dict,
    init_budget: float = 10_000.0,
) -> float:
    """
    GRPO reward verifier — rewritten to produce within-group variance.

    The critical change: we now compute a `validity_score` entirely from
    the parsed action_taken dict. Different completions produce different
    actions → different validity_scores → non-zero GRPO advantages → learning.

    action_taken must include:
        action_type   : str   e.g. "harden", "wait", "recover"
        target_node_id: str | None
        invalid_reason: str   e.g. "", "unknown_target", "already_hardened",
                              "cooldown_active", "insufficient_budget",
                              "parse_failed", "node_not_failed"
        action_valid  : bool  (True if env accepted the action)
    """

    # ── 1. Parse quality ──────────────────────────────────────────────────────
    parse_failed   = action_taken.get("invalid_reason", "") == "parse_failed" \
                     or action_taken.get("parse_failed", False)
    invalid_reason = action_taken.get("invalid_reason", "")
    action_valid   = action_taken.get("action_valid", False)
    action_type    = action_taken.get("action_type", "wait")
    target         = action_taken.get("target_node_id", None)

    # ── 2. Compute validity_score ─────────────────────────────────────────────
    # This is the component that differs BETWEEN completions in the same group.
    if parse_failed:
        validity_score = -PENALTY_PARSE_FAIL           # -0.30: truncated / garbled output

    elif invalid_reason == "unknown_target":
        validity_score = -PENALTY_UNKNOWN_TGT          # -0.20: null or hallucinated node

    elif invalid_reason == "already_hardened":
        validity_score = -PENALTY_ALREADY_DONE         # -0.12: repeated no-op

    elif invalid_reason == "cooldown_active":
        validity_score = -PENALTY_COOLDOWN             # -0.10: action on cooldown

    elif invalid_reason == "insufficient_budget":
        validity_score = -PENALTY_BUDGET               # -0.08: can't afford

    elif invalid_reason == "node_not_failed":
        validity_score = -PENALTY_ALREADY_DONE         # -0.12: recover on healthy node

    elif not action_valid and invalid_reason:
        # Any other env rejection
        validity_score = -0.08

    elif action_valid:
        # Base parse success + valid action bonus
        validity_score = BONUS_PARSE_OK + BONUS_VALID_ACTION   # +0.10

        # ── 3. Strategic bonuses (valid actions only) ─────────────────────────
        cascade_depth = obs.get("cascade_depth", 0)
        pressure      = obs.get("system_pressure", 0.0)

        # Repairing after cascade resolves — high value
        if action_type in ("recover", "repair") and cascade_depth == 0:
            validity_score += 0.08

        # Isolation / shutdown when cascade is shallow — contained it
        if action_type in ("isolate", "emergency_shutdown") and cascade_depth < 2:
            validity_score += 0.06

        # Hardening proactively before pressure builds
        if action_type == "harden" and pressure < 0.3:
            validity_score += 0.04

        # Patch SCADA / deploy crew — expensive but correct
        if action_type in ("patch_scada", "deploy_repair_crew"):
            validity_score += 0.05

        # Shed load when pressure is high — correct reactive action
        if action_type == "shed_load" and pressure >= 0.5:
            validity_score += 0.04

        # Penalise wait when cascade is deepening — passive under pressure
        if action_type == "wait" and pressure >= 0.6:
            validity_score -= 0.08

    else:
        validity_score = 0.0   # neutral — shouldn't happen but safe fallback

    # ── 4. Chain-of-thought bonus ─────────────────────────────────────────────
    has_cot   = "<think>" in completion and "</think>" in completion
    cot_bonus = 0.03 if has_cot else 0.0

    # ── 5. Base env-state reward (same for all completions in a group) ────────
    node_states = obs.get("node_states", {})
    centrality  = {
        nid: d.get("centrality", 0.0)
        for nid, d in node_states.items()
        if isinstance(d, dict)
    }

    base = compute_reward(
        sector_health    = obs.get("sector_health", {}),
        cascade_depth    = obs.get("cascade_depth", 0),
        budget_remaining = obs.get("budget", init_budget),
        init_budget      = init_budget,
        node_states      = node_states,
        centrality       = centrality,
        action_quality   = validity_score,   # injected per-completion score
    )

    return round(max(-1.0, min(1.0, base + cot_bonus)), 4)


def parse_action_from_completion(text: str) -> Dict:
    """
    Parse LLM output text into an action dict.

    Supports formats (checked in priority order):
        1. <action>TYPE(TARGET)</action>         ← primary training format
        2. ACTION: TYPE(TARGET)                  ← alternative plain text
        3. action_type: recover  target_node_id: HOSP_1  ← key-value lines
    """
    import re

    VALID_ACTIONS = {
        "harden", "shed_load", "coordinate", "recover", "isolate", "wait",
        "reroute", "prioritize", "deploy_repair_crew", "emergency_shutdown",
        "cross_sector_bridge", "patch_scada", "redistribute_load",
        "request_mutual_aid", "controlled_cascade", "multi_sector_lockdown",
    }

    def _build_action(atype: str, raw_args: str) -> Dict:
        atype = atype.lower().strip()
        if atype not in VALID_ACTIONS:
            return {"action_type": "wait"}
        args = [a.strip() for a in raw_args.split(",") if a.strip() and a.strip().lower() != "null"]
        if atype in ("reroute",) and len(args) >= 2:
            return {"action_type": atype, "parameters": {"source": args[0], "target": args[1]},
                    "target_node_id": args[0]}
        if atype == "cross_sector_bridge" and len(args) >= 2:
            return {"action_type": atype, "parameters": {"sector_a": args[0], "sector_b": args[1]},
                    "target_node_id": args[0]}
        if atype == "redistribute_load" and len(args) >= 2:
            return {"action_type": atype, "parameters": {"node_a": args[0], "node_b": args[1]},
                    "target_node_id": args[0]}
        if atype == "request_mutual_aid" and args:
            return {"action_type": atype, "parameters": {"sector": args[0]}, "target_node_id": args[0]}
        if atype in ("wait", "multi_sector_lockdown"):
            return {"action_type": atype, "target_node_id": None}
        return {"action_type": atype, "target_node_id": args[0] if args else None}

    # Format 1: <action>TYPE(args)</action>  ← used by cot_prompt.py
    m = re.search(r'<action>\s*(\w+)\(([^)]*)\)\s*</action>', text, re.IGNORECASE)
    if m:
        return _build_action(m.group(1), m.group(2))

    # Format 2: ACTION: TYPE(args)
    m = re.search(r'ACTION[:\s]+([A-Z_]+)\(?([^)\n]*)\)?', text, re.IGNORECASE)
    if m:
        return _build_action(m.group(1), m.group(2))

    # Format 3: key-value lines
    at_m = re.search(r'action_type[:\s]+(\w+)', text, re.IGNORECASE)
    tn_m = re.search(r'target_node_id[:\s]+(\w+)', text, re.IGNORECASE)
    if at_m:
        result: Dict = {"action_type": at_m.group(1).lower()}
        if tn_m:
            result["target_node_id"] = tn_m.group(1)
        return result

    return {"action_type": "wait"}

