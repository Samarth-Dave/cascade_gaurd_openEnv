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
W_HOSPITAL   = 0.35
W_CASCADE    = 0.25
W_BUDGET     = 0.20
W_CENTRALITY = 0.20

INIT_BUDGET  = 10_000  # dollar-equivalent label (env uses float units internally)


def compute_reward(
    sector_health: Dict[str, float],
    cascade_depth: int,
    budget_remaining: float,
    init_budget: float,
    node_states: Dict[str, Dict],
    centrality: Dict[str, float],
    base_action_reward: float = 0.0,
) -> float:
    """
    Multi-component reward — identical contract to training reward.

    Components
    ----------
    hospital_health  (0.35)  Protect hospitals above all else.
    cascade_penalty  (0.25)  Penalise cascade spread (depth proxy).
    budget_efficiency(0.20)  Reward budget conservation.
    centrality_defended(0.20) Reward protecting high-centrality nodes.

    Returns
    -------
    float in [-1.0, 1.0]
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
        W_HOSPITAL   * hospital_score   +
        W_CASCADE    * cascade_score    +
        W_BUDGET     * budget_score     +
        W_CENTRALITY * centrality_score +
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
    GRPO reward verifier — called by TRL GRPOTrainer.

    Evaluates the quality of the LLM's response given the resulting
    environment state after the action was executed.

    Args
    ----
    completion    : LLM's full response text (may include <think>…</think>).
    obs           : Observation dict AFTER the action was executed.
                    Expected keys: sector_health, cascade_depth, budget,
                    node_states (each with "status"), centrality (optional).
    action_taken  : Parsed action dict with at least {"action_type": str}.
    init_budget   : Starting budget for normalisation.

    Returns
    -------
    Scalar reward for GRPO update in [-1.0, 1.0].
    """
    # Chain-of-thought bonus
    has_cot   = "<think>" in completion and "</think>" in completion
    cot_bonus = 0.03 if has_cot else 0.0

    # Build centrality map from obs if available
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
    )

    # Action-quality bonuses
    action_type = action_taken.get("action_type", "wait")
    if action_type in ("repair", "recover") and obs.get("cascade_depth", 1) == 0:
        base += 0.05   # Repair resolved the cascade
    elif action_type in ("isolate", "emergency_shutdown") and obs.get("cascade_depth", 5) < 2:
        base += 0.03   # Isolation contained cascade
    elif action_type == "harden":
        base += 0.02   # Proactive hardening is always good

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

