"""
cot_prompt.py — Chain-of-Thought system prompt for CascadeGuard
================================================================
v2 improvements:
  - Legal action list appended to every prompt (reduces invalid action rate)
  - Geo/radius awareness: agent understands service coverage
  - Narrative phase in prompt header
  - Strict JSON action format for GRPO training (reduces parse errors)
  - CoT-before-action enforced for better small-model reasoning

API compatibility notes (verified against models.py):
  - EdgeStatus.active (not .is_active)
  - CascadeObservation.pending_recoveries is List[str]
  - NodeStatus.observation_delayed (not .is_delayed)
  - NodeStatus.observation_confidence — v2 field (0.0–1.0)
  - CascadeObservation.narrative_phase — v2 field
  - CascadeObservation.radius_coverage_events — v2 field
"""

from __future__ import annotations
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cascade_guard.models import CascadeAction, CascadeObservation


SYSTEM_PROMPT = """\
You are an AI infrastructure coordinator managing a cross-sector critical infrastructure network.
Your goal is to keep power, water, hospital, and telecom systems operational during cascading failures.

## Network Rules

**Dependency order is mandatory:**
- You CANNOT recover a node until ALL its upstream suppliers are operational.
- Example: WATER_TREAT_1 depends on POWER_DIST_1. Recover POWER_DIST_1 FIRST.
- Wrong order = wasted action + penalty.

**Priority hierarchy:**
1. Hospital nodes (HOSP_*, EMERG_*) — protect at all costs
2. High-centrality backbone nodes (GEN > TRANS > DIST > leaf nodes)
3. Nodes that unlock multiple failed downstream dependents

**Geo / Radius Coverage:**
- Each node has a service radius. If a power plant (50km radius) is operational,
  it partially protects hospitals within that radius against cascade damage.
- Keeping telecom towers operational maintains observation confidence.
  Without telecom coverage, node health readings become noisy.

**Actions available:**
- harden(NODE_ID): costs 2.0 budget. Reduces failure threshold. Do BEFORE storm events.
- recover(NODE_ID): restores a failed node. Only works if upstream is operational.
- isolate(NODE_ID): temporarily stops a failed node from damaging downstream dependents.
- shed_load(NODE_ID): reduces load on NON-critical nodes only. Only valid when load > 0.5.
- coordinate(NODE_ID): clears observation delay. Use when confidence is low.
- wait(null): ONLY valid when there is genuinely nothing actionable.
  WARNING: Waiting during active failures escalates cascade debt and triggers extra penalties.

**Budget management:**
- Do NOT exhaust budget early on low-impact actions.
- Harden high-centrality nodes BEFORE a storm arrives.
- Save budget for recoveries after fault events.

## Response Format

You MUST respond in EXACTLY this format (two parts, nothing else):
<think>
1. Current phase: [narrative phase]
2. Critical failures: [list failed nodes]
3. Upstream dependencies for each failed: [trace chain]
4. Hospital status: [health values]
5. At-risk nodes: [from diagnostics]
6. Budget: [remaining] — what can I afford?
7. Radius coverage: [which nodes provide backup]
8. Best action: [reason why this is optimal vs waiting]
</think>
<action>ACTION_TYPE(TARGET_NODE_OR_null)</action>

Examples:
<action>recover(POWER_GEN_1)</action>
<action>isolate(POWER_GEN_1)</action>
<action>harden(POWER_TRANS_1)</action>
<action>coordinate(WATER_PUMP_1)</action>
<action>shed_load(POWER_DIST_2)</action>
<action>wait(null)</action>
"""

# Strict JSON format for GRPO training (better for small models)
SYSTEM_PROMPT_GRPO = """\
You are an AI managing critical infrastructure (power, water, hospital, telecom).
Keep all nodes operational. Hospitals are highest priority.

RULES:
- Recover nodes in dependency order (upstream first).
- Harden nodes before storms hit. Each harden costs 2.0 budget.
- DO NOT wait when there are active failures — this causes cascade debt.
- Telecom coverage maintains sensor confidence. Protect telecom nodes.
- If recovery order is blocked, isolate a failed upstream node to stop cascade spread.

Decision rule:
- system_pressure < 0.2: harden or coordinate.
- 0.2 <= system_pressure < 0.5: recover failed nodes in dependency order, then harden.
- system_pressure >= 0.5: CRISIS - recover failed nodes immediately, never wait.
- If budget < 1.0 and failures exist: coordinate, shed_load, or isolate when legal.

Think step-by-step before acting:
<think>1. Failures: [list] | 2. At-risk: [list] | 3. Best action: [reason]</think>

Then output ONLY one action tag:
<action>ACTION_TYPE(TARGET_OR_null)</action>

Legal action types: harden, recover, isolate, shed_load, coordinate, wait
"""


def build_system_prompt(grpo_mode: bool = False) -> str:
    """Return the system prompt. Use grpo_mode=True for training (stricter format)."""
    return SYSTEM_PROMPT_GRPO if grpo_mode else SYSTEM_PROMPT


def build_user_prompt(obs: "CascadeObservation") -> str:
    """
    Convert a CascadeObservation into a structured markdown prompt for the LLM.

    v2: includes narrative phase, observation confidence, radius events.
    """
    lines: list = []

    # Header
    narrative = getattr(obs, "narrative_phase", "pre_storm")
    consec    = getattr(obs, "consecutive_waits", 0)
    stress_level = getattr(obs, "upcoming_stress_level", "none")
    if stress_level == "imminent":
        lines.append("STRESS IMMINENT: harden now to reduce damage.")
    elif stress_level == "soon":
        lines.append("Stress event approaching in about 5-8 steps.")
    lines.append(
        f"## [{narrative.upper()}] Step {obs.step} / {obs.max_steps} "
        f"| Budget: {obs.budget_remaining:.1f}"
    )
    lines.append(f"Weather: {obs.weather_forecast}")
    if consec >= 2:
        lines.append(f"WARNING: Consecutive waits: {consec} — cascade debt is accumulating!")
    lines.append("")

    # Node table
    lines.append("## Node Status")
    lines.append("| Node | Sector | Health | Load | Operational | Hardened | Delayed | Confidence |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n in sorted(obs.nodes, key=lambda x: x.node_id):
        delayed    = getattr(n, "observation_delayed", False)
        confidence = getattr(n, "observation_confidence", 1.0)
        conf_str   = f"{confidence:.2f}" if confidence < 1.0 else "1.00"
        op_str     = "YES" if n.is_operational else "**FAILED**"
        lines.append(
            f"| {n.node_id} | {n.sector} | {n.health:.2f} | "
            f"{n.load:.2f} | {op_str} | "
            f"{'yes' if n.is_hardened else 'no'} | "
            f"{'YES' if delayed else 'no'} | {conf_str} |"
        )

    # Dependency edges
    if obs.edges:
        lines.append("")
        lines.append("## Dependency Graph (source -> target)")
        for e in obs.edges:
            status = "OK" if e.active else "BROKEN"
            lines.append(f"  {e.source_id} -> {e.target_id} [{status}]")

    # Active failures / pending recoveries
    if obs.active_failures:
        lines.append("")
        lines.append(f"## FAILURES: {', '.join(obs.active_failures)}")
    if obs.pending_recoveries:
        lines.append("")
        lines.append("## Pending Recoveries (in progress):")
        for node_id in obs.pending_recoveries:
            lines.append(f"  - {node_id}")

    # Diagnostics
    if obs.diagnostics:
        d = obs.diagnostics
        if d.critical_failures:
            lines.append("")
            lines.append(f"## CRITICAL FAILURES: {', '.join(d.critical_failures)}")
        if d.at_risk_nodes:
            lines.append(f"## At Risk: {', '.join(d.at_risk_nodes)}")
        if d.dependency_alerts:
            lines.append(f"## Dependency Alerts: {'; '.join(d.dependency_alerts)}")
        if d.recommended_recovery_order:
            lines.append(
                f"## Recommended Recovery Order: {' -> '.join(d.recommended_recovery_order)}"
            )
        if d.system_pressure:
            lines.append(f"## System Pressure: {d.system_pressure:.2f}")

    # Radius coverage events (v2)
    radius_events = getattr(obs, "radius_coverage_events", [])
    if radius_events:
        lines.append("")
        lines.append("## Radius Coverage Active:")
        for evt in radius_events[:4]:
            lines.append(f"  - {evt}")

    # Sector summary
    lines.append("")
    lines.append("## Sector Health Summary")
    for sector, health in sorted(obs.sector_summary.items()):
        filled = int(health * 10)
        bar    = "#" * filled + "." * (10 - filled)
        lines.append(f"  {sector:12s}: {health:.2f} [{bar}]")

    lines.append("")
    lines.append("Choose your action. Think step-by-step, then respond in the required format.")
    return "\n".join(lines)


def make_training_prompt(obs: "CascadeObservation", env=None, grpo_mode: bool = True) -> str:
    """
    Build a full training prompt including system + user + legal action list.

    The legal action list dramatically reduces invalid actions from the 71% seen
    in baseline runs. It shows the model exactly what moves are valid right now.
    """
    if isinstance(env, bool):
        grpo_mode = env
        env = None

    system = build_system_prompt(grpo_mode=grpo_mode)
    user   = build_user_prompt(obs)

    legal = []
    if env is not None and hasattr(env, "get_legal_actions"):
        for action in env.get_legal_actions():
            action_type = action.get("action_type", "wait")
            target = action.get("target_node_id")
            legal.append(f"{action_type}({target if target else 'null'})")
    else:
        pending_set = set(obs.pending_recoveries)

        for n in obs.nodes:
            if not n.is_operational and n.node_id not in pending_set:
                legal.append(f"recover({n.node_id})")
                legal.append(f"isolate({n.node_id})")
            if n.is_operational and not n.is_hardened and obs.budget_remaining >= 2.0:
                legal.append(f"harden({n.node_id})")
            if (n.is_operational
                    and not n.is_critical
                    and n.sector != "hospital"
                    and n.load > 0.5):
                legal.append(f"shed_load({n.node_id})")

        delayed_nodes = [n for n in obs.nodes if getattr(n, "observation_delayed", False)]
        if delayed_nodes and obs.budget_remaining >= 1.0:
            legal.append(f"coordinate({delayed_nodes[0].node_id})")

        legal.append("wait(null)")

    legal_str  = " | ".join(legal[:12])
    legal_line = f"\n\nLegal actions this step: {legal_str}"

    return system + "\n\n" + user + legal_line


def parse_action_from_response(response: str, obs: Optional["CascadeObservation"]) -> "CascadeAction":
    """
    Parse <action>ACTION_TYPE(TARGET)</action> from LLM response.
    Falls back to JSON format, then plain text, then wait() on any failure.
    """
    from cascade_guard.models import CascadeAction

    VALID_ACTIONS = {"harden", "shed_load", "coordinate", "recover", "isolate", "wait"}

    # 1. Try strict XML-tag format first
    strict_pattern = r"<action>\s*(\w+)\(([^)]*)\)\s*</action>"
    match = re.search(strict_pattern, response, re.IGNORECASE)

    # 2. Fallback: JSON format {"action_type": ..., "target_node_id": ...}
    if not match:
        json_pattern = r'"action_type"\s*:\s*"(\w+)".*?"target_node_id"\s*:\s*(?:"([^"]*)"|(null))'
        jmatch = re.search(json_pattern, response, re.IGNORECASE | re.DOTALL)
        if jmatch:
            action_type = jmatch.group(1).lower()
            raw_target  = jmatch.group(2) or ""
            target: Optional[str] = raw_target if raw_target and raw_target.lower() not in ("", "null", "none") else None
            if action_type in VALID_ACTIONS:
                return CascadeAction(action_type=action_type, target_node_id=target)

    # 3. Fallback: plain function-call anywhere in response
    if not match:
        loose_pattern = r"\b(harden|recover|isolate|shed_load|coordinate|wait)\(([^)]*)\)"
        match = re.search(loose_pattern, response, re.IGNORECASE)

    if not match:
        return CascadeAction(action_type="wait", target_node_id=None)

    action_type = match.group(1).lower()
    raw_target  = match.group(2).strip()
    target_parsed: Optional[str] = (
        raw_target
        if raw_target and raw_target.lower() not in ("", "null", "none")
        else None
    )

    if action_type not in VALID_ACTIONS:
        return CascadeAction(action_type="wait", target_node_id=None)

    # Validate target node against observation
    if target_parsed is not None and obs is not None:
        valid_nodes = {n.node_id for n in obs.nodes}
        if target_parsed not in valid_nodes:
            hits = [nid for nid in valid_nodes if nid.upper() == target_parsed.upper()]
            target_parsed = hits[0] if hits else None

    return CascadeAction(action_type=action_type, target_node_id=target_parsed)
