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
    from models import CascadeAction, CascadeObservation


ADVANCED_ACTION_TYPES = (
    "reroute",
    "prioritize",
    "deploy_repair_crew",
    "emergency_shutdown",
    "cross_sector_bridge",
    "patch_scada",
    "redistribute_load",
    "request_mutual_aid",
    "controlled_cascade",
    "multi_sector_lockdown",
)

CORE_ACTION_TYPES = (
    "recover",
    "harden",
    "coordinate",
    "shed_load",
    "isolate",
    "wait",
)


SYSTEM_PROMPT = """\
You are an AI infrastructure coordinator managing a real-world cross-sector critical infrastructure network.
Your goal is to keep power, water, hospital, and telecom systems operational during cascading failures.

Each node in the network corresponds to a REAL infrastructure site (e.g. "KEM Hospital Parel",
"Tata Power Trombay Plant", "BT Tower"). Node IDs are shown alongside their real names.
Prioritise named hospitals above all other nodes.

## Network Rules

**Dependency order is mandatory:**
- You CANNOT recover a node until ALL its upstream suppliers are operational.
- Example: Worli Water Treatment Plant depends on Dadar Distribution Centre. Recover power FIRST.
- Wrong order = wasted action + penalty.

**Priority hierarchy:**
1. Hospital nodes (HOSP_*, EMERG_*) — protect at all costs (real patients depend on them)
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
- reroute(SOURCE,TARGET): redirects healthy supply from SOURCE to TARGET (0.6 budget).
- prioritize(NODE_ID): +0.15 health + 1-step stress immunity; cooldown 3 steps.
- deploy_repair_crew(NODE_ID): halve repair time on a node already in recovery (1.2 budget).
- emergency_shutdown(NODE_ID): controlled shutdown when health < 30% (0.2 budget).
  Tier 3 hint: use when a dying node would cascade; beats letting it fail uncontrolled.
- cross_sector_bridge(SECTOR_A,SECTOR_B): 3-step cross-sector redundancy link (1.5 budget).
  Tier 3 hint: bridge telecom\u2192hospital during hospital crisis to restore observability.
- patch_scada(NODE_ID): stops active SCADA anomaly drain (0.8 budget).
  Tier 3 hint: use immediately in task_cyberattack once SCADA drain starts; every step of delay compounds.
- redistribute_load(NODE_A,NODE_B): moves 0.2 load from overloaded to underloaded same-sector node (0.5 budget).
  Tier 3 hint: use before a node hits 0.9 load to prevent overload cascade.
- request_mutual_aid(SECTOR): +0.2 health to all nodes in sector; ONCE per episode (2.5 budget).
  Tier 3 hint: emergency reserve \u2014 only when a full sector is in unrecoverable freefall.
- controlled_cascade(NODE_ID): sacrifice node takes -0.5 health; neighbors gain +0.15 (0 budget, grader penalty).
  Tier 4 hint: pick the node with the fewest downstream dependents and lowest centrality.
- multi_sector_lockdown(null): freeze entire system 2 steps \u2014 no cascade, no recovery (2.0 budget).
  Tier 4 hint: last resort when simultaneous multi-sector failure is spiraling out of control.

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
6. Budget: [remaining] \u2014 what can I afford?
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
<action>reroute(POWER_GEN_2,HOSP_MAIN)</action>
<action>prioritize(HOSP_MAIN)</action>
<action>deploy_repair_crew(POWER_GEN_1)</action>
<action>emergency_shutdown(WATER_TREAT_2)</action>
<action>cross_sector_bridge(telecom,hospital)</action>
<action>patch_scada(POWER_GEN_1)</action>
<action>redistribute_load(POWER_DIST_1,POWER_DIST_2)</action>
<action>request_mutual_aid(power)</action>
<action>controlled_cascade(POWER_LEAF_1)</action>
<action>multi_sector_lockdown(null)</action>
<action>wait(null)</action>
"""

# Strict JSON format for GRPO training (better for small models)
SYSTEM_PROMPT_GRPO = """\
You are an AI managing critical infrastructure (power, water, hospital, telecom).
Keep all nodes operational. Hospitals are highest priority.

RULES:
- Recover nodes in dependency order (upstream first).
- Harden nodes before storms hit. Each harden costs 2.0 budget.
- DO NOT wait when there are active failures \u2014 this causes cascade debt.
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

Legal action types: harden, recover, isolate, shed_load, coordinate, wait,
  reroute, prioritize, deploy_repair_crew, emergency_shutdown,
  cross_sector_bridge, patch_scada, redistribute_load,
  request_mutual_aid, controlled_cascade, multi_sector_lockdown

Tier 3/4 hints (use only when situation demands it):
- emergency_shutdown: when a node health < 30% would cascade uncontrolled.
- cross_sector_bridge: when a sector loses observability during crisis.
- patch_scada: in cyberattack task, use as soon as SCADA drain starts.
- redistribute_load: before any node hits 0.9 load to prevent overload failure.
- request_mutual_aid: only when a full sector is in unrecoverable freefall.
- controlled_cascade: sacrifice the least-central node to stabilise neighbors.
- multi_sector_lockdown: absolute last resort during simultaneous multi-sector spiral.
"""


SYSTEM_PROMPT_GRPO_ACTION_ONLY = """\
You are an AI coordinator for critical infrastructure (power, water, hospital, telecom).
Choose exactly one valid action for the current step.

Hard constraints:
- Do NOT output chain-of-thought.
- Do NOT output explanations, markdown, bullets, or extra text.
- Output exactly one line and nothing else.
- The one line MUST be exactly one action tag:
  <action>ACTION_TYPE(TARGET_OR_null)</action>

Use only legal action types:
harden, recover, isolate, shed_load, coordinate, wait,
reroute, prioritize, deploy_repair_crew, emergency_shutdown,
cross_sector_bridge, patch_scada, redistribute_load,
request_mutual_aid, controlled_cascade, multi_sector_lockdown

Examples of valid one-line outputs:
<action>recover(POWER_GEN_1)</action>
<action>patch_scada(TELECOM_SWITCH_1)</action>
<action>cross_sector_bridge(telecom,hospital)</action>
<action>wait(null)</action>
"""


def build_system_prompt(grpo_mode: bool = False, action_only: bool = False) -> str:
    """Return the system prompt. action_only applies to GRPO mode only."""
    if grpo_mode and action_only:
        return SYSTEM_PROMPT_GRPO_ACTION_ONLY
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
        lines.append(f"WARNING: Consecutive waits: {consec} \u2014 cascade debt is accumulating!")
    lines.append("")

    # Node table — show real_name alongside node_id when available
    lines.append("## Node Status")
    lines.append("| Node ID | Real Name | Sector | Health | Load | Operational | Hardened | Delayed | Conf |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for n in sorted(obs.nodes, key=lambda x: x.node_id):
        delayed    = getattr(n, "observation_delayed", False)
        confidence = getattr(n, "observation_confidence", 1.0)
        conf_str   = f"{confidence:.2f}" if confidence < 1.0 else "1.00"
        op_str     = "YES" if n.is_operational else "**FAILED**"
        real_name  = getattr(n, "real_name", "") or ""
        name_col   = real_name if real_name else "-"
        lines.append(
            f"| {n.node_id} | {name_col} | {n.sector} | {n.health:.2f} | "
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


def _legal_action_to_call(action: dict) -> str:
    action_type = str(action.get("action_type", "wait"))
    params = action.get("parameters") or {}

    if action_type == "reroute":
        source = params.get("source") or action.get("target_node_id") or "SOURCE"
        target = params.get("target") or "TARGET"
        return f"reroute({source},{target})"

    if action_type == "cross_sector_bridge":
        sector_a = params.get("sector_a") or action.get("target_node_id") or "SECTOR_A"
        sector_b = params.get("sector_b") or "SECTOR_B"
        return f"cross_sector_bridge({sector_a},{sector_b})"

    if action_type == "redistribute_load":
        node_a = params.get("node_a") or action.get("target_node_id") or "NODE_A"
        node_b = params.get("node_b") or "NODE_B"
        return f"redistribute_load({node_a},{node_b})"

    if action_type == "request_mutual_aid":
        sector = params.get("sector") or action.get("target_node_id") or "SECTOR"
        return f"request_mutual_aid({sector})"

    target = action.get("target_node_id")
    return f"{action_type}({target if target else 'null'})"


def _summarize_legal_actions(legal_actions: list[dict], max_examples: int = 30) -> str:
    if not legal_actions:
        return "\n\nLegal action types now: wait(1)\nRepresentative legal actions: wait(null)"

    by_type: dict[str, list[str]] = {}
    type_counts: dict[str, int] = {}
    for action in legal_actions:
        action_type = str(action.get("action_type", "wait"))
        type_counts[action_type] = type_counts.get(action_type, 0) + 1
        call = _legal_action_to_call(action)
        bucket = by_type.setdefault(action_type, [])
        if call not in bucket:
            bucket.append(call)

    ordered_types: list[str] = []
    for action_type in CORE_ACTION_TYPES:
        if action_type in by_type:
            ordered_types.append(action_type)
    for action_type in ADVANCED_ACTION_TYPES:
        if action_type in by_type and action_type not in ordered_types:
            ordered_types.append(action_type)
    for action_type in sorted(by_type.keys()):
        if action_type not in ordered_types:
            ordered_types.append(action_type)

    prioritized_types = [a for a in ADVANCED_ACTION_TYPES if a in by_type]
    prioritized_types.extend([a for a in ordered_types if a not in prioritized_types])

    examples: list[str] = []
    for action_type in prioritized_types:
        for call in by_type[action_type][:2]:
            if call not in examples:
                examples.append(call)
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    types_line = ", ".join(f"{name}({type_counts[name]})" for name in ordered_types)
    sample_line = " | ".join(examples) if examples else "wait(null)"
    advanced_legal = [a for a in ADVANCED_ACTION_TYPES if a in by_type]

    legal_summary = (
        f"\n\nLegal action types now: {types_line}"
        f"\nRepresentative legal actions (sampled across action types): {sample_line}"
    )
    if advanced_legal:
        legal_summary += "\nAdvanced actions currently legal: " + ", ".join(advanced_legal)
    return legal_summary


def make_training_prompt(
    obs: "CascadeObservation",
    env=None,
    grpo_mode: bool = True,
    action_only: bool = False,
) -> str:
    """
    Build a full training prompt including system + user + legal action list.

    The legal action list dramatically reduces invalid actions from the 71% seen
    in baseline runs. It shows the model exactly what moves are valid right now.
    """
    # Back-compat: older callers passed grpo_mode as the second positional arg
    if isinstance(env, bool):
        grpo_mode = env
        env = None

    system = build_system_prompt(grpo_mode=grpo_mode, action_only=action_only)
    user   = build_user_prompt(obs)
    if grpo_mode and action_only:
        user += (
            "\n\nSTRICT OUTPUT CONTRACT: Output exactly one line with a single "
            "<action>...</action> tag and no extra text."
        )

    legal = []
    if env is not None and hasattr(env, "get_legal_actions"):
        legal_actions = env.get_legal_actions()
        legal_line = _summarize_legal_actions(legal_actions)
        return system + "\n\n" + user + legal_line
    else:
        pending_set = set(obs.pending_recoveries)

        for n in obs.nodes:
            if not n.is_operational and n.node_id not in pending_set:
                legal.append(f"recover({n.node_id})")
                legal.append(f"isolate({n.node_id})")
                if n.health < 0.30 and obs.budget_remaining >= 0.2:
                    legal.append(f"emergency_shutdown({n.node_id})")
            if n.is_operational and not n.is_hardened and obs.budget_remaining >= 2.0:
                legal.append(f"harden({n.node_id})")
            if (n.is_operational
                    and not n.is_critical
                    and n.sector != "hospital"
                    and n.load > 0.5):
                legal.append(f"shed_load({n.node_id})")
            if n.is_operational:
                legal.append(f"prioritize({n.node_id})")
                legal.append(f"controlled_cascade({n.node_id})")
            if n.node_id in pending_set and obs.budget_remaining >= 1.2:
                legal.append(f"deploy_repair_crew({n.node_id})")

        delayed_nodes = [n for n in obs.nodes if getattr(n, "observation_delayed", False)]
        if delayed_nodes and obs.budget_remaining >= 1.0:
            legal.append(f"coordinate({delayed_nodes[0].node_id})")

        if obs.budget_remaining >= 0.5:
            legal.append("redistribute_load(NODE_A,NODE_B)")
        if obs.budget_remaining >= 0.6:
            legal.append("reroute(SOURCE,TARGET)")
        if obs.budget_remaining >= 0.8:
            legal.append("patch_scada(NODE_ID)")
        if obs.budget_remaining >= 1.5:
            legal.append("cross_sector_bridge(SECTOR_A,SECTOR_B)")
        if obs.budget_remaining >= 2.5:
            legal.append("request_mutual_aid(SECTOR)")
        if obs.budget_remaining >= 2.0:
            legal.append("multi_sector_lockdown(null)")

        legal.append("wait(null)")

    legal_str = " | ".join(legal[:30])
    legal_line = f"\n\nLegal actions this step: {legal_str}"

    return system + "\n\n" + user + legal_line


def parse_action_from_response(response: str, obs: Optional["CascadeObservation"]) -> "CascadeAction":
    """
    Parse <action>ACTION_TYPE(TARGET)</action> from LLM response.
    Falls back to JSON format, then plain text, then wait() on any failure.
    """
    from models import CascadeAction

    VALID_ACTIONS = {
        "harden", "shed_load", "coordinate", "recover", "isolate", "wait",
        "reroute", "prioritize", "deploy_repair_crew", "emergency_shutdown",
        "cross_sector_bridge", "patch_scada", "redistribute_load",
        "request_mutual_aid", "controlled_cascade", "multi_sector_lockdown",
    }

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
        loose_pattern = (
            r"\b(harden|recover|isolate|shed_load|coordinate|wait"
            r"|reroute|prioritize|deploy_repair_crew|emergency_shutdown"
            r"|cross_sector_bridge|patch_scada|redistribute_load"
            r"|request_mutual_aid|controlled_cascade|multi_sector_lockdown)\(([^)]*)\)"
        )
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

    parameters: dict = {}
    if target_parsed and "," in target_parsed:
        parts = [p.strip() for p in target_parsed.split(",")]
        if action_type == "reroute" and len(parts) >= 2:
            parameters = {"source": parts[0], "target": parts[1]}
            target_parsed = parts[0]
        elif action_type == "cross_sector_bridge" and len(parts) >= 2:
            parameters = {"sector_a": parts[0], "sector_b": parts[1]}
            target_parsed = parts[0]
        elif action_type == "redistribute_load" and len(parts) >= 2:
            parameters = {"node_a": parts[0], "node_b": parts[1]}
            target_parsed = parts[0]

    # Validate target node against observation
    # Multi-parameter actions define their own valid forms, so we whitelist them.
    if target_parsed is not None and obs is not None and action_type not in ("cross_sector_bridge", "request_mutual_aid", "redistribute_load", "reroute"):
        valid_nodes = {n.node_id for n in obs.nodes}
        if target_parsed not in valid_nodes:
            hits = [nid for nid in valid_nodes if nid.upper() == target_parsed.upper()]
            target_parsed = hits[0] if hits else None

    return CascadeAction(action_type=action_type, target_node_id=target_parsed, parameters=parameters)