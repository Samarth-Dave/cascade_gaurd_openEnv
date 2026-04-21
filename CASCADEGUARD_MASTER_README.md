# CascadeGuard — Master README for AI Hand-off
## Meta × HuggingFace OpenEnv Hackathon 2026 | Themes 2.3.1 + 1.1

> **Purpose of this document:** Feed this entire file to any AI assistant (Claude, GPT-4, Gemini, etc.) and it will have full context to continue building CascadeGuard. Every gap, every improvement, every file change, and every run command is here.

---

## 0. What CascadeGuard IS (One-Paragraph Pitch for Judges)

CascadeGuard is an OpenEnv reinforcement-learning environment that simulates cascading failures across power, water, hospital, and telecom sectors. An AI agent acts as the infrastructure coordinator and must harden nodes, recover failed ones in the correct dependency order, shed load intelligently, and coordinate delayed observations — all under a finite budget and imperfect observability. It directly maps to **Theme 2.3.1 (AI for Infrastructure Resilience)** and **Theme 1.1 (Adversarial AI / Attacker-Defender)** by including a persistent attacker (`inject_attack`) that fires cyber-physical events the defender agent must survive.

---

## 1. Repository File Map

```
cascade_guard/
├── models.py                  # Pydantic data models (NodeStatus, CascadeAction, etc.)
├── tasks.py                   # 5 task configs + seed splits + materialize_task_config()
├── cascade_environment.py     # Core OpenEnv Environment — step/reset/state physics
├── graders.py                 # Deterministic per-task scorers, all output strictly (0.01, 0.99)
├── cot_prompt.py              # Chain-of-Thought system prompt + build_user_prompt + parser
├── inference.py               # Baseline runner (LLM + heuristic + planner hybrid)
├── grpo_train.py              # GRPO training script (Unsloth + TRL)
├── curriculum_scheduler.py    # Adaptive curriculum: easy → hard, graduates on rolling mean
├── trajectory_logger.py       # Per-episode JSON + markdown comparison report
└── plot_curves.py             # reward_log.csv → matplotlib PNG figures
```

---

## 2. Current State — What Already Works ✅

| Component | Status | Notes |
|---|---|---|
| 5-task environment (easy → cyberattack) | ✅ Done | All tasks have seed splits, stress schedules, partial observability |
| Cascade physics (Bug 1–6 fixed) | ✅ Done | Threshold update before cascade, pre-step snapshot, correct terminal check |
| Graders (all 5, deterministic, (0,1) invariant) | ✅ Done | `grade()` dispatch + double `_safe()` fence |
| OpenEnv step/reset/state API | ✅ Done | Accepts `task_id`, `seed`, `scenario_split` in reset kwargs |
| Adversarial attacker hook (`inject_attack`) | ✅ Done | Fires after defender picks action, before physics commit |
| Heuristic + planner hybrid in inference | ✅ Done | Smart rollout lookahead, playbook, safety filter |
| Curriculum scheduler | ✅ Done | Graduates on rolling mean ≥ threshold |
| Trajectory logger | ✅ Done | JSON per episode + markdown comparison |
| Reward: dense + terminal + clamped to (0,1) | ✅ Done | `raw_reward` also exposed in metadata |
| Background load dynamics | ✅ Done | Mean-reverting stochastic load drift |
| GRPO training scaffold | ✅ Done | Unsloth + TRL, rollout_episode, CoT prompts |
| Plot curves | ✅ Done | reward_curve.png + per_task.png |

---

## 3. Gaps Found (Must Fix for First Place) 🔴

### 3.1 Environment Dynamism — Currently Mostly Static

**Problem:** `tasks.py` uses `_jitter_schedule()` which applies only ±1 step jitter and 35% target swap. The node graph (edges, node count, budget) is **completely fixed** per task regardless of seed. Judges explicitly look for "not static" environments.

**Fix Required in `tasks.py` → `materialize_task_config()`:**
```python
# After _jitter_schedule and _sample_observability, also call:
_sample_graph_variation(cfg, rng)   # NEW FUNCTION (see section 5.1)
_sample_budget_variation(cfg, rng)  # NEW FUNCTION (see section 5.2)
```

**New function `_sample_graph_variation(cfg, rng)`:**
- For task_hard and task_cyberattack: with 30% probability, add 1 extra telecom node `TELECOM_3` and its edge
- For task_medium: with 20% probability, add a second water treatment path `WATER_TREAT_2 → WATER_DIST_1`
- This makes the topology genuinely seed-dependent, not just schedule-dependent

**New function `_sample_budget_variation(cfg, rng)`:**
- Apply ±10% multiplicative jitter to budget per seed: `cfg["budget"] *= rng.uniform(0.9, 1.1)`
- Round to 1 decimal place

### 3.2 Attacker Agent is a Stub — Not a Real Agent

**Problem:** `inject_attack()` exists in `cascade_environment.py` but there is NO actual attacker policy. It is only called if the training loop manually fires it. Judges looking at Theme 1.1 (Attacker-Defender) will expect a real autonomous attacker.

**Fix Required — New file `adversarial_attacker.py`:**
```python
class AdversarialAttacker:
    """
    Rule-based attacker that observes the defender's last action and
    picks the maximally damaging counter-move.
    
    Strategy:
    - If defender just hardened node X → attack a different high-centrality node
    - If defender is recovering node X → fault a downstream of X to undo recovery value
    - If defender waited → attack the most critical unhardened node
    - If budget pressure on defender is high → escalate storm (weather event)
    """
    
    def __init__(self, attack_budget: float = 5.0, aggression: float = 0.5):
        self.attack_budget = attack_budget
        self.aggression = aggression   # 0.0 = passive, 1.0 = always attack
    
    def pick_attack(self, env_state: dict, defender_last_action: str) -> Optional[dict]:
        """
        Returns an attack_event dict or None.
        attack_event = {"type": "equipment_fault", "target": "NODE_ID", "effect": -0.2}
        """
        ...
    
    def escalate(self, env_state: dict) -> Optional[dict]:
        """Trigger weather escalation as an attacker move."""
        ...
```

**Usage in training loop (grpo_train.py):**
```python
attacker = AdversarialAttacker(attack_budget=5.0, aggression=0.6)
obs = env.reset(task_id="task_cyberattack", seed=seed)
for step in range(max_steps):
    action = defender_model.act(obs)
    attack = attacker.pick_attack(env.state.__dict__, action.action_type)
    env.inject_attack(attack)          # fires BEFORE env.step
    obs = env.step(action)
    done = obs.done
```

### 3.3 CLI Output is Sparse — Judges Can't Follow What's Happening

**Problem:** `inference.py` only prints `[START]`, `[STEP]`, and `[END]` lines. There's no visual context about what the network looks like or why an action was taken.

**Fix Required — New function `print_step_summary()` in `inference.py`:**
```python
def print_step_summary(step: int, obs, action, reward: float, task_id: str):
    """
    Prints a clean, readable step summary to stdout.
    Example output:
    
    ━━━ Step 3/10 | task_easy | Budget: 3.0 | Weather: clear ━━━
    NETWORK:  power=0.72 | water=N/A | hospital=N/A
    FAILURES: POWER_DIST_1
    ACTION:   recover(POWER_DIST_1)  [reward: +0.42]
    ALERT:    POWER_GEN_1 at risk (health=0.31)
    """
    sector_str = " | ".join(f"{k}={v:.2f}" for k, v in obs.sector_summary.items())
    fail_str = ", ".join(obs.active_failures) if obs.active_failures else "none"
    target = action.target_node_id or "—"
    print(f"\n{'━'*55}")
    print(f"  Step {step}/{obs.max_steps} | {task_id} | Budget: {obs.budget_remaining:.1f} | Weather: {obs.weather_forecast}")
    print(f"  NETWORK:  {sector_str}")
    print(f"  FAILURES: {fail_str}")
    print(f"  ACTION:   {action.action_type}({target})  [reward: {reward:+.3f}]")
    diag = getattr(obs, "diagnostics", None)
    if diag and diag.at_risk_nodes:
        print(f"  AT RISK:  {', '.join(diag.at_risk_nodes[:3])}")
    print(f"{'━'*55}")
```

Call this in `run_task()` inside the step loop, replacing the current terse `[STEP]` print.

### 3.4 CoT Prompt Agent Response Quality — Think Block Not Actually Used in Inference

**Problem:** `cot_prompt.py` defines a beautiful `<think>...</think><action>...</action>` format but `inference.py` uses a completely different compact JSON system prompt and never uses the CoT parser. The two are disconnected.

**Fix Required:** Add a `COT_MODE` flag to `inference.py`:
```python
COT_MODE: bool = os.getenv("COT_MODE", "0") == "1"
```

When `COT_MODE=1`, use `build_system_prompt()` and `build_user_prompt(obs)` from `cot_prompt.py`, then parse with `parse_action_from_response()`. This makes the CoT infrastructure actually used and demonstrates the think-then-act reasoning pattern to judges.

In `_get_action()`, branch:
```python
if COT_MODE:
    from cascade_guard.training.cot_prompt import (
        build_system_prompt, build_user_prompt as cot_build_user, parse_action_from_response
    )
    system = build_system_prompt()
    user = cot_build_user(obs)
    max_tok = 512   # CoT needs more tokens for <think> block
else:
    system = SYSTEM_PROMPT
    user = _build_user_prompt(obs, step, last_reward)
    max_tok = MAX_TOKENS
```

### 3.5 GRPO Training — `collect_rollouts()` is Incomplete (Truncated in File)

**Problem:** `grpo_train.py` has `collect_rollouts()` that is cut off — the function body is missing the return statement and reward broadcasting logic. This means training will crash.

**Fix Required — Complete `collect_rollouts()` in `grpo_train.py`:**
```python
def collect_rollouts(
    env: CascadeEnvironment,
    model,
    tokenizer,
    task_id: str,
    n_episodes: int,
    system_prompt: str,
    seed_list: List[int],
    device: str = "cuda",
) -> Tuple[List[str], List[str], List[float]]:
    """
    Collect n_episodes rollouts.
    Returns (all_prompts, all_responses, all_rewards).
    Reward is broadcast: every step in an episode gets the episode-level grader score.
    """
    all_prompts: List[str] = []
    all_responses: List[str] = []
    all_rewards: List[float] = []
    
    for ep_idx in range(n_episodes):
        seed = seed_list[ep_idx % len(seed_list)]
        score, prompts, responses = rollout_episode(
            env, model, tokenizer, task_id, seed, system_prompt, device=device
        )
        # Broadcast episode score as reward for every step
        all_prompts.extend(prompts)
        all_responses.extend(responses)
        all_rewards.extend([score] * len(prompts))
    
    return all_prompts, all_responses, all_rewards
```

### 3.6 Grader: `_centrality_protection_score` Ignores Action History

**Problem:** `_centrality_protection_score()` takes `action_history` as a parameter but never uses it — the function only looks at `failure_history`. The docstring claims it rewards the agent for protecting backbone nodes, but it only penalises for failing them. "Protection" (proactive hardening) is never measured.

**Fix Required in `graders.py`:**
```python
def _centrality_protection_score(
    action_history: List[str],
    high_centrality_nodes: List[str],
    failure_history: List[Set[str]],
) -> float:
    if not high_centrality_nodes or not failure_history:
        return _safe(0.5)

    all_failed: Set[str] = set()
    for s in failure_history:
        all_failed.update(s)

    central_failed = sum(1 for n in high_centrality_nodes if n in all_failed)
    protected_ratio = 1.0 - (central_failed / max(len(high_centrality_nodes), 1))
    
    # NEW: bonus for proactive hardening actions
    harden_actions = sum(1 for a in action_history if a == "harden")
    proactive_bonus = min(0.15, 0.03 * harden_actions)  # capped at 0.15
    
    return _clamp(protected_ratio + proactive_bonus)
```

### 3.7 `cascade_environment.py` — SCADA Recurring Damage is Buggy

**Problem:** The SCADA recurring loop iterates over `self._stress_schedule` every step and applies damage once per matching event. If there are TWO scada_anomaly events marked `recurring`, both fire every step from `scada_start` onwards. Also, `self._scada_start` is set to the LAST recurring event found (loop overwrites), not the first.

**Fix Required in `cascade_environment.py`:**
```python
# In reset(), fix scada_start detection:
self._scada_start = None
for step_n, event in sorted(self._stress_schedule.items()):
    if event.get("type") == "scada_anomaly" and event.get("recurring"):
        self._scada_start = step_n
        self._scada_event = event   # Store the event, not just the step
        break  # Take the FIRST one only

# In step(), replace the current SCADA loop with:
if self._scada_start is not None and self._step >= self._scada_start:
    t = self._scada_event.get("target")
    if t and t in self._node_states:
        ns = self._node_states[t]
        if ns.health > 0.0:
            ns.health = max(0.0, ns.health + self._scada_event["effect"])
```

### 3.8 Observation Delayed Nodes Not Shown in `_build_observation` Correctly

**Problem:** `_build_observation` pushes `(ns.health, ns.load)` into `_obs_buffer` AFTER using it to build the node list. This means the FIRST observation of a delayed node always shows current state (buffer is empty), only showing stale data from step 2 onwards. The buffer fill should happen BEFORE building the observation.

**Fix Required in `cascade_environment.py`:**
Move the buffer fill loop BEFORE the node_list building loop:
```python
def _build_observation(self) -> CascadeObservation:
    # STEP 1: First update obs_buffer with current true state
    for nid, ns in self._node_states.items():
        buf = self._obs_buffer.setdefault(nid, [])
        buf.append((ns.health, ns.load))
        if len(buf) > 4:
            buf.pop(0)
    
    # STEP 2: Then build observed node list (buffer has correct history now)
    node_list: List[NodeStatus] = []
    for nid, ns in self._node_states.items():
        is_delayed = self._is_node_delayed(nid)
        if is_delayed:
            buf = self._obs_buffer.get(nid, [])
            if len(buf) >= 2:
                health_obs, load_obs = buf[-2]   # 1-step stale
            else:
                health_obs, load_obs = ns.health, ns.load
        else:
            health_obs, load_obs = ns.health, ns.load
        ...
```

---

## 4. Feature Additions for Winning (Theme Alignment) 🟡

### 4.1 Real-Time Attacker-Defender Score Tracking (Theme 1.1)

Add to `graders.py` a new public function:
```python
def grade_adversarial(
    defender_score: float,
    attacker_damage_total: float,   # sum of health damage the attacker dealt
    attacker_budget_spent: float,
    attacker_budget_total: float,
) -> dict:
    """
    Returns a dict with both attacker and defender scores for multi-agent evaluation.
    attacker_score = damage_dealt / budget_spent (efficiency)
    defender_score = passed through _safe()
    """
    attacker_efficiency = _clamp(attacker_damage_total / max(attacker_budget_spent, 0.1))
    return {
        "defender_score": _safe(defender_score),
        "attacker_efficiency": _safe(attacker_efficiency),
        "combined": _safe(0.7 * defender_score - 0.3 * attacker_efficiency),
    }
```

### 4.2 Add `task_surge_demand` — New 6th Task (Judges Love Task Variety)

Add to `tasks.py` a new task that uses **no equipment faults**, but instead uses load surges (hospital emergency + telecom overload) to test the agent's `shed_load` + `coordinate` skills specifically. This shows load dynamics are a real mechanic, not just background noise.

```python
"task_surge_demand": {
    "task_id": "task_surge_demand",
    "seed": 555,
    "max_steps": 18,
    "budget": 7.0,
    "nodes": [
        # power (4)
        {"node_id": "POWER_GEN_1",  "sector": "power",    "is_critical": False},
        {"node_id": "POWER_DIST_1", "sector": "power",    "is_critical": False},
        {"node_id": "POWER_DIST_2", "sector": "power",    "is_critical": False},
        # hospital (3)
        {"node_id": "HOSP_1",       "sector": "hospital", "is_critical": True},
        {"node_id": "HOSP_2",       "sector": "hospital", "is_critical": True},
        {"node_id": "EMERG_1",      "sector": "hospital", "is_critical": False},
        # telecom (2)
        {"node_id": "TELECOM_1",    "sector": "telecom",  "is_critical": False},
        {"node_id": "TELECOM_SWITCH_1", "sector": "telecom", "is_critical": False},
    ],
    "edges": [
        {"source_id": "POWER_GEN_1",  "target_id": "POWER_DIST_1"},
        {"source_id": "POWER_GEN_1",  "target_id": "POWER_DIST_2"},
        {"source_id": "POWER_DIST_1", "target_id": "HOSP_1"},
        {"source_id": "POWER_DIST_2", "target_id": "HOSP_2"},
        {"source_id": "POWER_DIST_1", "target_id": "EMERG_1"},
        {"source_id": "POWER_DIST_1", "target_id": "TELECOM_1"},
        {"source_id": "TELECOM_1",    "target_id": "TELECOM_SWITCH_1"},
    ],
    # No equipment_fault events — pure demand surge via load dynamics + weather
    "stress_schedule": {
        3:  {"type": "weather", "target": None, "effect": "storm_warning"},
        7:  {"type": "weather", "target": None, "effect": "extreme_storm"},
        12: {"type": "weather", "target": None, "effect": "clear"},
    },
    "delayed_sectors": ["telecom"],
    "partial_obs_nodes": [],
    "description": (
        "No equipment faults — tests pure load management and coordination. "
        "Storm-driven overload forces shed_load on telecom/power, coordinate on delayed telecom. "
        "Hospitals must stay safe through the storm."
    ),
}
```

Also add to `TASK_SEED_SPLITS`, `TASK_NAMES` in `inference.py`, and `MAX_STEPS`.

### 4.3 Structured CLI Output Mode (Easy to Demo to Judges)

Add `VERBOSE_CLI=1` environment variable. When set, `inference.py` calls `print_step_summary()` (from Gap 3.3) for every step. When not set, only emits `[START]`/`[STEP]`/`[END]` for clean machine-readable output.

### 4.4 Episode Replay Summary at End of Each Task

At the end of each `run_task()`, print a concise summary table:
```
━━━ EPISODE SUMMARY: task_easy (seed=42) ━━━
 Score: 0.89  |  Steps: 7  |  Budget Used: 4.0/5.0
 Failures Survived: POWER_DIST_1 (recovered step 3)
 Hospital Health: min=0.72 across 7 steps
 Actions: harden×1  recover×2  wait×4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.5 Leakage-Safe Diagnostic Verification

The `diagnostics` block in `_build_observation` uses `_build_dependency_feedback()` which correctly only uses OBSERVED data. But `recommended_recovery_order` internally calls `sum(1 for e in observed_edges if e.target_id == nid...)` using the observed edge list which may differ from truth for delayed nodes. Add an assertion in debug mode:
```python
# In _build_observation(), at the end:
if os.getenv("DEBUG_LEAKAGE", "0") == "1":
    _verify_no_ground_truth_leak(obs, self._node_states)
```

---

## 5. Detailed Code Changes (Ready to Implement)

### 5.1 `tasks.py` — Add `_sample_graph_variation()`

```python
def _sample_graph_variation(cfg: dict, rng: random.Random) -> None:
    """Add seed-dependent topology variation to prevent static graph overfitting."""
    task_id = cfg.get("task_id", "")
    
    if task_id == "task_medium" and rng.random() < 0.25:
        # Add a redundant water treatment node
        cfg["nodes"].append({"node_id": "WATER_TREAT_2", "sector": "water", "is_critical": False})
        cfg["edges"].append({"source_id": "POWER_DIST_2", "target_id": "WATER_TREAT_2"})
        cfg["edges"].append({"source_id": "WATER_TREAT_2", "target_id": "WATER_DIST_1"})
    
    if task_id in ("task_hard", "task_cyberattack") and rng.random() < 0.30:
        # Add an extra telecom node (increases cascade depth complexity)
        cfg["nodes"].append({"node_id": "TELECOM_3", "sector": "telecom", "is_critical": False})
        existing_switch = next(
            (n["node_id"] for n in cfg["nodes"] if "SWITCH" in n["node_id"]), None
        )
        if existing_switch:
            cfg["edges"].append({"source_id": existing_switch, "target_id": "TELECOM_3"})
        # Maybe mark new node as partially observed
        if rng.random() < 0.5:
            cfg.setdefault("partial_obs_nodes", []).append("TELECOM_3")
```

### 5.2 `tasks.py` — Add `_sample_budget_variation()`

```python
def _sample_budget_variation(cfg: dict, rng: random.Random) -> None:
    """Apply seed-dependent budget jitter (±10%)."""
    base_budget = float(cfg.get("budget", 10.0))
    jitter = rng.uniform(0.90, 1.10)
    cfg["budget"] = round(base_budget * jitter, 1)
```

Call both at the end of `materialize_task_config()`:
```python
def materialize_task_config(task_id: str, seed: int, split: str = "train") -> dict:
    ...  # existing code
    cfg["stress_schedule"] = _jitter_schedule(cfg, rng)
    _sample_observability(cfg, rng)
    _sample_graph_variation(cfg, rng)   # ADD THIS
    _sample_budget_variation(cfg, rng)  # ADD THIS
    return cfg
```

### 5.3 `cascade_environment.py` — Add `_init_scada_event`

In `__init__()`, add:
```python
self._scada_event: Optional[dict] = None
```

In `reset()`, replace the existing scada_start detection block:
```python
self._scada_start = None
self._scada_event = None
for step_n, event in sorted(self._stress_schedule.items()):
    if event.get("type") == "scada_anomaly" and event.get("recurring"):
        self._scada_start = step_n
        self._scada_event = event
        break  # first recurring scada only
```

In `step()`, replace the entire SCADA recurring block with:
```python
if self._scada_start is not None and self._step >= self._scada_start and self._scada_event:
    t = self._scada_event.get("target")
    if t and t in self._node_states:
        ns = self._node_states[t]
        if ns.health > 0.0:
            ns.health = max(0.0, ns.health + self._scada_event["effect"])
```

### 5.4 New file: `adversarial_attacker.py`

```python
"""
adversarial_attacker.py — Rule-based attacker for Theme 1.1 (Attacker-Defender)
==================================================================================
The attacker observes the environment state and the defender's last action,
then picks the maximally damaging next attack event.

Attacker strategies:
  COUNTER_HARDEN  — defender hardened X, so attack a DIFFERENT high-value node
  CHAIN_DISRUPT   — defender is recovering X, attack X's downstream to waste the recovery
  OPPORTUNISTIC   — defender waited, attack the most critical unhardened node
  ESCALATE        — storm escalation when defender's budget is low

Usage:
    attacker = AdversarialAttacker()
    env.inject_attack(attacker.pick_attack(env, last_defender_action))
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import random


class AdversarialAttacker:
    FAULT_DAMAGE = -0.40   # single attack health hit

    def __init__(self, aggression: float = 0.6, seed: int = 0) -> None:
        self.aggression = aggression
        self._rng = random.Random(seed)
        self._attacks_fired: int = 0

    def pick_attack(self, env, last_defender_action: str) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Returns an attack_event dict or None.
        
        Args:
            env: CascadeEnvironment instance (reads _node_states, _budget, etc.)
            last_defender_action: action_type string from the defender's last step
        """
        if self._rng.random() > self.aggression:
            return None   # attacker rests this step

        node_states = env._node_states
        budget = env._budget

        # Sort candidate targets by criticality × vulnerability
        candidates = [
            ns for ns in node_states.values()
            if ns.is_operational and not ns.is_hardened
        ]
        if not candidates:
            return None

        if last_defender_action == "harden":
            # Just hardened something — attack something else
            candidates = [ns for ns in candidates]  # all unhardened
        elif last_defender_action == "recover":
            # In recovery — hit a node that would cascade to the one being recovered
            # (upstream disruption)
            recovering = list(env._pending_recoveries.keys())
            for nid in recovering:
                upstreams = [
                    ns for ns in node_states.values()
                    if ns.is_operational and not ns.is_hardened
                    and any(key.startswith(f"{ns.node_id}->") and key.endswith(f"->{nid}")
                            for key in env._edge_states)
                ]
                if upstreams:
                    candidates = upstreams
                    break
        elif last_defender_action == "wait":
            # Punish inaction — hit the most critical node
            critical = [ns for ns in candidates if ns.is_critical or ns.sector == "hospital"]
            if critical:
                candidates = critical

        # Pick highest-value target (lowest health = closest to failing)
        candidates.sort(key=lambda ns: (ns.is_hardened, ns.health))
        target = candidates[0]

        self._attacks_fired += 1
        return {
            "type": "equipment_fault",
            "target": target.node_id,
            "effect": self.FAULT_DAMAGE,
        }

    def escalate_weather(self, env) -> Optional[Dict[str, Any]]:
        """Trigger a storm escalation if current weather is clear."""
        if env._weather_forecast == "clear" and self._rng.random() < 0.4:
            return {"type": "weather", "target": None, "effect": "storm_warning"}
        return None

    @property
    def attacks_fired(self) -> int:
        return self._attacks_fired
```

---

## 6. How to Run (Complete Commands)

### 6.1 Prerequisites

```bash
# Install in development mode
pip install -e .

# Or minimal server-only install
pip install -r server/requirements.txt
```

Required environment variables:
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"   # or any OpenAI-compat endpoint
export MODEL_NAME="llama-3.3-70b-versatile"             # Groq model, or any available model
export API_KEY="your_groq_or_hf_api_key"
export ENV_BASE_URL="https://your-hf-space-url.hf.space"
```

### 6.2 Start the Environment Server

```bash
# Primary
uv run server

# Alternative
python -m cascade_guard.server.app

# Health check
curl http://localhost:8000/health
```

### 6.3 Run Baseline Inference (Single-Seed, All 5 Tasks)

```bash
python inference.py
```

Expected stdout (each line is machine-readable):
```
[START] task=task_easy env=cascade_guard model=llama-3.3-70b-versatile
[STEP] step=1 action=harden reward=0.52 done=false error=null
[STEP] step=2 action=harden reward=0.54 done=false error=null
[STEP] step=3 action=recover reward=0.71 done=false error=null
[STEP] step=4 action=wait reward=0.50 done=false error=null
[END] success=true steps=7 score=0.89 rewards=0.52,0.54,0.71,...
```

### 6.4 Run With Verbose CLI (Human-Readable, for Demos)

```bash
VERBOSE_CLI=1 python inference.py
```

Output (after implementing Gap 3.3):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1/10 | task_easy | Budget: 5.0 | Weather: clear
  NETWORK:  power=1.00
  FAILURES: none
  ACTION:   harden(POWER_GEN_1)  [reward: +0.517]
  AT RISK:  POWER_DIST_1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.5 Run With Chain-of-Thought Mode (Full Reasoning Output)

```bash
COT_MODE=1 python inference.py
```

The LLM will now output `<think>...</think><action>...</action>` blocks. Enable `DEBUG_LOGS=1` to see the full reasoning:
```bash
COT_MODE=1 DEBUG_LOGS=1 python inference.py 2>&1 | grep "LLM raw"
```

### 6.6 Multi-Seed Holdout Evaluation (Reproducibility Report)

```bash
EVAL_MODE=multiseed EVAL_SPLIT=holdout python inference.py
```

Writes `baseline_results.json` with per-task mean/std/min/max scores.

### 6.7 Run GRPO Training

```bash
# Install training dependencies first
pip install unsloth trl transformers accelerate peft bitsandbytes

# Train on easy task for 200 steps
python grpo_train.py --task task_easy --steps 200

# Full curriculum (auto-advances)
python grpo_train.py --curriculum --steps 1000

# Colab mode (smaller batch, less VRAM)
python grpo_train.py --task task_easy --steps 100 --colab
```

Training writes `training/reward_log.csv` per step.

### 6.8 Generate Training Curves

```bash
python plot_curves.py --log training/reward_log.csv --out figs/
```

Produces `figs/reward_curve.png` and `figs/per_task.png`.

### 6.9 Run Trajectory Logger Comparison

```python
from cascade_guard.training.trajectory_logger import TrajectoryLogger
from cascade_guard.server.cascade_environment import CascadeEnvironment

logger = TrajectoryLogger("logs/trajectories")
env = CascadeEnvironment()
obs = env.reset(task_id="task_easy", seed=42)

ep_id = logger.start_episode("task_easy", "baseline_agent", seed=42)
while not obs.done:
    # ... take action
    logger.record_step(obs, "recover", "POWER_DIST_1", obs.reward, obs.metadata or {})
    obs = env.step(action)
logger.end_episode(env.state.score)

logger.write_comparison_report("logs/comparison.md")
```

### 6.10 Run Attacker-Defender (After implementing adversarial_attacker.py)

```python
from cascade_guard.server.cascade_environment import CascadeEnvironment
from cascade_guard.adversarial_attacker import AdversarialAttacker
from cascade_guard.models import CascadeAction

env = CascadeEnvironment()
attacker = AdversarialAttacker(aggression=0.6)

obs = env.reset(task_id="task_cyberattack", seed=314)
last_action = "wait"
while not obs.done:
    # Defender picks action (replace with your LLM/heuristic)
    action = CascadeAction(action_type="wait", target_node_id=None)
    
    # Attacker fires BEFORE env.step
    attack = attacker.pick_attack(env, last_action)
    env.inject_attack(attack)
    
    obs = env.step(action)
    last_action = action.action_type
    
print(f"Defender score: {env.state.score:.3f}")
print(f"Attacker fired: {attacker.attacks_fired} attacks")
```

### 6.11 Docker

```bash
docker build -t cascade-guard:latest .
docker run -p 8000:8000 cascade-guard:latest
curl http://localhost:8000/health
```

### 6.12 Pre-Submit Validation

```bash
./validate-submission.sh https://your-hf-space-url.hf.space
```

---

## 7. Judging Criteria Mapping (Meta OpenEnv Hackathon 2026)

| Judging Dimension | How CascadeGuard Addresses It | Where in Code |
|---|---|---|
| **Real-world utility** | Models actual power-water-hospital-telecom coupling; dependency-order recovery is how real grid operators work | `tasks.py` topology, `README` problem statement |
| **OpenEnv compliance** | `reset()`/`step()`/`state()` API, typed `Action`/`Observation`/`State` models, `(0,1)` reward contract | `cascade_environment.py`, `models.py` |
| **Task quality + difficulty progression** | 5 tasks (+ 1 new) ranging from 1-sector to multi-sector adversarial; clear grader criteria per task | `tasks.py`, `graders.py` |
| **Reward quality (dense, not terminal-only)** | Per-step shaping (hospital health, cascade, overload, dependency order) + terminal sector bonus | `_compute_reward()` in `cascade_environment.py` |
| **Dynamic environment (not static)** | Seed-dependent schedule jitter, target swap, topology variation, budget jitter | `materialize_task_config()`, `_sample_graph_variation()` |
| **Theme 1.1 Attacker-Defender** | `inject_attack()` hook + `AdversarialAttacker` rule-based agent; `grade_adversarial()` multi-agent scorer | `adversarial_attacker.py`, `graders.py` |
| **Theme 2.3.1 Infrastructure Resilience** | Real sectors, real dependency order, real failure modes (SCADA, cyber-physical, weather, equipment) | Entire environment |
| **Baseline reproducibility** | `baseline_results.json` artifact with timestamp, model config, per-run records | `_write_reproducibility_report()` in `inference.py` |
| **Agent response quality (CoT)** | `<think>…</think><action>…</action>` format, dependency reasoning, priority hierarchy | `cot_prompt.py`, `COT_MODE` flag |
| **GRPO training** | Full training loop with curriculum scheduler, reward logging, Unsloth 4-bit LoRA support | `grpo_train.py`, `curriculum_scheduler.py` |

---

## 8. Reference Baseline Scores (Holdout Split)

| task_id | seed 1 | seed 2 | mean |
|---|---|---|---|
| task_easy | 0.8900 | 0.8900 | 0.8900 |
| task_medium | 0.3418 | 0.3359 | 0.3389 |
| task_hard | 0.3718 | 0.3648 | 0.3683 |
| task_gen_blackout | 0.7563 | 0.7263 | 0.7413 |
| task_cyberattack | 0.2480 | 0.2797 | 0.2639 |

After applying all fixes in this document, expected improvement targets:
- task_medium: 0.34 → 0.48 (CoT + correct delayed obs)
- task_hard: 0.37 → 0.50 (centrality protection fix + graph variation)
- task_cyberattack: 0.26 → 0.42 (attacker-defender training signal + SCADA fix)

---

## 9. Priority Implementation Order (For Remaining Hackathon Time)

**Do these first (highest judge impact):**

1. **Gap 3.1** — Add `_sample_graph_variation()` and `_sample_budget_variation()` to `tasks.py` (~30 min). This makes the env provably non-static.
2. **Gap 3.3** — Add `print_step_summary()` and `VERBOSE_CLI` to `inference.py` (~20 min). Makes your demo look incredible.
3. **Gap 3.2** — Create `adversarial_attacker.py` with `AdversarialAttacker` (~45 min). This is Theme 1.1 made explicit.
4. **Gap 3.7** — Fix SCADA recurring bug in `cascade_environment.py` (~10 min). Prevents subtle scoring errors.
5. **Gap 3.8** — Fix delayed observation buffer order in `_build_observation()` (~10 min). Correctness fix.

**Do these second (training quality):**

6. **Gap 3.5** — Complete `collect_rollouts()` in `grpo_train.py` (~20 min). Training will crash without this.
7. **Gap 3.4** — Wire `COT_MODE` in `inference.py` to actually use `cot_prompt.py` (~25 min). Makes your agent reasoning visible.
8. **Feature 4.2** — Add `task_surge_demand` to `tasks.py` (~30 min). 6th task shows thoroughness.

**Do last (polish):**

9. **Feature 4.1** — `grade_adversarial()` in `graders.py` (~15 min). Multi-agent scoring.
10. **Gap 3.6** — Fix `_centrality_protection_score()` to use `action_history` (~10 min). Scoring accuracy.

---

## 10. Key Environment Variables Reference

| Variable | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | None (required) | LLM API endpoint (Groq, HF, etc.) |
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B-Instruct` | Model to use |
| `API_KEY` | None (required) | API key |
| `ENV_BASE_URL` | HF Space URL | Where to find the env server |
| `EVAL_MODE` | `baseline` | `baseline` or `multiseed` |
| `EVAL_SPLIT` | `holdout` | `train`, `validation`, or `holdout` |
| `BASELINE_RESULTS_PATH` | `baseline_results.json` | Output path for reproducibility artifact |
| `DEBUG_LOGS` | `0` | Set to `1` for verbose stderr logs |
| `COT_MODE` | `0` | Set to `1` to use `<think>` CoT prompting |
| `VERBOSE_CLI` | `0` | Set to `1` for human-readable step output |
| `DEBUG_LEAKAGE` | `0` | Set to `1` to verify observation is ground-truth-free |
| `LOCAL_IMAGE_NAME` | `cascade-guard:latest` | Docker image name for local fallback |

---

## 11. Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────┐
│                    CascadeGuard System                       │
│                                                             │
│  ┌──────────┐    step/reset/state    ┌────────────────────┐ │
│  │ Defender │ ──────────────────────▶│  CascadeEnvironment│ │
│  │  Agent   │ ◀──── observation ─────│  (physics engine)  │ │
│  │ (LLM +   │                        │  - cascade prop.   │ │
│  │ heuristic│                        │  - weather/SCADA   │ │
│  │ + planner│                        │  - load dynamics   │ │
│  └──────────┘                        │  - delayed obs     │ │
│        ▲                             └────────────────────┘ │
│        │ RL signal                            ▲             │
│        │                                      │ inject_attack│
│  ┌──────────┐                        ┌────────────────────┐ │
│  │  GRPO    │                        │ AdversarialAttacker│ │
│  │ Trainer  │                        │ (Theme 1.1)        │ │
│  │ (Unsloth │                        └────────────────────┘ │
│  │  + TRL)  │                                              │
│  └──────────┘                                              │
│        │                             ┌────────────────────┐ │
│        └──── grader score ──────────▶│    Graders         │ │
│                                      │ (per-task scoring) │ │
│                                      └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Notes for the AI Receiving This File

If you are an AI reading this to continue development:

1. All file paths are relative to the `cascade_guard/` package root.
2. The environment inherits from `openenv.core.env_server.Environment`. Do not change the method signatures of `reset()`, `step()`, or `state`.
3. `grade()` in `graders.py` is the single entry point for scoring. Never call individual graders directly from outside the graders module.
4. All scores must satisfy `0.0 < score < 1.0` strictly. Use `_safe()` before returning any score.
5. `CascadeAction.action_type` is validated by Pydantic — only `{"harden", "shed_load", "coordinate", "recover", "wait"}` are allowed.
6. `EdgeStatus.active` is a plain bool (NOT `is_active`). `NodeStatus.observation_delayed` is a plain bool (NOT `is_delayed`).
7. `CascadeObservation.pending_recoveries` is `List[str]` (node IDs only), NOT objects with `steps_remaining`.
8. The `inject_attack()` method exists on `CascadeEnvironment` and is safe to call between `reset()` and `step()` in any loop. It is a no-op when passed `None`.
9. When running locally without a deployed HF Space, set `ENV_BASE_URL=http://localhost:8000` and ensure the server is running via `uv run server`.
10. The training script `grpo_train.py` uses `sys.path.insert` to handle running from the `training/` subdirectory — do not remove this.

---

*Generated: April 20, 2026 | CascadeGuard v3.0 | Meta OpenEnv Hackathon 2026*
