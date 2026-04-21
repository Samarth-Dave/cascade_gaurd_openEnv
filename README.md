---
title: CascadeGuard Environment
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - infrastructure
---

# CascadeGuard Package

CascadeGuard is an OpenEnv environment for cross-sector infrastructure resilience.
It simulates cascading failures across power, water, hospital, and telecom systems,
where local failures can quickly become system-wide outages if the agent responds late
or in the wrong dependency order.

## 1. Problem Description

This environment simulates cascading failures in interdependent infrastructure systems.
The core idea is to train and evaluate an AI coordinator that must keep critical
services alive during multi-step crises (equipment faults, weather escalation, and
cyber-physical degradation) under finite budget and imperfect observability.

Why this problem matters:
- Real infrastructure failures are coupled, not isolated. A power disruption can
  disable water treatment, which then degrades hospitals and emergency response.
- Recovery is constrained by dependency order. Recovering a downstream node before
  its upstream suppliers are operational is often invalid or ineffective.
- Emergency operations are budget-constrained. Hardening and coordination improve
  resilience but consume limited resources.

How this environment maps to the PS (problem statement) expectations:
- Real-world utility: models a real infrastructure resilience task, not a toy game.
- OpenEnv compliance: typed action/observation/state models and reset/step/state APIs.
- Task quality: multiple tasks with clear difficulty progression and grader-based scoring.
- Reward quality: dense trajectory-level shaping, not only terminal binary rewards.
- Baseline reproducibility: inference flow is supported and benchmark-friendly.

Current task set:
- task_easy: single-sector fault handling
- task_medium: tri-sector storm with delayed telemetry
- task_hard: four-sector crisis with partial observability and recurring anomaly
- task_gen_blackout: root generator blackout cascade
- task_cyberattack: cyber-physical compound attack

### Task Objectives and Grader Criteria

- task_easy (easy)
  - Objective: stabilize a power-only network after a single equipment fault.
  - Grader signals: average sector health, blackout avoidance, cascade containment, and low cascade depth.
  - Expected behavior: proactive hardening and fast recovery sequencing.

- task_medium (medium)
  - Objective: maintain power, water, and hospitals through storm escalation with delayed water observations.
  - Grader signals: average health, hospital maintenance, blackout avoidance, cascade containment, depth, and dependency-order correctness.
  - Expected behavior: coordination under delayed telemetry and dependency-safe recoveries.

- task_hard (hard)
  - Objective: manage four sectors with partial observability, recurring SCADA degradation, and weather stress.
  - Grader signals: medium metrics plus budget efficiency and intervention efficiency.
  - Expected behavior: budget triage, critical-node protection, and failure-chain-aware recovery.

- task_gen_blackout (hard+ specialized)
  - Objective: survive a root-generator failure chain and a follow-up distribution fault.
  - Grader signals: high emphasis on hospital continuity plus blackout/cascade control.
  - Expected behavior: centrality-aware hardening (root and backbone before leaves).

- task_cyberattack (hardest)
  - Objective: withstand persistent cyber-physical degradation plus multiple faults and storm pressure.
  - Grader signals: hospital continuity, blackout prevention, cascade control, budget/intervention efficiency.
  - Expected behavior: sustained threat management over long horizons.

All grader outputs are deterministic and bounded strictly inside (0, 1) for each task run.

## 2. Action Space

The agent has 15 actions organized across four difficulty tiers. This expanded action
space gives GRPO sufficient decision-surface to learn meaningful strategy differentiation
across tasks with 4 sectors, up to 18 nodes, and up to 35 steps per episode.

### Tier 1 — Basic

**wait()**
The no-op. Valid but penalized under actionable conditions (active failure, pressure
above threshold). Use only during genuine repair cooldowns when no other valid action
exists. Cost: 0 budget.

**harden(node)**
Spends budget to lower a node's failure threshold, making it harder to knock out during
stress windows. Effect persists for the episode. Use proactively when `upcoming_stress_level`
is `soon` or `imminent`. Cost: 0.8–1.2 budget units.

**recover(node)**
Starts a multi-step restoration on a failed node. Only valid when all upstream
dependencies are operational. Invalid uses are penalized. Takes 2–4 steps depending on
node type. Cost: 0.6–1.0 budget units.

### Tier 2 — Intermediate

**isolate(node)**
Severs all outgoing cascade edges from a node to prevent it from propagating its failure
state downstream. The node itself stays degraded, but the cascade stops at that boundary.
Effect lasts 3 steps. Use when a failed node's downstream neighbors are healthy and
worth protecting. Cost: 0.3 budget.

**reroute(source, target)**
Redirects power or water flow from a healthy alternative node to a target that has lost
its upstream supply. Only valid when a viable alternate supply path exists in the graph.
Allows hospitals or water treatment to remain operational while a generator is being
repaired. Cost: 0.6 budget.

**prioritize(node)**
Marks a node as highest-priority for the current step, granting it a +0.15 stability
bonus and making it immune to stress escalation for 1 step. No direct budget cost, but
limited to once per 3 steps (cooldown enforced). Use to protect critical nodes during
peak stress windows. Cost: 0 (cooldown gated).

**deploy_repair_crew(node)**
Accelerated recovery — reduces repair time to 1–2 steps at 1.5× the standard recover
budget cost. Use when a hospital or critical node is at critically low health and cannot
survive the normal repair window. Cost: 1.5× recover cost.

**coordinate(node)**
Reduces observation delay on delayed-sector nodes for 2 steps, improving the agent's
observability confidence score for that sector. Especially important in task_medium and
task_cyberattack where delayed sectors obscure real node state. Cost: 0.4 budget.

**shed_load(node)**
Reduces overload pressure on a node. Unsafe use on critical or hospital nodes is
strongly penalized. Use only on non-critical nodes that are at risk of failure due to
sustained high load rather than external stress events. Cost: 0.3 budget.

### Tier 3 — Hard

**emergency_shutdown(node)**
Performs a controlled shutdown of a node whose health is below 30%, preventing it from
triggering a chaotic cascade. Unlike isolate, this also halts load on the node, which
can relieve pressure on its upstream suppliers. The agent controls the timing rather
than letting the environment trigger an uncontrolled failure. Cost: 0.2 budget.

**cross_sector_bridge(sector_a, sector_b)**
Establishes a temporary cross-sector redundancy link between two sectors for 3 steps,
allowing one sector's operational nodes to partially compensate for a gap in the other.
For example, bridging telecom to hospital can restore partial observability during a
hospital-sector crisis. Cost: 1.5 budget.

**patch_scada(node)**
Specifically counters an active `scada_anomaly` stress event. Stops the recurring health
drain on the targeted node and restores observation fidelity. Only valid on nodes
currently under SCADA anomaly. Critical in task_cyberattack where the drain compounds
every step from the start. Cost: 0.8 budget.

**redistribute_load(node_a, node_b)**
Moves excess load from an overloaded node (node_a) to an underloaded node (node_b)
within the same sector. Both nodes must be operational. Lowers node_a's failure
probability for 2 steps. Requires the agent to reason about relative load levels, not
just binary healthy/failed states. Cost: 0.5 budget.

**request_mutual_aid(sector)**
Invokes an external resource injection, adding a one-time +0.2 health boost to all
nodes in the target sector. Represents calling in outside mutual aid. Can only be used
once per episode — this is the emergency reserve. Use only when a sector is in
unrecoverable freefall and individual node actions cannot stop it. Cost: 2.5 budget.

### Tier 4 — Expert

**controlled_cascade(node)**
Deliberately triggers a controlled partial failure on a low-importance node to release
pressure across the graph. The sacrifice node takes an immediate -0.5 health hit, while
each neighboring node gains +0.15 stability for 2 steps. No direct budget cost, but the
grader penalizes the sacrificed node's health loss. Requires full graph centrality
awareness to identify which node is genuinely least critical. Cost: 0 budget (grader
penalty applies).

**multi_sector_lockdown()**
Freezes the entire system in a defensive state for 2 steps: no cascade can propagate,
all health values stabilize, but no recovery is possible during the lockdown window.
After 2 steps the system unfreezes and normal dynamics resume. Use only when
simultaneous multi-sector failure is spiraling faster than individual actions can
address. Cost: 2.0 budget.

### Action Summary Table

| # | Action | Tier | Budget Cost | Primary Use |
|---|--------|------|-------------|-------------|
| 1 | wait() | Basic | 0 | No-op during cooldowns |
| 2 | harden(node) | Basic | 0.8–1.2 | Proactive resilience before stress |
| 3 | recover(node) | Basic | 0.6–1.0 | Restore a failed node |
| 4 | isolate(node) | Intermediate | 0.3 | Stop cascade propagation |
| 5 | reroute(src, tgt) | Intermediate | 0.6 | Alternate supply path |
| 6 | prioritize(node) | Intermediate | 0 (cooldown) | Protect critical node this step |
| 7 | deploy_repair_crew(node) | Intermediate | 1.5× recover | Fast repair when time-critical |
| 8 | coordinate(node) | Intermediate | 0.4 | Fix observation delay |
| 9 | shed_load(node) | Intermediate | 0.3 | Relieve overload pressure |
| 10 | emergency_shutdown(node) | Hard | 0.2 | Controlled pre-failure shutdown |
| 11 | cross_sector_bridge(s_a, s_b) | Hard | 1.5 | Cross-sector redundancy link |
| 12 | patch_scada(node) | Hard | 0.8 | Counter active SCADA drain |
| 13 | redistribute_load(n_a, n_b) | Hard | 0.5 | Balance overloaded nodes |
| 14 | request_mutual_aid(sector) | Hard | 2.5 (once/ep) | Emergency sector-wide boost |
| 15 | controlled_cascade(node) | Expert | 0 + grader penalty | Sacrifice one node to save many |
| 16 | multi_sector_lockdown() | Expert | 2.0 | Freeze system during multi-sector spiral |

## 3. Observation Space

What the agent sees includes:
- nodes
- health
- load
- failures
- dependencies

In practice, each observation also includes important context for planning:
- step index and max steps
- remaining budget
- weather forecast
- active failures and pending recoveries
- sector-level health summary
- diagnostics block (critical failures, at-risk nodes, dependency alerts,
  recommended recovery order, and system pressure)
- upcoming stress level (`none`, `soon`, or `imminent`) so agents can time
  proactive hardening before scheduled stress events

This makes the environment useful for both raw policy learning and
analysis-friendly decision support experiments.

## 4. Reward Function (IMPORTANT)

Reward is based on:
- maintaining system stability
- minimizing cascading failures
- protecting critical infrastructure
- efficient resource usage

IMPORTANT: this is where judges look closely.

Reward design intent:
- Encourages proactive resilience (timely hardening and safe coordination).
- Rewards successful recoveries, especially in correct dependency order.
- Penalizes newly failed nodes, cascade propagation, invalid actions, and
  inaction under pressure.
- Applies strong safety penalties for harmful critical actions (for example,
  unsafe load shedding on hospital/critical nodes).
- Uses terminal shaping for overall sector outcomes, while preserving dense
  step-level signals for learning stability.

Runtime reward contract:
- Exposed step reward is normalized and clamped strictly to (0.0, 1.0) for evaluator compatibility.
- Raw shaped reward is still available in observation metadata as `raw_reward`.
- `reward_mode="uncapped"` exposes raw shaped reward directly.
- `reward_mode="grpo"` exposes signed, scaled reward for GRPO and enables small
  reward noise plus grader-aligned shaping.
- `training_mode=True` disables early success/failure exits so training batches
  compare fixed-length episodes.

## 5. Setup Instructions

```bash
pip install -e .
```

Alternative server-only install:

```bash
pip install -r server/requirements.txt
```

Required environment variables for inference:
- API_BASE_URL
- MODEL_NAME
- GROQ_API_KEY (preferred)

Optional fallback variables:
- HF_TOKEN (legacy fallback if GROQ_API_KEY is not set)
- ENV_BASE_URL (defaults to your HF Space URL and is used before Docker)

## 6. How to Run

Run the environment server (uv script entrypoint):

```bash
uv run server
```

Alternative launch command:

```bash
python -m cascade_guard.server.app
```

Explicit uvicorn command:

```bash
python -m uvicorn cascade_guard.server.app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Run baseline inference:

```bash
python inference.py
```

Run local GRPO training before pushing the environment:

```bash
# From the repo root, open and run:
jupyter notebook cascade_guard/training/CascadeGuard_GRPO_Colab.ipynb
```

The notebook trains against the local checkout directly. It does not require the
server or the deployed Hugging Face Space. The training reset path is:

```python
env.reset(task_id=task_id, seed=seed, reward_mode="grpo", training_mode=True)
```

## 7. Baseline Inference and Reproducibility

The baseline runner writes a reproducibility artifact to `baseline_results.json`
by default (override with `BASELINE_RESULTS_PATH`).

The artifact includes:
- timestamp and model config
- per-run records (task_id, seed, split, scenario index, score)
- per-task summary (runs, mean/std/min/max)

Recommended reproducibility run commands:

```bash
# Single-run baseline (train split, deterministic seed schedule)
python inference.py

# Multi-seed evaluation (holdout split)
EVAL_MODE=multiseed EVAL_SPLIT=holdout python inference.py
```

Reference baseline snapshot (holdout, 2 seeds each):

| task_id | sample scores | mean |
|---|---|---:|
| task_easy | 0.8900, 0.8900 | 0.8900 |
| task_medium | 0.3418, 0.3359 | 0.3389 |
| task_hard | 0.3718, 0.3648 | 0.3683 |
| task_gen_blackout | 0.7563, 0.7263 | 0.7413 |
| task_cyberattack | 0.2480, 0.2797 | 0.2639 |

## 8. Docker and Submission Hardening

- Docker build context is constrained via `.dockerignore` to reduce noisy or
  oversized uploads.
- Run pre-submit checks before submission:

```bash
./validate-submission.sh <your_hf_space_url>
```
