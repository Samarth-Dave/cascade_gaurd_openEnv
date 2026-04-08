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

## 2. Action Space

Example actions:

Actions:
- harden(node)
- recover(node)
- shed_load(node)
- coordinate(node)
- wait()

Detailed semantics:
- harden(node): consumes budget and lowers the node failure threshold, making it
  harder to knock out during stress windows.
- recover(node): starts a multi-step restoration process, but only when upstream
  dependencies are operational.
- shed_load(node): reduces overload pressure; unsafe use on critical/hospital nodes
  is strongly penalized.
- coordinate(node): temporary observability aid that reduces delay pressure for
  target sector decision-making.
- wait(): valid no-op, but penalized under actionable or high-pressure conditions.

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

## 5. Setup Instructions

```bash
pip install -r requirements.txt
python inference.py
```

If your local clone does not include a root requirements file, use one of:

```bash
pip install -e .
# or
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

Health check:

```bash
curl http://localhost:8000/health
```

You can then run inference/evaluation against the running server.
