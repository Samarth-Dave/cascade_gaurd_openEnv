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

Runtime reward contract:
- Exposed step reward is normalized and clamped strictly to (0.0, 1.0) for evaluator compatibility.
- Raw shaped reward is still available in observation metadata as `raw_reward`.

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

Health check:

```bash
curl http://localhost:8000/health
```

Run baseline inference:

```bash
python inference.py
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
