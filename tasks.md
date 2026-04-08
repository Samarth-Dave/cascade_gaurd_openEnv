# CascadeGuard Task Specification

This document explains the task suite in reviewer-friendly language, based on the current runtime definitions in [tasks.py](tasks.py) and grader logic in [server/graders.py](server/graders.py).

## Scope

The environment evaluates an agent on cross-sector infrastructure resilience under:

- finite action budget
- cascading dependency failures
- weather and cyber stress
- partial or delayed observability

The task set contains five scenarios with increasing coupling and pressure.

## Task Matrix

| Task ID | Steps | Budget | Nodes | Edges | Primary Stress Pattern |
|---|---:|---:|---:|---:|---|
| `task_easy` | 10 | 5.0 | 5 | 4 | Single equipment fault in power distribution |
| `task_medium` | 20 | 10.0 | 12 | 14 | Escalating storm with delayed observability |
| `task_hard` | 30 | 15.0 | 15 | 18 | Persistent SCADA degradation + storm |
| `task_gen_blackout` | 15 | 8.0 | 8 | 7 | Root-generator blackout cascade |
| `task_cyberattack` | 25 | 13.0 | 11 | 11 | Persistent SCADA + faults + storm + late fault |

## Action Semantics (What the Agent Can Do)

- `harden(node)`: costs budget, improves survival against stress
- `recover(node)`: restores failed node, dependency constraints apply
- `shed_load(node)`: reduces load pressure, unsafe on hospital/critical nodes
- `coordinate(node|null)`: improves observability when telemetry is delayed
- `wait()`: legal no-op, but penalized under pressure

## Per-Task Breakdown

### 1) `task_easy`

**Objective**
- Stabilize a single power graph after one severe distribution fault.

**Topology**
- Power-only chain: `GEN -> TRANS -> DIST`
- No delayed telemetry, no partial observation.

**Base stress schedule**
- Step 2: `equipment_fault` on `POWER_DIST_1` with effect `-0.75`

**Expected good behavior**
- Early hardening of high-centrality upstream nodes
- Fast recovery ordering after the fault

### 2) `task_medium`

**Objective**
- Keep power, water, and hospitals healthy through a storm sequence.

**Topology**
- Coupled tri-sector graph: power supports water and hospital services.

**Base stress schedule**
- Step 4: `weather = storm_warning`
- Step 8: `weather = extreme_storm`
- Step 11: `weather = clear`

**Observability profile**
- Delayed sector includes water by default
- In sampled scenarios, hospital can also become delayed with low probability

**Expected good behavior**
- Pre-storm hardening of shared backbone nodes
- Coordination during delayed visibility windows
- Hospital continuity preservation

### 3) `task_hard`

**Objective**
- Manage four-sector dependencies with persistent cyber pressure and weather.

**Topology**
- Power + Water + Hospital + Telecom coupling

**Base stress schedule**
- Step 1 onward: recurring `scada_anomaly` on `TELECOM_SWITCH_1`, effect `-0.02`
- Step 5: `weather = extreme_storm`
- Step 10: `weather = clear`

**Observability profile**
- Deterministic partial-observation nodes:
  - `WATER_PUMP_1`, `WATER_PUMP_2`, `TELECOM_1`, `TELECOM_2`
- No random observability override for this task

**Expected good behavior**
- Budget triage across sectors (not all-in on one sector)
- Priority on high-leverage dependency nodes
- Timely recoveries in dependency-safe order

### 4) `task_gen_blackout`

**Objective**
- Prevent root fault from cascading through transmission/distribution into hospitals.

**Topology**
- Power backbone feeding direct hospital endpoints

**Base stress schedule**
- Step 3: `equipment_fault` on `POWER_GEN_1`, effect `-0.85`
- Step 8: `equipment_fault` on `POWER_DIST_2`, effect `-0.80`

**Expected good behavior**
- Centrality-aware hardening (root/backbone before leaves)
- Hospital-protective sequencing before second fault window

### 5) `task_cyberattack`

**Objective**
- Survive sustained cyber-physical pressure and late-stage compounded failures.

**Topology**
- Power-hospital-telecom coupling with delayed/partial observability

**Base stress schedule**
- Step 1 onward: recurring `scada_anomaly` on `TELECOM_SWITCH_1`, effect `-0.03`
- Step 4: `equipment_fault` on `POWER_DIST_1`, effect `-0.65`
- Step 10: `weather = storm_warning`
- Step 14: `weather = extreme_storm`
- Step 17: `weather = clear`
- Step 20: `equipment_fault` on `POWER_DIST_2`, effect `-0.60`

**Observability profile**
- Partial-observation nodes are sampled per scenario for this task
- Sampled size is about 20% of eligible non-critical nodes (min 2)

**Expected good behavior**
- Persistent threat management (do not exhaust budget early)
- Protect hospital-serving power paths
- Recover quickly after stacked stress windows

## Scenario Materialization Rules (Important for Review)

Tasks are deterministic per `(task_id, seed, split)` but not identical to base templates:

1. **Step jittering**
- Each scheduled event is jittered by `-1, 0, +1` step
- Jitter is bounded to valid timeline and collision-resolved

2. **Equipment target mutation**
- For `equipment_fault` events, target can switch to another same-sector node with probability `0.35`

3. **Split-specific randomness**
- RNG is offset by split (`train`, `validation`, `holdout`) for reproducible but distinct scenarios

## Seed Splits

Each task defines train/validation/holdout seed lists in [tasks.py](tasks.py).

- `task_easy`: train `[42, 1042, 2042, 3042, 4042]`, validation `[5042, 6042]`, holdout `[7042, 8042]`
- `task_medium`: train `[123, 1123, 2123, 3123, 4123]`, validation `[5123, 6123]`, holdout `[7123, 8123]`
- `task_hard`: train `[999, 1999, 2999, 3999, 4999]`, validation `[5999, 6999]`, holdout `[7999, 8999]`
- `task_gen_blackout`: train `[777, 1777, 2777, 3777, 4777]`, validation `[5777, 6777]`, holdout `[7777, 8777]`
- `task_cyberattack`: train `[314, 1314, 2314, 3314, 4314]`, validation `[5314, 6314]`, holdout `[7314, 8314]`

## Grading Signals by Task

Scores are deterministic weighted composites in [server/graders.py](server/graders.py), then clamped to `[0.01, 0.99]`.

- `task_easy`: avg health `0.35`, no blackout `0.35`, cascade containment `0.20`, cascade depth `0.10`
- `task_medium`: avg `0.25`, hospital maintenance `0.30`, no blackout `0.20`, cascade `0.10`, depth `0.10`, dependency order `0.05`
- `task_hard`: avg `0.20`, hospital `0.25`, no blackout `0.15`, cascade `0.10`, budget efficiency `0.10`, depth `0.10`, dependency order `0.05`, intervention efficiency `0.05`
- `task_gen_blackout`: avg `0.20`, hospital `0.35`, no blackout `0.20`, cascade `0.10`, depth `0.10`, dependency order `0.05`
- `task_cyberattack`: avg `0.18`, hospital `0.28`, no blackout `0.16`, cascade `0.10`, budget efficiency `0.10`, depth `0.08`, dependency order `0.05`, intervention efficiency `0.05`

## Reviewer Checklist

When reviewing logs or trajectories, verify that:

1. agent prioritizes dependency-central nodes before leaf-only nodes
2. hospital continuity is maintained under stress windows
3. recoveries follow upstream dependency constraints
4. budget is not exhausted too early on low-leverage actions
5. coordination is used when observability is degraded
6. final score remains inside `[0.01, 0.99]`

## Notes

- This document describes the **current** task definitions and materialization behavior.
- Runtime scenarios are seeded and reproducible, but event timing/targets may differ from base template due to deterministic jitter and mutation logic.