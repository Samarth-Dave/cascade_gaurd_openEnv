# CascadeGuard Improvements Log

A versioned record of every improvement sprint.

---

## v2.3 — 2026-04-22 | Remaining Plan Execution (Phases 1–5)

### Scope Completed

This sprint completes the remaining checklist items from the master plan: training
diversity controls, runtime correctness fixes, new surge-demand task integration,
evaluation observability upgrades, and grader budget-efficiency wiring.

### 1) Runtime Correctness Fixes

| Fix | File | Outcome |
|---|---|---|
| Observation buffer fill order moved before node-list construction in `_build_observation()` | `server/cascade_environment.py` | Delayed observations now use correctly ordered episode-local history; removes stale first-observation behavior |
| SCADA event initialized and cached at reset (`_scada_event`) instead of schedule scans each step | `server/cascade_environment.py` | Recurring SCADA behavior is deterministic per episode and no longer lazily inferred during step execution |
| `_has_active_scada()` switched to cached event path | `server/cascade_environment.py` | `patch_scada` validity checks align with runtime SCADA application |

### 2) New Task: `task_surge_demand`

| Addition | File | Outcome |
|---|---|---|
| New task config with +20% load surges at steps 3, 8, 14 and no direct equipment faults | `tasks.py` | Adds isolated load-management benchmark to curriculum |
| Seed families for train/validation/holdout | `tasks.py` (`TASK_SEED_SPLITS`) | Reproducible multi-seed evaluation for surge task |
| `load_surge` stress-event handling | `server/cascade_environment.py` | Surge events now increase node loads during runtime |
| Surge grader `grade_surge_demand()` + dispatch wiring | `server/graders.py` | Deterministic scoring for the new task |
| Sanity verification task list updated | `verify_tasks.py` | Local structural checks now include surge task |

### 3) Training Diversity and Collapse-Prevention

| Addition | File | Outcome |
|---|---|---|
| Three teacher sub-policies: `proactive_hardener`, `load_manager`, `recovery_prioritizer` | `training/train_grpo.py` | State collection now covers distinct strategic modes |
| Epsilon exploration in `collect_training_states()` (`ε=0.30`, first 5 steps) | `training/train_grpo.py` | Forces non-wait coverage in early rollout states |
| Consecutive-wait cap (`MAX_CONSECUTIVE_WAITS=3`) | `training/train_grpo.py` | Prevents long passive segments in collected trajectories |
| Default training task list includes `task_surge_demand` | `training/train_grpo.py` | Curriculum includes load-only stress regime by default |

### 4) Evaluation & Observability Upgrades

| Metric/Output | File | Outcome |
|---|---|---|
| Per-episode action distribution (`action_distribution`) | `training/train_grpo.py` | Direct visibility into action usage patterns |
| Wait-rate + action-diversity | `training/train_grpo.py` | Better anti-collapse monitoring |
| Parse-failure count/rate | `training/train_grpo.py` | Tracks output-format robustness in policy evaluation |
| Per-task summary CSV (`cascadeguard_eval_task_summary.csv`) | `training/train_grpo.py` | Task-level comparison and drift analysis |

### 5) Grader Budget-Efficiency Component

| Addition | File | Outcome |
|---|---|---|
| `_budget_efficiency_score()` helper (0 spend → floor, 50% spend → full credit) | `server/graders.py` | Explicitly rewards useful budget utilization |
| Wired into `grade_easy`, `grade_medium`, `grade_gen_blackout` | `server/graders.py` | Removes residual zero-spend safe harbor in these task graders |

### 6) Inference Task Registry Update

| Update | File | Outcome |
|---|---|---|
| `TASK_NAMES`, `MAX_STEPS`, and budget map include `task_surge_demand` | `inference.py` | Baseline/eval sweeps can run on the new task without manual edits |

---

## v2.2 — 2026-04-21 | Action Space Expansion (Round 3)

### What Changed

This sprint expands the action space from 5 actions to 15, organized across four
difficulty tiers. The motivation: with only 5 actions the GRPO decision surface was too
narrow for the agent to learn meaningful strategy differentiation across tasks with
4 sectors, up to 18 nodes, and up to 35 steps per episode. With 15 actions the agent
has enough variety to develop real triage and sacrifice reasoning while staying
computationally tractable.

#### New Actions Added

| Action | Tier | Files Affected | What It Does |
|--------|------|----------------|--------------|
| `reroute(source, target)` | Intermediate | `models.py`, `cascade_environment.py` | Redirects flow from a healthy alternate node to a target that lost its upstream supply |
| `prioritize(node)` | Intermediate | `models.py`, `cascade_environment.py` | Grants +0.15 stability and stress immunity for 1 step; cooldown 3 steps |
| `deploy_repair_crew(node)` | Intermediate | `models.py`, `cascade_environment.py` | Halves repair time at 1.5× budget cost |
| `emergency_shutdown(node)` | Hard | `models.py`, `cascade_environment.py` | Controlled shutdown of a node below 30% health; prevents chaotic cascade trigger |
| `cross_sector_bridge(sector_a, sector_b)` | Hard | `models.py`, `cascade_environment.py` | 3-step cross-sector redundancy link between any two sectors |
| `patch_scada(node)` | Hard | `models.py`, `cascade_environment.py` | Stops active SCADA anomaly drain and restores observation fidelity on target node |
| `redistribute_load(node_a, node_b)` | Hard | `models.py`, `cascade_environment.py` | Moves excess load between two operational same-sector nodes for 2 steps |
| `request_mutual_aid(sector)` | Hard | `models.py`, `cascade_environment.py` | One-time +0.2 health boost to all nodes in a sector; once per episode |
| `controlled_cascade(node)` | Expert | `models.py`, `cascade_environment.py` | Deliberate -0.5 hit on a sacrifice node; neighboring nodes gain +0.15 stability for 2 steps |
| `multi_sector_lockdown()` | Expert | `models.py`, `cascade_environment.py` | 2-step system freeze — no cascade propagation, no recovery, then unfreezes |

#### Existing Actions Retained (clarified semantics)

| Action | Change |
|--------|--------|
| `wait()` | Unchanged. Penalized under actionable conditions. |
| `harden(node)` | Unchanged. Cost range documented as 0.8–1.2. |
| `recover(node)` | Unchanged. Cost range documented as 0.6–1.0. |
| `isolate(node)` | Unchanged from v2.1. Retained in Tier 2. |
| `shed_load(node)` | Unchanged. Explicitly moved to Tier 2 with documented cost 0.3. |
| `coordinate(node)` | Unchanged from v2.1. Retained in Tier 2. |

#### Environment Changes Required

| Change | File | Detail |
|--------|------|--------|
| Action enum / discriminated union | `models.py` | Add 10 new action types with typed parameters |
| `_apply_action()` dispatch | `cascade_environment.py` | Implement mechanics for each new action |
| `get_legal_actions()` updates | `cascade_environment.py` | Validity checks: `reroute` requires alternate path, `patch_scada` requires active anomaly, `redistribute_load` requires both nodes operational and same sector, `request_mutual_aid` once-per-episode flag, `emergency_shutdown` requires health < 0.30, `prioritize` requires cooldown check |
| Cooldown tracking | `cascade_environment.py` | Add `_prioritize_cooldown` counter (resets every 3 steps) |
| Once-per-episode flag | `cascade_environment.py` | Add `_mutual_aid_used: bool` state field |
| Reroute graph traversal | `cascade_environment.py` | BFS/DFS to find valid alternate supply path between source and target |
| Cross-sector bridge state | `cascade_environment.py` | Track active bridges with TTL counter (3 steps) |
| Lockdown state | `cascade_environment.py` | `_lockdown_remaining: int` field; skip cascade propagation and recovery ticks when > 0 |
| Repair crew fast-track | `cascade_environment.py` | `_fast_repair_nodes: set` checked in recovery tick to halve step count |
| SCADA patch flag | `cascade_environment.py` | Per-node `_scada_patched` set; suppress anomaly drain when flagged |
| Grader awareness | `graders.py` | `controlled_cascade` sacrifice penalized in health grader component; `multi_sector_lockdown` freeze window excluded from cascade containment window |
| Reward shaping | `cascade_environment.py` | Add small positive shaping for: correct `patch_scada` timing, valid `reroute` that keeps a hospital alive, `request_mutual_aid` that averts a full-sector blackout |
| Prompt parser | `cot_prompt.py` | Accept all 10 new action signatures in XML/action-tag completions |
| Legal action list in prompts | `cot_prompt.py` | New actions appear in per-step legal action list so agent knows they exist |

#### Why These Actions Improve Training

- **Richer gradient signal.** 15 actions × 4 sectors × 18 nodes = a much larger
  combinatorial space, giving GRPO more paths to differentiate good from bad trajectories.
- **Curriculum-compatible tiers.** Tier 1 actions dominate task_easy, Tier 2 dominate
  task_medium, Tiers 3–4 are required to score well on task_hard and task_cyberattack.
  This naturally maps onto the existing curriculum learning setup.
- **Sacrifice reasoning.** `controlled_cascade` and `emergency_shutdown` force the agent
  to learn graph centrality — the hardest reasoning skill in the environment — through
  direct reward consequence rather than indirect penalty.
- **Budget pressure scaling.** `request_mutual_aid` (2.5) and `multi_sector_lockdown`
  (2.0) are expensive enough that the agent must develop episode-level budget planning,
  not just greedy per-step optimization.

---

## v2.1 — 2026-04-21 | GRPO Training Alignment + Local Notebook

### What Changed

#### Environment
| Feature | File | Impact |
|---|---|---|
| Added `reward_mode="grpo"` | `cascade_environment.py` | Exposes signed, scaled rewards instead of clamped `(0, 1)` rewards for GRPO |
| Added `training_mode=True` | `cascade_environment.py` | Keeps training episodes consistent length by disabling early success/failure exits |
| Added `get_legal_actions()` | `cascade_environment.py` | Training rewards and prompts now use the environment's true validity rules |
| Added `get_state_features()` | `cascade_environment.py` | Flat state vector for reward shaping, debugging, and lightweight baselines |
| Added `isolate(node)` action | `models.py`, `cascade_environment.py` | Failed nodes can be temporarily isolated for 3 steps to stop cascade spread |
| Isolation-aware cascade propagation | `cascade_environment.py` | Isolated failed upstream nodes no longer damage downstream dependents |
| Fixed root-node dependency recovery score | `cascade_environment.py` | Root recoveries now receive meaningful dependency-order credit |
| Exposed `upcoming_stress_level` | `models.py`, `cascade_environment.py` | Prompts can time hardening before imminent stress events |
| GRPO reward noise and grader-shaped bonus | `cascade_environment.py` | Adds reward variance and aligns step reward with final grader components |
| Step reward component metadata | `cascade_environment.py` | Metadata now includes reward breakdown, legal action count, and state features |
| Dynamic load mean reversion target | `cascade_environment.py` | `shed_load` becomes useful under storm/failure pressure |

#### Graders
| Fix | File | Impact |
|---|---|---|
| Cascade containment blends peak and average failure exposure | `graders.py` | Agents get credit for recovering after a bad cascade step |
| Split proactive action score from budget conservation | `graders.py` | Removes contradictory "act more but spend less" target |
| Easy task includes hospital maintenance | `graders.py` | Aligns easy-task final scoring with hospital-heavy step penalties |

#### Prompting and Training
| Change | File | Impact |
|---|---|---|
| Legal actions come from `env.get_legal_actions()` | `cot_prompt.py`, notebook | Prevents notebook prompt/reward validity drift |
| GRPO prompt includes stress-imminence guidance | `cot_prompt.py` | Makes hardening timing learnable |
| Parser accepts `isolate(...)` | `models.py`, `cot_prompt.py` | New action works in XML/action-tag completions |
| GRPO script defaults to local `reward_mode="grpo"` | `training/grpo_train.py` | Training uses signed reward signal by default |
| Colab notebook is local-first | `CascadeGuard_GRPO_Colab.ipynb` | Trains against this checkout before Hugging Face deployment |
| Removed hardcoded token cell | `CascadeGuard_GRPO_Colab.ipynb` | Safer local/Colab runs |
| Added optional local server launcher | `CascadeGuard_GRPO_Colab.ipynb` | API smoke testing without needing deployed HF Space |

### Recommended Local Run

```bash
cd "c:\Users\Samarth Dave\Desktop\metaV1"
python -m uvicorn cascade_guard.server.app:app --host 127.0.0.1 --port 8000
```

The GRPO notebook does not require that server; it imports `CascadeEnvironment`
directly from the local checkout and resets with:

```python
env.reset(task_id=task_id, seed=seed, reward_mode="grpo", training_mode=True)
```

### Verification

| Check | Result |
|---|---|
| `python -m compileall ...` | Passed |
| `python cascade_guard\verify_tasks.py` | Passed |
| `python verify_v2.py` | Passed |
| `python test_env_v2.py` | Passed |
| Notebook JSON/code-cell parse check | Passed |
| `isolate(...)` injected-failure smoke test | Passed |

---

## v2.0 — 2026-04-21 | Major Overhaul

### What Changed

#### Environment
| Feature | File | Impact |
|---|---|---|
| Geo-aware radius coverage model | `geo_utils.py` (NEW) | Cascade damage reduced by nearby backup nodes |
| Real Mumbai-inspired coordinates | `tasks.py`, `models.py` | Realistic city-scale geography |
| New `task_real_city` (18 nodes, 35 steps) | `tasks.py` | Full Metro City monsoon scenario |
| Dynamic partial observability | `cascade_environment.py` | Telecom coverage → observation confidence |
| Narrative phase tracker | `cascade_environment.py` | PRE_STORM→CRISIS_PEAK→RECOVERY phases |
| Structured invalid action reasons | `cascade_environment.py` | `unknown_target`, `already_hardened`, etc. |
| Progressive wait penalty | `cascade_environment.py` | Anti-wait pressure scales with consecutive waits |
| Cascade debt pressure | `cascade_environment.py` | +0.04/step penalty per unresolved failure after 3 steps |

#### GRPO Training Fixes
| Fix | Expected Impact |
|---|---|
| Default model upgraded: 0.5B → 1.5B | Better JSON/instruction following |
| Training steps: 10 → 200 (min viable) | Eliminates zero-loss collapse |
| Per-batch reward normalization | Guarantees reward variance (no zero gradient) |
| Wait penalty: -0.15 → -0.50 (progressive) | Breaks "always wait" mode collapse |
| Added legal action list to every prompt | Reduces invalid action rate (was 71%) |
| Multi-objective reward function | More informative training signal |
| Curriculum learning (easy→hard) | Faster convergence |
| `num_generations`: 4 → 8 | More reward variance per batch |
| LoRA rank: 16 → 32, alpha: 16 → 64 | More expressive adapter |

#### Agent Prompting
| Change | File |
|---|---|
| Legal action list appended to prompts | `cot_prompt.py` |
| Narrative phase in prompt header | `cot_prompt.py` |
| Observation confidence column in node table | `cot_prompt.py` |
| Radius coverage events in prompt | `cot_prompt.py` |
| GRPO-mode compact prompt format | `cot_prompt.py` |
| JSON fallback parser | `cot_prompt.py` |

#### CLI & Display
| Feature | File |
|---|---|
| New `cli_display.py` module | `cli_display.py` (NEW) |
| ANSI color health bars | `cli_display.py` |
| Narrative phase banners with emojis | `cli_display.py` |
| Budget gauge | `cli_display.py` |
| Invalid reason display | `cli_display.py` |
| Consecutive wait warning | `cli_display.py` |

#### Documentation
| File | What |
|---|---|
| `HOW_TO_RUN.md` | Complete copy-paste execution guide |
| `IMPROVEMENTS_LOG.md` | This file |
| `FURTHER_IMPROVEMENTS.md` | Master roadmap |

### Baseline vs Target

| Metric | Before (v1) | Target (v2) |
|---|---|---|
| GRPO training loss | 0.000 (collapsed) | > 0.01 (learning) |
| `grpo` score on task_easy | 0.50 (all wait) | ≥ 0.60 |
| Invalid action rate (groq) | 71% | < 25% |
| `wait` rate (grpo_smoke) | 100% | < 40% |
| Score differentiation | None | ≥ 0.15 spread |

---

## v1.x — 2026-04-05 to 2026-04-20 | Foundation

- OpenEnv environment scaffold
- 5 tasks (easy → cyberattack)
- Adversarial attacker module
- GRPO training pipeline
- Trajectory logger
- Multi-seed evaluator
- Dependency-aware diagnostics
- Hospital penalty unification fix
- Catastrophic termination stabilization
