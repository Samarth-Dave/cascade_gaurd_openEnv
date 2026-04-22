# CascadeGuard Improvements Log

A versioned record of every improvement sprint.

---

## v2.3 — 2026-04-22 | Real-World OSM Infrastructure Upgrade

### What Changed

This sprint finalizes the infrastructure upgrade to a real-world, OpenStreetMap-backed environment for GRPO RL training. The focus was on replacing the fictional topologies with real city-scale infrastructure maps and ensuring accurate propagation of real-world node names.

#### New Tasks Added (Phase 2 Curriculum)

| Task | Difficulty | Description |
|------|------------|-------------|
| `task_osm_london` | Hard | London: Manageable topology as the entry point to OSM cities. |
| `task_osm_bangalore`| Hard | Bangalore: Well-connected grid topology. |
| `task_osm_delhi` | Hard | Delhi: Dense urban graph. |
| `task_osm_mumbai` | Hard | Mumbai: Familiar sector names with real coordinates. |
| `task_osm_nyc` | Hard+ | NYC: Highly complex inter-borough topology. |
| `task_osm_tokyo` | Hardest | Tokyo: Dense metro, the most difficult real-world task. |

#### Major Improvements

| Component | File | Detail |
|-----------|------|--------|
| OSM Data Fetching | `scripts/fetch_real_nodes.py` | Query OpenStreetMap via Overpass API to extract real hospitals, power plants, and telecom towers. |
| Node Propagation | `models.py`, `tasks.py` | Nodes now support a `real_name` attribute (e.g., "KEM Hospital") passed throughout the environment. |
| Prompt Clarity | `training/cot_prompt.py` | Appended a `Real Name` column to the observation table so the GRPO agent protects recognizable assets. |
| Curriculum Learning | `training/curriculum_scheduler.py` | Added Phase 2: agents graduate from fictional Phase 1 tasks before tackling Phase 2 real-world OSM tasks. |
| Adversarial Logic | `adversarial_attacker.py` | Updated attacker to prioritize high-value named assets (e.g., hospitals and power plants). |
| Server Extensions | `server/app.py`, `Dockerfile` | Expanded the FastAPI server to support real-time querying or serving of OSM-backed scenarios. |
| OpenEnv Config | `openenv.yaml` | Registered all new `task_osm_*` scenarios for OpenEnv compatibility. |

#### Why These Changes Improve Training

- **Real-World Relevance:** Agents now train on real city graphs (Tokyo, NYC, London), learning domain-specific infrastructure resilience instead of abstract graph solving.
- **Improved Grounding:** The inclusion of `real_name` gives the language model stronger semantic grounding. Protecting "Bellevue Hospital" invokes stronger safety priors than protecting "node_3".
- **Progressive Difficulty:** The curriculum is now smoothly staggered from simple single-sector fictional graphs up to dense, multi-borough real-world networks.

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
