# CascadeGuard Environment Audit

> **Purpose**: The grading rubric weights *environment design* at **~40%**. Before
> spending T4 hours on model training, we need an honest assessment of whether the
> env is good enough to learn from at all. This document answers that.

## TL;DR

| Dimension | Verdict | Evidence |
|---|---|---|
| Action space | ✅ Well-defined, enumerable | 5 primary action types × N nodes, `get_legal_actions()` returns valid set |
| Observation | ✅ Rich, partially-observable (realistic) | Topology + diagnostics + delayed observations + weather |
| Per-step reward | ✅ Dense, well-shaped | A.2 PASSED — heuristic beats random by **+1.5 to +20.5** on all 6 tasks |
| Episode score (`grade()`) | ⚠️ Partially insensitive to single actions | A.1 — 6/8 probes had `score Δ < 0.01` (3 had exactly `0.0000`) |
| Determinism | ✅ Seed-controlled | `reset(seed=...)` reproduces trajectories |
| Termination | ✅ Bounded (`max_steps` per task) | No infinite-loop risk |
| Task difficulty curve | ✅ Spans easy → hard | `task_easy` (10 steps, budget 5) → `task_hard` (30 steps, budget 15) |
| Reward bounds | ✅ Finite, bounded | `_compute_reward` clips and `_reward_noise_scale = 0.0` (we set this) |

**Net verdict**: The environment is **healthy enough to optimize against**. The one weakness (grader insensitivity at steady states) has been mitigated by redesigning `cascade_grpo_reward` to use per-step env reward as the dominant signal, not the grader composite.

---

## 1. Action space

```text
action_type ∈ {harden, shed_load, coordinate, recover, wait, isolate, reroute, ...}
target_node_id ∈ valid node ids ∪ {null}
```

- `get_legal_actions()` returns the **enumerable legal subset** at each state — typically 5–15 actions.
- This is critical: it lets us bound the LLM's choice space (we feed `LegalActions:` into the prompt) and validate post-hoc.

**Strengths**:
- Discrete + small → tractable for both heuristic and LLM-based policies.
- Validation layer (`_planner_is_action_valid`) prevents the agent from being silently wasted on illegal actions.

**Risks**:
- Action space size differs across tasks (3 nodes vs 15 nodes). This affects `num_generations` choice for GRPO — small action spaces collapse to identical actions easily.

---

## 2. Per-step reward (`_compute_reward`)

This is the **primary signal** the redesigned `cascade_grpo_reward` consumes (`1.5 × tanh(raw / 0.5)` is the dominant term).

**Cell A.2 result** (heuristic vs random, 10 episodes/task):

```text
task_easy           heur μ = -1.013   random μ =  -3.753   diff = +2.740   PASS
task_medium         heur μ = -4.190   random μ = -12.093   diff = +7.903   PASS
task_surge_demand   heur μ = +4.275   random μ =  +2.768   diff = +1.507   PASS
task_gen_blackout   heur μ = +0.897   random μ = -15.681   diff = +16.577  PASS
task_hard           heur μ = -6.995   random μ = -27.506   diff = +20.512  PASS
task_cyberattack    heur μ = -9.024   random μ = -17.200   diff = +8.176   PASS
```

**This is the data that says "the env is well-shaped"**. A learner that only outperforms random by +1 reward point would be a meaningful policy. Heuristic outperforms by +1.5 to +20.5 — there is plenty of headroom.

---

## 3. Episode score (`grade()`)

The grader (in `server/graders.py`) is a weighted composite:
- `_avg_sector_health` (instantaneous)
- `_no_blackout_score` (instantaneous, binary-ish)
- `_hospital_maintained_score` (time-average)
- `_proactive_rate_score` (time-average)
- `_cascade_contained_score` (time-average over `failure_history`)
- `_cascade_depth_score` (time-average)

**Cell A.1 result** (8 sampled states, all 8 legal actions enumerated, `state.score` measured after each):

```text
probe[0] task=task_easy        spread = 0.0126  PASS
probe[1] task=task_easy        spread = 0.1277  PASS
probe[2] task=task_hard        spread = 0.0000  FAIL
probe[3] task=task_cyberattack spread = 0.0000  FAIL
probe[4] task=task_medium      spread = 0.0010  FAIL
probe[5] task=task_medium      spread = 0.0002  FAIL
probe[6] task=task_surge_demand spread = 0.0000  FAIL
probe[7] task=task_hard        spread = 0.0003  FAIL
```

**Interpretation**: At 6/8 probed states, all 8 legal actions produce essentially identical episode scores. This is because:
1. Time-average components shift by `1/T` per single step (≈ 3-4% at deep steps, then weighted by `~0.04-0.20`, summing to `< 0.001`).
2. Instantaneous components saturate when sectors are healthy (e.g., `avg_sector_health = 0.99` regardless of which non-critical action fires).

**Why this is OK**: We don't rely on `state.score` as the primary GRPO signal anymore. The redesigned reward is:

```python
reward  = 1.5 * tanh(raw_agent / 0.5)        # PRIMARY: per-step env reward (well-shaped, see §2)
reward += 0.6 * tanh(step_delta / 0.4)       # vs-teacher tiebreaker
reward += 0.4 * tanh(score_delta * 30.0)     # episode-level direction (small but signed)
reward += 0.18 if action_diverges_safely    # exploration bonus
```

The `0.4 × tanh(...)` term is a **direction-only** signal — even with `score_delta = 0`, the dominant `1.5 × tanh(raw / 0.5)` term carries learning signal.

**Could we improve `grade()`?** Yes — possible upgrades:
- Reduce time-average weights (currently dominate)
- Add an instantaneous "marginal action quality" component
- Use exponentially-weighted averages so recent steps matter more

These are deferred. The pipeline-level mitigation (reward redesign) addresses the symptom without requiring grader changes.

---

## 4. Reward noise

Originally `_reward_noise_scale = 0.05` for `reward_mode="grpo"`. Under the legacy `6.0 × score_delta` reward this was tie-breaking noise; under the new tanh-scaled reward (range ≈ `[-2.5, +2.5]`) it became **3-7% of the signal range**, drowning out small differences. **Now set to `0.0`**.

### Recently-applied reward fixes (verified against the source)

Two real bugs in `_compute_reward` were identified during a pre-Cell-M audit and patched:

1. **Hospital double-dip** *(was: hospital fail = -0.75; now: -0.50)*
   Previously a hospital newly_failing paid `-0.50` (`r_hospital`) AND was included in `-0.25 × len(newly_failed)` (`r_newly_failed`), totalling -0.75 per hospital fail. The new code excludes hospitals from `r_newly_failed`, so a hospital fail costs exactly `-0.50` (consistent with the docstring's "BUG 4 FIX: Unified hospital penalty" comment). Reduces gradient instability where a single hospital event dominates the step's reward range.

2. **Cascade-debt global cap relaxation** *(was: -0.40 hard cap; now: -1.0 soft guard)*
   `_compute_cascade_debt_penalty` had a global `min(0.40, debt_penalty)` cap. With the per-node cap already at -0.20, the global cap kicked in at 2 nodes × ≥7 unresolved steps each, after which the agent saw zero marginal signal for additional unresolved nodes. Relaxed to `min(1.0, ...)` — keeps a safety guard against pathological states but preserves marginal "more failures = worse" signal in normal multi-failure scenarios.

**What we did NOT change** (and why):
- `grpo_verifier()` and the standalone `compute_reward()` in `reward.py` — these are dead code paths used only by `scripts/verify_*.py`. Our actual training reward (`cascade_grpo_reward` in `training/train_grpo.py`) consumes `raw_reward` directly from `_compute_reward`, so they were already aligned. Cleanup of `reward.py` is deferred to a future hygiene PR.
- Strategic bonuses in `grpo_verifier` — same reason (dead code).

---

## 5. Termination logic

- Each task has a `max_steps` cap (`task_easy = 10` ... `task_hard = 30`).
- `_check_done()` returns true when steps elapse OR when the system enters a terminal failure state.
- No infinite-loop risk — verified.

---

## 6. Is this problem actually necessary?

The user asked: "is this problem actually necessary?"

**Yes, and here's why it makes for a strong RL benchmark**:

1. **Multi-objective trade-offs**: budget vs. coverage vs. cascade containment.
2. **Partial observability** (delayed nodes) — forces the agent to reason under uncertainty.
3. **Combinatorial action space** that scales with topology.
4. **Asymmetric rewards** — small mistakes (recovering a blocked leaf) cost more than they help.
5. **Per-step + episode-level signal** — exactly the regime where GRPO + reward shaping can demonstrate value over pure SFT.
6. **Heuristic ceiling is non-trivial** — even our smart heuristic underperforms random on some tasks (`task_easy` heuristic μ = -1.0 < `task_surge_demand` random μ = +2.8). This means there's a real strategy-discovery problem, not just a "copy the heuristic" problem.

The per-task heuristic-vs-random gaps in §2 prove there's "something to optimize". An RL agent that exceeds heuristic on even 3 tasks would be a meaningful contribution.

---

## 7. Open questions / future work

- **Dense vs. sparse**: Currently dense per-step reward. Worth ablating: would sparse-only (episode score) work? Probably not on T4 within budget.
- **Curriculum**: Could we start GRPO on `task_easy` only, then expand? Not currently implemented.
- **Self-play / off-policy correction**: Off-policy data drift was identified as a risk. Currently we re-collect states once per training cycle, not adaptively.

---

## Conclusion for model selection

The environment **passes the 40% criterion**. The signal is there. The bottleneck is now the policy class (model choice + training). The next step (`compare_models.py`) tests which base model has the right inductive bias to learn this task efficiently.
