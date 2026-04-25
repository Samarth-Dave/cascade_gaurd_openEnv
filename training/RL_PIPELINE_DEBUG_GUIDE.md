# RL Pipeline Debug Guide — Diagnosed Problems and Verified Fixes

This document is the post-mortem and runbook for the CascadeGuard GRPO pipeline.
It traces every problem we hit, what we observed, which classic RL-debugging
principle it maps to (Andy Jones, "Debugging RL Without the Agonizing Pain";
the senior-mentor TPM article on RL pipeline failures; common beginner-mistake
articles), what we changed, and **how to manually verify each fix**.

> If you only have 5 minutes, read **§ Executive summary** + **§ 15-minute
> manual verification checklist**. If you have 20 minutes, read everything.

---

## Executive summary

**Symptom we kept seeing**: SFT loss decreased cleanly from 1.9 → 1.1, but
GRPO `loss = 0` and `reward = 0` for every step.

**Why classic answers ("raise LR / lower KL / tune hyperparameters") didn't
work**: the bottleneck was upstream of the algorithm. Per the senior mentor:

> Most reinforcement learning pipeline failures are decided before a single
> gradient update runs. […] By the time training begins, the outcome is
> often already written.

We diagnosed **four upstream pipeline failures**, each of which alone would
suppress the GRPO learning signal, and they were all firing simultaneously:

| # | Upstream failure | Maps to |
|---|---|---|
| 1 | One-step grader signal is bounded by `1/T` of an already-small composite | Reward misspecification (mentor §"Reward Function Failures") |
| 2 | Post-SFT model copies the recorded teacher action ⇒ `score_delta ≡ 0` by construction | Sim-to-real mismatch / off-policy drift (mentor §"Data problem") |
| 3 | TRL's `scale_rewards="group"` divides by within-group std, which is 0 when (1) and (2) hold | Misconfigured normalization (Andy Jones §"Use a really large batch size", §"Logging — entropy") |
| 4 | States are collected once from a heuristic teacher ⇒ agent never sees its own distribution | Off-policy data drift (mentor §"off-policy data drift") |

**Eight changes shipped** (three repo files + five notebook insertions) — all
described below with their verification step.

---

## How to use this document

Every fix below has the same structure:

> **Observation**: what we saw in the logs
>
> **Diagnosis**: which RL-debug principle it maps to + the actual root cause
>
> **Fix**: file path + code reference + reasoning
>
> **Manual verification**: a concrete check you can run today

You can grade each fix as `PASS` / `WARN` / `FAIL` independently of the others.

---

## Problem 1 — "Treating RL like supervised learning"

> Beginners often expect: "more data → better results." In RL, your agent
> creates its own dataset by interacting with the environment.

### Observation
We were judging GRPO success by the same metric as SFT (decreasing loss
curve). When the loss stayed flat we kept tweaking SFT-style knobs (more
data, more epochs, higher LR). Loss stayed at 0.

### Diagnosis
Andy Jones is brutal about this:

> **Loss curves are a red herring.** […] The shape of your loss curve says
> very little about where in your code you've messed up.

In RL, "loss = 0" can mean:
- The reward signal is constant within each group (advantage = 0)
- The KL term cancels the policy term
- The reward function is correct but the policy already maximises it
- The gradient never reaches the trainable params (frozen / dtype mismatch)

A flat loss curve does not localise the error. We were optimising the
wrong telescope.

### Fix
Stop relying on loss curves alone. We added **three new diagnostic cells**
to the notebook (Cells A, B, C) and **rich per-call telemetry inside the
reward function** that surfaces *which component carries variance*.

References:
- `training/train_grpo.py` `cascade_grpo_reward` (the new
  `GRPO[N] n=K mu=… sd=… unique_act=… teacher_match=… score_d=[…] step_d=[…]`
  log lines)
- `training/CascadeGuard_TRL_Colab_fixed_(1) (1).ipynb` cells 25 / 28 / 30

### Manual verification
1. After running Cell A, look for two banners ending in `A.1 PASS` and
   `A.2 PASS`.
2. After Cell B, look for `Cell B PASS` (or a clear WARN explaining what to
   raise).
3. During training, scan the console for `GRPO[N]` lines. You should see
   `sd >= 0.05` inside the first 10 calls. If you see
   `ZERO-VARIANCE GROUP` or `ALL completions matched teacher action exactly`,
   the WARNING text tells you exactly which knob to turn.

---

## Problem 2 — Reward misspecification: score-delta near zero by construction

> Reward shaping is powerful — and dangerous. […] If the reward is the only
> truth the agent sees, it will exploit it like a lawyer reading loopholes.

### Observation
The legacy reward was

```python
reward = 6.0 * score_delta + 0.35 * step_delta + format_bonuses
```

Where `score_delta = state.score_agent - state.score_teacher` after **one**
env step. But `state.score` is computed by `server/graders.py::grade()`
over the **whole episode history** (proactive_rate, hospital_maintained,
cascade_contained — all time-averages).

A one-step difference shifts a time-average by `1/T`. For `task_hard`
(`max_steps=30`) with `proactive` weight `0.04`:

- Composite-score change ≤ `0.04 × 1/T` ≈ `0.0013` at T=30
- After `6.0 ×` scaling: ≈ `0.008`
- Format bonuses (constant across the group): ±0.20

The "main objective" was 25 × smaller than the constant noise floor.

### Diagnosis
Mentor's framework (#1 cause of zero-reward barriers):

> Improper Reward Scaling: If reward values are too large or too small
> relative to the neural network's initial outputs, the agent will either
> explode or fail to update.

Andy Jones, "Hand-tune your reward scale":

> The single most common issue for newbies writing custom RL implementations
> is that the targets arriving at their neural net aren't [-1, +1].
> Anything [-.1, +.1]ish to [-10, +10]ish is good.

### Fix
Rewrote `cascade_grpo_reward` in
`training/train_grpo.py` so the **per-step env reward** is the dominant
signal (it actually varies per action: hospital save +0.08, hospital fail
-0.50, recovery +0.30, cascade impact -N), and the score-based terms become
small bounded tiebreakers:

```python
# tanh squashes outliers; 0.5 ≈ env's natural per-step magnitude.
reward = 1.5 * math.tanh(raw_agent / 0.5)        # absolute step quality (high-variance)
reward += 0.6 * math.tanh(step_delta / 0.4)      # vs-teacher tiebreaker
reward += 0.4 * math.tanh(score_delta * 30.0)    # episode-level direction
```

Each term is hard-bounded so a single catastrophic completion can't dominate
the group. Total reward range stays in roughly `[-2.5, +2.7]`, well inside
Andy Jones' "[-10, +10]" sanity zone.

### Manual verification
1. Open `training/train_grpo.py`, jump to `cascade_grpo_reward`. The first
   reward assignment must be `reward = 1.5 * math.tanh(...)`. The legacy
   `6.0 * score_delta + 0.35 * step_delta` line is gone.
2. After running 5+ GRPO steps, the `GRPO[N]` log lines should show
   `mu` in `[-2.0, +2.0]` and `sd >= 0.05`. If `mu` is consistently in
   `[-3.5, -2.5]`, the format/parseability penalties are dominating —
   inspect a few raw completions to make sure they parse.
3. Inside Cell A.1 output, the per-state score spread should be in the
   `0.001 – 0.05` range. That confirms `score_delta` is small (as expected)
   and validates that we **needed** to demote it.

---

## Problem 3 — Sim-to-real / data-pipeline trap: model is the teacher

### Observation
The SFT dataset (`cascadeguard_sft_dataset.csv`) was generated from
heuristic-policy rollouts. The GRPO `teacher_action_json` recorded during
state collection comes from the same heuristic-family policies. Result:

```
SFT trains model to imitate teacher
   →  post-SFT model.action == teacher.action  (most of the time)
   →  agent_env.step(...) == teacher_env.step(...)
   →  score_agent == score_teacher
   →  score_delta == 0  by construction
```

This is the #1 cause. The model literally cannot get a signal because it's
being scored against itself.

### Diagnosis
Mentor's framework calls this out explicitly:

> Simulation-to-real mismatch in training data that produces policies which
> fail outside the training environment […]

A more general principle: **anytime your reward compares the policy against
something the policy is being trained to imitate, the reward will collapse**.
We need either an absolute reference (env reward) or a reward for *deviating
when deviation is safe*.

### Fix
1. Demoted the teacher-relative term to a small tiebreaker (Problem 2).
2. Added an explicit **exploration bonus** in `cascade_grpo_reward`:

```python
agent_act_key = (agent_action.action_type, agent_action.target_node_id)
teacher_act_key = (teacher_action.action_type, teacher_action.target_node_id)
if agent_act_key != teacher_act_key and agent_meta.get("action_valid", True):
    if critical_after <= critical_before and failed_after <= failed_before + 1:
        reward += 0.18  # try something new and don't make it worse
```

Translated: the agent gets +0.18 for choosing **any** different action from
the teacher, so long as the result isn't catastrophically worse. This
creates positive gradient *toward divergence* exactly when SFT pulls *toward
imitation*.

### Manual verification
1. Cell B prints `agent==teacher rate`. Pass: `<= 0.70`. If the rate is
   `> 0.85`, SFT has fully over-fit; either re-run SFT for fewer epochs or
   trust the exploration bonus to push divergence over time.
2. During GRPO, the `GRPO[N]` lines show `explore=K/N`. Healthy training:
   `K` between 1 and (N-1) most steps. `K = 0` for many consecutive steps
   means the model has collapsed onto teacher behaviour.
3. If you see the WARNING `ALL completions matched teacher action exactly`,
   that's a hard failure of this fix — raise temperature, lower SFT epochs,
   or both.

---

## Problem 4 — Misconfigured normalization: TRL `scale_rewards="group"`

> Failing to normalize observations or state information properly causes the
> network to receive inconsistent inputs, making learning slower and unstable.

### Observation
Even with a perfectly designed reward, GRPO loss stayed at 0. Reading TRL's
source we found that `GRPOConfig` defaults to `scale_rewards="group"`,
meaning advantages are computed as

```
advantage = (reward - group_mean) / group_std
```

When all completions in a group parse to the same action (Problem 3),
all rewards are equal → `group_std = 0` → division by ~0 → advantages
collapse to 0 (or NaN, then masked) → zero gradient → flat loss.

This is a **silent** failure: TRL doesn't warn you, the loss just sits at 0.

### Diagnosis
Andy Jones in "Logging excessively → Relative policy entropy":

> If it drops to zero or close to zero, then your agent has 'collapsed' into
> some — likely myopic — policy, and isn't exploring any more. […] usually
> because you've […] forgotten to include an exploration mechanism, or
> because your rewards are much larger than whatever you're using to
> encourage exploration.

We were the second case: the format bonus (constant across group) was much
larger than the exploration signal (zero across group).

### Fix
In `GRPOConfig` (notebook cell 28):

```python
scale_rewards = False    # CRITICAL — stop dividing by group_std
beta          = 0.04     # small KL anchor against ref model
epsilon       = 0.2      # PPO clip
temperature   = 1.0      # was 0.9
top_p         = 0.95
top_k         = 50
```

Wrapped in an `inspect.signature` filter so older TRL versions silently
drop unsupported kwargs (and tell you which) instead of crashing.

`scale_rewards=False` is the **single most important config knob**. Even
small absolute reward differences now flow through to the gradient.

### Manual verification
1. Open notebook cell 28 (after the modifications). Look for the
   `_filtered_kwargs` build pattern. The `scale_rewards = False` line must
   be present in `_gc_kwargs`.
2. Run the cell. The first thing it prints is either nothing (all kwargs
   accepted) or `[GRPOConfig] dropped kwargs unsupported by this TRL version: [...]`.
   If `scale_rewards` is in the dropped list, **upgrade TRL**:
   `pip install -U "trl>=0.24.0"`. This fix is critical and version-gated.
3. During training, the `GRPO[N]` lines should show `sd >= 0.05` reasonably
   often. With `scale_rewards=True` and a flat group, `sd ≈ 0` would still
   show in the reward log but the underlying advantages would be 0; with
   `scale_rewards=False`, even `sd = 0.02` produces a usable gradient.

---

## Problem 5 — Off-policy data drift: states from heuristic only

> Off-policy data drift where stale trajectories degrade training quality
> over time.

### Observation
`collect_training_states()` rolls heuristic-family policies once at the
start to build the GRPO dataset. The agent learns from those states for
the entire training run, never visiting any state its current policy
naturally generates.

### Diagnosis
This is the textbook off-policy drift the mentor article warns about:

> If the data pipeline isn't collecting fresh on-policy data at the right
> rate, […] the RL agent is effectively training on its own outdated
> behavior. Performance gains stall, and the team assumes the algorithm
> needs adjustment when the pipeline simply needs better data hygiene.

### Fix (partial)
We did not implement full periodic on-policy refresh — that's listed as
Phase 4 in the plan and only triggers if the other fixes don't restore the
gradient. What we did do:

1. The state-collection loop in notebook cell 24 already does **balanced
   per-task collection** + shuffle (`states_per_task = STATES // len(TASKS)`).
   This prevents the dataset from being dominated by easy tasks.
2. Cell A.1 verifies that each sampled state has a meaningful score
   spread across legal actions. If too many states are "converged" (low
   spread), the dataset itself is the problem.

### Manual verification
1. Cell 24 should print a balanced task distribution (e.g., 20 states per
   task for STATES=120). If `task_easy` shows 50+ and others show 0–5, the
   balanced collection didn't run.
2. Cell A.1 reports `passing probes (spread > 0.01)`. If this is `<= 1/8`,
   the dataset has too many "dead" states; consider increasing `STATES` or
   sampling earlier-step states (currently `collect_training_states` favours
   mid-episode states which are more likely to be in-crisis).

---

## Problem 6 — Broken termination logic / env sanity (mentor's #7)

> RL agents can "learn" around bugs. That's the worst part.

### Observation (resolved earlier)
Earlier in the project we found two bugs in this category:
- `server/graders.py::grade()` was over-counting critical failures by
  summing `len(failure_history[t])` across steps instead of counting unique
  failed nodes. This pushed `_critical_failure_penalty` to its cap on
  every episode, flooring scores at `0.01` for both agent and teacher and
  collapsing `score_delta` to 0.
- `server/cascade_environment.py::_reward_noise_scale = 0.05` for GRPO mode
  added Gaussian noise sized for the legacy reward; under the new
  `1.5 * tanh(raw / 0.5)` term it became 3–7% of the signal range.

### Fix
- `server/graders.py::grade()` now counts unique failed node IDs and caps
  the deduction at `0.50`.
- `server/cascade_environment.py` line 196 sets `_reward_noise_scale = 0.0`
  with a comment explaining why noise is no longer needed.

### Manual verification
1. In `server/graders.py::grade()`, find the `if explicit_cf is None` block.
   It must build `unique_failed: Set[str]` from `failure_history` and use
   `len(unique_failed)`, not `sum(len(s) for s in failure_history)`.
2. In `server/cascade_environment.py`, line 196 must read
   `self._reward_noise_scale = 0.0` (not the conditional `0.05 if grpo`).
3. Cell A.1 spreads should now span a meaningful range (typical:
   `[0.05, 0.25]`). If they all cluster around `0.01`, the grader is still
   flooring scores — re-check the `unique_failed` change.

---

## Problem 7 — No black boxes / few narrow interfaces (Andy Jones)

> RL has surprisingly few of these black boxes. […] You're required to know
> how your environment works, how your network works, how your optimizer
> works, how backprop works, how multiprocessing works, how stat collection
> and logging work.

### Observation
Before the diagnostic cells, when GRPO loss was 0 we had to read TRL
internals + the reward function + the environment + the grader to even
*localize* the failure. We had no way to tell "is the model not learning"
from "is the reward dead" from "is the env returning garbage".

### Diagnosis
Andy Jones' "Localise errors" + the **probe environments** technique:

> Construct environments that do localise errors. […] One action, zero
> observation, one timestep long, +1 reward every timestep — this isolates
> the value network. If my agent can't learn that, there's a problem with
> the value loss calculation or the optimizer.

We can't easily build minimal envs around our domain, but we can apply the
same principle: **probe agents and probe states**.

### Fix
Three new notebook cells, each acting as a probe:

| Cell | Probes | Localises |
|---|---|---|
| A.1 | Enumerate every legal action on 8 sampled states; step a clone for each; measure score spread | Whether the **grader** distinguishes actions |
| A.2 | Heuristic policy vs random_legal policy on 10 episodes per task | Whether the **env reward** is well-shaped |
| B   | Generate `num_generations` completions for 8 dataset rows; parse to actions | Whether the **model** produces diverse outputs at the chosen temperature |
| C   | Trained model vs heuristic vs random_legal on 5 seeds × 6 tasks | Whether **GRPO actually moved the policy** |

Plus the in-reward telemetry that Andy Jones calls "Log excessively":
- mean / sd of group rewards (≈ value distribution)
- unique parsed actions (≈ relative policy entropy)
- teacher-match rate (≈ KL div to ref)
- score_delta range (≈ residual variance for grader)
- step_delta range (≈ value target distribution)
- explore bonus count (≈ action entropy proxy)

### Manual verification
You don't need to read GRPO internals anymore. Read in this order:

1. Cell A output → did it print `A.1 PASS` and `A.2 PASS`?
2. Cell B output → did it print `Cell B PASS`?
3. First 5–10 `GRPO[N]` log lines → is `sd > 0.05`?
4. Cell C output → does the trained model beat random everywhere?

If all four are PASS, GRPO is genuinely working. If any single one fails,
the failing cell tells you exactly which subsystem is broken.

---

## In-depth techniques applied (mapped to Andy Jones)

| Andy Jones technique | What we did |
|---|---|
| **Hand-tune reward scale** | Replaced unbounded `6.0 × score_delta` with three `tanh`-bounded terms, each with explicit scale. Total reward in `[-2.5, +2.7]` |
| **Use a really large batch size** | Increased `num_generations` to 6 (within T4 memory budget). More samples per group → higher chance of within-group variance |
| **Use a really small network** | Already using Qwen-0.5B + LoRA on T4 |
| **Avoid pixels** | Already text-only state representation |
| **Mix vectorised envs** | We don't run vec envs, but Cell 24 shuffles the dataset to mix tasks across batches |
| **Probe environments** | Cell A.1 enumerates legal actions on real states (closest analogue) |
| **Probe agents** | Cell A.2 (random_legal as floor) and Cell C (heuristic as ceiling) |
| **Log excessively** | New `GRPO[N]` telemetry covers reward sd, action diversity, teacher overlap, signal range, exploration count |
| **Loss curves are red herring** | We stopped relying on loss alone; the four diagnostic cells + telemetry give independent signals |
| **Pursue anomalies** | The whole "loss = 0" investigation followed Andy's "Don't hope it goes away" |
| **Assume you have a bug** | We found four compounding bugs upstream of the algorithm |

---

## 15-minute manual verification checklist

Run this against the notebook end-to-end. Each row is a discrete pass/fail.

| # | Question | How to check | Pass criterion |
|---|---|---|---|
| 1 | Can a random policy get non-zero reward sometimes? | Cell A.2 output, `random μ` column | At least one task has `random μ ≠ 0` |
| 2 | Does reward correlate with obvious progress? | Cell A.2, compare `heur μ` and `random μ` | Heuristic beats random on ≥ 3/6 tasks |
| 3 | Do episodes terminate correctly? | Cell 24 already calls `run_env_sanity_check` | "All tasks passed environment sanity check" line printed |
| 4 | Can the agent overfit a tiny version of the task? | SFT loss curve | Loss decreases monotonically (you saw 1.9 → 1.1) |
| 5 | Does the grader distinguish actions? | Cell A.1, `passing probes (spread > 0.01)` | At least 4 / 8 |
| 6 | Does the model produce diverse completions? | Cell B, `mean unique-action ratio` | `>= 0.40` |
| 7 | Does the model copy the teacher? | Cell B, `agent==teacher rate` | `<= 0.70` |
| 8 | Do completions parse? | Cell B, `parseable completions` | `>= 0.80` |
| 9 | Does GRPO see within-group variance? | First 5 `GRPO[N]` log lines | At least one with `sd >= 0.05` |
| 10 | Are TRL's GRPOConfig kwargs accepted? | Notebook cell 28 first print after `_filtered_kwargs` | Either no print or `dropped: []` |
| 11 | Does GRPO move the policy? | Cell C summary table | Trained beats random on all 6 tasks |
| 12 | Does GRPO beat the heuristic anywhere? | Cell C summary table | Trained beats heuristic on ≥ 3/6 tasks |
| 13 | Are there no zero-variance group warnings? | scan training log | Few or zero `ZERO-VARIANCE GROUP` warnings |
| 14 | No `ALL completions matched teacher` warnings? | scan training log | Few or zero of these |
| 15 | No `GRADER INSENSITIVE` warnings? | scan training log | Few or zero of these |

---

## What to do when each check fails

| Failed check | Most likely cause | What to change |
|---|---|---|
| #1 random ≡ 0 | Env reward only fires on rare events | Inspect `_compute_reward` in `server/cascade_environment.py`; rewards are per-step, not terminal |
| #2 heuristic ≤ random | Env reward is misshaped or noise > signal | Re-examine `_compute_reward` weights; verify `_reward_noise_scale = 0.0` |
| #5 spread ≈ 0 | Grader's components are all clamped or capped at this state | Check graders.py weights; consider raising weight on `_avg_sector_health` (instantaneous, sensitive) |
| #6 unique_actions < 0.40 | Sampling temperature too low | Raise `temperature` toward 1.2–1.5 in GRPOConfig; verify `top_p` and `top_k` were accepted by TRL |
| #7 teacher_match > 0.70 | SFT over-fit teacher | Reduce SFT epochs (currently uses entire CSV); the in-reward exploration bonus (+0.18) helps but only over many GRPO steps |
| #8 parseable < 0.80 | Format prompt is unclear or `MAX_COMP_LEN` too short | Inspect a few raw completions; raise `MAX_COMP_LEN`; tighten the format example in the system prompt |
| #9 sd < 0.05 always | Either #6 or #7 is the root cause | Fix the upstream issue first |
| #10 dropped kwargs | TRL too old | `pip install -U "trl>=0.24.0"` then restart |
| #11 trained < random somewhere | GRPO destabilised the policy | Lower `LR_GRPO` (try 5e-6); shorten `GRPO_STEPS`; check `max_grad_norm = 0.30` is being hit (look for clipped gradient warnings) |
| #12 trained < heuristic everywhere | More training needed (not necessarily a bug) | Increase `GRPO_STEPS`; verify Cell B passed (no diversity collapse during training) |

---

## What success looks like (acceptance criteria for "GRPO is working")

You can reasonably claim the pipeline is healthy when **all** of these hold:

- Cell A: `A.1 PASS` and `A.2 PASS`
- Cell B: `Cell B PASS` (all three sub-thresholds met)
- During GRPO training:
  - Console shows `GRPO[N]` log lines with `sd` between `0.05` and `1.5`
    most of the time
  - At most 1 in 5 calls shows a `ZERO-VARIANCE GROUP` warning
  - `unique_act` ≥ 2 / `num_generations` on most calls
  - Loss is non-zero and (slowly) decreasing
- Cell C: trained model beats random on **all 6** tasks AND beats
  heuristic on **at least 3 of 6** tasks

---

## Reference: files changed in this round

| File | Change |
|---|---|
| `training/train_grpo.py` | Rewrote `cascade_grpo_reward`: tanh-bounded terms, exploration bonus, rich telemetry with three distinct WARNING types |
| `server/cascade_environment.py` | Set `_reward_noise_scale = 0.0` always (was `0.05 if grpo`) |
| `training/CascadeGuard_TRL_Colab_fixed_(1) (1).ipynb` cell 25 (NEW) | Cell A — pre-training signal sanity (grader sensitivity + heuristic vs random) |
| `training/CascadeGuard_TRL_Colab_fixed_(1) (1).ipynb` cell 28 | Hardened GRPOConfig (scale_rewards=False, beta, epsilon, top_p, top_k, temperature=1.0) + embedded Cell B diversity check |
| `training/CascadeGuard_TRL_Colab_fixed_(1) (1).ipynb` cell 30 (NEW) | Cell C — post-training validation (trained vs heuristic vs random on 5 seeds × 6 tasks) |

---

## Reference: which RL pitfall maps to which fix

| Beginner-mistakes article pitfall | Mentor's pipeline failure | Andy Jones technique | Our fix |
|---|---|---|---|
| #1 Treating RL like SL | (none — meta) | "Loss curves are a red herring" | Cells A/B/C + telemetry |
| #4 Reward shaping that teaches the wrong behavior | "Reward function misspecification" | "Hand-tune reward scale" | Tanh-bounded reward + exploration bonus |
| #7 Broken termination + reward bugs | (mentor's "data problem") | "Unit test the tricky bits" | graders.py CF-count fix; env reward noise → 0 |
| #10 Compute, not cleverness, is the bottleneck | (mentor's TPM section) | "Use a really small network + large batch" | Already on 0.5B + LoRA + GENERATIONS=6 |
| (zero-reward barrier) | "Sparse rewards" | "Probe environments" | Cell A.1 enumerates legal actions |
| (zero-reward barrier) | "Sim-to-real mismatch" | "Pursue anomalies" | Diagnosed teacher-imitation trap; added exploration bonus |
| (zero-reward barrier) | "Off-policy data drift" | "Mix vectorised envs" | Balanced per-task state collection + shuffle (already in cell 24) |
| (config bug) | "Misconfigured normalization" | (logging — entropy collapse signature) | `scale_rewards=False`, `beta=0.04` |

---

## Open items (not yet implemented — only if needed)

If the above fixes still leave loss flat after a clean run:

1. **Periodic on-policy state refresh**: add a `refresh_states()` helper to
   `training/train_grpo.py` that uses the *current* model to roll N partial
   trajectories every K GRPO steps and merge into the dataset. Listed as
   "Phase 4" in the original plan; skipped because the other fixes should
   restore the gradient first.
2. **Per-component reward decomposition logging**: currently we log totals.
   We could break out the contribution of each tanh term, the parseability
   penalty, the exploration bonus, etc. on every K-th call. Useful only if
   we see specific reward levels that don't match expectations.
3. **Curriculum**: start with `task_easy` only for the first ~50 GRPO
   steps, then enable harder tasks. Helps when zero-reward barrier persists
   on harder tasks even after the above fixes.

These are deliberately not implemented — they trade complexity for marginal
gain, and the diagnostic infrastructure we've added will tell us *whether*
they're needed before we spend the implementation time.
