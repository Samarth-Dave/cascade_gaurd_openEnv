# CascadeGuard GRPO Refactor Notes

Date: 2026-04-23

## What changed

### 1) Reward now measures relative causal impact
- File: training/train_grpo.py
- `make_grpo_reward_fn(...)` now compares the model action against a teacher baseline from the same reconstructed state.
- For each sample:
  - replay state from `task_id`, `seed`, `history_json`
  - apply model action in one env and teacher action in a cloned replay env
  - compute `score_delta = score_agent - score_teacher`
  - add a small step-level delta term from environment `raw_reward`
- Base reward is now anchored to relative improvement, not absolute score on teacher-prefix states.

Reward core:
- `reward = 6.0 * score_delta + 0.35 * step_delta`
- then small shaping terms (format, invalid action, wait laziness, recover bonus, budget spend)
- then group normalization with a floor to avoid all-zero collapse when variance is tiny

### 2) Parseability is now high-priority
- File: training/train_grpo.py
- Added shared `_looks_parseable_action(...)` and used it in both training reward and evaluation.
- Unparseable completion gets a strong penalty (`-1.50`) so fallback-to-wait is no longer cheap.
- Stronger positive reward for correct action formatting:
  - tag + function format bonus
  - JSON action format bonus

### 3) Prompt/format mismatch removed between train and eval
- Files: training/train_grpo.py, training/cot_prompt.py
- Training state collection now uses canonical `cot_prompt.make_training_prompt(...)`.
- Evaluation policy now also uses canonical `make_training_prompt(...)` with live legal actions from env.
- The legal-action block in `cot_prompt.py` now uses stratified sampling by action type instead of first-N truncation, so advanced legal actions are represented when available.

### 4) Teacher and exploration now cover advanced actions
- File: training/train_grpo.py
- Extended `make_training_sub_policies(...)` with guarded advanced interventions for:
  - `patch_scada` in cyberattack pressure windows
  - `reroute` and `cross_sector_bridge` in backbone/hospital stress patterns
  - `deploy_repair_crew`, `prioritize`, `multi_sector_lockdown`, and conservative `controlled_cascade` use
- Added safety guardrails to avoid harmful advanced usage (especially around hospital/critical targets).
- Exploration now has advanced-action bias when safe advanced actions are legal.

### 5) Optional SFT warm-start hook
- File: training/train_grpo.py
- Added CLI:
  - `--sft-warmstart-steps`
  - `--sft-warmstart-lr`
- If enabled, an SFT stage runs before GRPO on prompt -> teacher action pairs from collected states.

## Baseline (before refactor)

From latest reported eval CSVs:
- `grpo_local mean_score ~= 0.33` vs heuristic `~= 0.73`
- `grpo_local success_rate ~= 0.33` vs heuristic `~= 0.67`
- `grpo_local mean_budget_spent = 0.0`
- `grpo_local wait_rate = 1.0`
- `grpo_local parse_failure_rate = 1.0`

## Verification commands

### Smoke test

```bash
python training/train_grpo.py --smoke-test --no-eval
```

Expected checks:
- run completes end-to-end
- reward smoke test log is printed
- reward values are non-trivial (not all identical zeros)

### Short run (example)

```bash
python training/train_grpo.py --steps 200 --states 384 --generations 8
```

Optional warm-start:

```bash
python training/train_grpo.py --steps 200 --states 384 --generations 8 --sft-warmstart-steps 80
```

## Before/after metric table (fill after run)

| Metric | Before | After (short run) |
|---|---:|---:|
| mean_score (grpo_local) | ~0.33 | |
| success_rate (grpo_local) | ~0.33 | |
| mean_budget_spent (grpo_local) | 0.0 | |
| wait_rate (grpo_local) | 1.0 | |
| parse_failure_rate (grpo_local) | 1.0 | |

Target direction:
- parse failures near zero
- wait rate significantly below 1.0
- non-zero budget spend
- score gap vs heuristic reduced, ideally matched on at least some tasks
