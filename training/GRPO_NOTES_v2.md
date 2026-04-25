# CascadeGuard GRPO Notes v2

Date: 2026-04-23

## Objective

Second tuning pass focused on four goals:

1. Reduce parse failures and overlong outputs.
2. Improve budget usage shaping by task difficulty.
3. Stabilize hard-task behavior (especially cyberattack and blackout stress).
4. Make GRPO optimization more stable and observable.

## Changes Implemented

### A) Parseability and output-format hardening

Files:
- training/cot_prompt.py
- training/train_grpo.py

Key updates:
- Added strict GRPO action-only system prompt mode.
- Added one-line output contract for action-only mode.
- Threaded `action_only` prompting through state collection and eval generation.
- Lowered default `--max-completion-len` from 128 to 32.
- Increased default SFT warm-start:
  - `--sft-warmstart-steps 180`
  - `--sft-warmstart-lr 1.5e-5`
- Added parseability debug snapshot flags:
  - `--debug-parseability`
  - `--debug-parseability-samples`
- Snapshot utility writes JSON samples to:
  - `cascadeguard_grpo_local/debug/parseability_pre_grpo.json`
  - `cascadeguard_grpo_local/debug/parseability_post_grpo.json`

### B) Reward shaping for budget and task context

File:
- training/train_grpo.py

Reward updates (`make_grpo_reward_fn`):
- Stronger unparseable-output and weak-format penalties.
- Stronger anti-wait penalties when better legal actions exist.
- Task-dependent budget shaping:
  - Higher spend-delta and utilization weights for:
    - `task_surge_demand`
    - `task_cyberattack`
    - `task_hard`
    - `task_gen_blackout`
  - Lighter budget pressure for `task_easy`.
- Added bounded overspend penalties and strong underspend penalties when teacher spending indicates required intervention.
- Added task-specific shaping hooks:
  - `task_surge_demand`: reward critical-failure reduction + intervention actions; penalize passive wait under pressure.
  - `task_easy`: penalize unnecessary extreme expert actions; small reward for stable valid actions.
  - hard tasks (`task_hard`, `task_gen_blackout`, `task_cyberattack`): nudge reroute/bridge/patch/repair when they reduce critical failures.
- Added pre-normalization reward stats logging every N calls:
  - `--reward-stats-every` (default 20)

### C) Hard-task collapse diagnostics and teacher refinement

File:
- training/train_grpo.py

Teacher-policy refinements:
- Cyberattack:
  - Earlier and more consistent `patch_scada` preference.
- Gen blackout:
  - Better use of `deploy_repair_crew` on pending critical generation/transmission nodes.
  - Stronger hospital-protective `reroute`/`cross_sector_bridge` preference.
- Expert-action safety gates tightened:
  - `multi_sector_lockdown` and `controlled_cascade` now require more severe pressure/failure conditions and no recover legal option.

Debug exporter:
- Added hard-rollout export mode with per-step diagnostics:
  - `--debug-hard-rollouts`
  - `--debug-hard-rollout-seeds`
- Outputs:
  - `cascadeguard_grpo_local/debug/hard_rollouts_episodes.csv`
  - `cascadeguard_grpo_local/debug/hard_rollouts_episodes.json`
  - `cascadeguard_grpo_local/debug/hard_rollouts_steps.csv`
  - `cascadeguard_grpo_local/debug/hard_rollouts_steps.json`
- Includes policy comparisons for:
  - `heuristic`
  - `grpo_local`
  - `advanced_crisis_manager` (debug)
  - `proactive_hardener` (debug)

### D) GRPO stability tuning

File:
- training/train_grpo.py

Defaults and optimizer setup:
- `--steps` default increased to 800.
- `--lr` default increased to `8e-6`.
- GRPO config now uses:
  - `gradient_accumulation_steps=2`
  - `temperature=0.55`
  - `warmup_ratio=0.08`
  - `lr_scheduler_type="cosine"`
  - `max_grad_norm=0.30`

## New CLI Flags Added

- `--action-only-prompts` / `--no-action-only-prompts`
- `--debug-parseability`
- `--debug-parseability-samples`
- `--debug-hard-rollouts`
- `--debug-hard-rollout-seeds`
- `--reward-stats-every`

## Validation Completed In This Iteration

- Python compile check passed for:
  - `training/train_grpo.py`
  - `training/cot_prompt.py`
- CLI startup and flag registration check passed:
  - `python training/train_grpo.py --help`
- End-to-end smoke train+eval completed:
  - `python training/train_grpo.py --smoke-test --eval-seeds 1 --debug-parseability --debug-hard-rollouts --debug-hard-rollout-seeds 1`

## Observed Smoke-Run Results (new code)

Policy summary from `cascadeguard_eval_results.csv`:

- `grpo_local`
  - mean_score: `0.4121`
  - success_rate: `0.3333`
  - mean_parse_failure_rate: `0.0905`
  - mean_wait_rate: `0.1323`
  - mean_budget_spent: `8.7000`
- `heuristic`
  - mean_score: `0.7281`
  - success_rate: `0.6667`
  - mean_parse_failure_rate: `0.0000`
  - mean_wait_rate: `0.4069`
  - mean_budget_spent: `9.8333`

Notable task-level behavior for `grpo_local`:

- `task_surge_demand`: score `0.9900`, parse failures `0.0000`, wait `0.1176`.
- `task_medium`: score `0.8597`, parse failures `0.0667`, wait `0.0667`.
- Hard tasks still collapsed:
  - `task_gen_blackout`: score `0.0100`
  - `task_hard`: score `0.0100`
  - `task_cyberattack`: score `0.0100`

Action distribution warning:

- The trained policy overused `reroute` on most tasks in this smoke run, indicating remaining action-space collapse despite improved parseability.

## Recommended Run Commands

### Quick smoke (no eval)

```bash
python training/train_grpo.py --smoke-test --no-eval
```

### Iteration run with diagnostics

```bash
python training/train_grpo.py \
  --steps 800 \
  --states 384 \
  --generations 8 \
  --debug-parseability \
  --debug-hard-rollouts \
  --debug-hard-rollout-seeds 2
```

### Optional stronger warm-start

```bash
python training/train_grpo.py \
  --sft-warmstart-steps 220 \
  --sft-warmstart-lr 1e-5
```

## Suggested Acceptance Checks for v2

Use `cascadeguard_eval_task_summary.csv` and `cascadeguard_eval_results.csv`:

- Parse failures:
  - `mean_parse_failure_rate` should trend toward <= 0.15 overall.
  - Hard tasks should be materially below prior baseline.
- Waiting collapse:
  - `mean_wait_rate` should trend toward <= 0.45.
- Budget usage:
  - Hard tasks should show non-trivial `mean_budget_spent` (not near 0).
- Hard task score floor:
  - `task_hard`, `task_gen_blackout`, `task_cyberattack` should avoid near-zero collapse and improve over prior run.

## Notes

- If Hugging Face download instability reappears on Windows (`WinError 10054`), rely on local cache/offline mode once model weights are cached.
