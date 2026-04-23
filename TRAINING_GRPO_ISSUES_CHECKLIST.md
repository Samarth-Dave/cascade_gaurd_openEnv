# Training & GRPO Issues Checklist

## Critical fixes
- [ ] Add graders for all `task_osm_*` tasks in `server/graders.py` and register them in `GRADERS`.
- [ ] Ensure OSM tasks do not fall back to neutral `0.5` scoring in `grade()`.
- [ ] Wire `training/grpo_train.py` parameters correctly (`normalise_rewards`, `num_generations`, `lora_r`, `lora_alpha`) so CLI/settings are actually used.
- [ ] Replace the current manual `compute_loss`/`train_dataset=None` flow with a robust GRPO training loop path.
- [ ] Remove broad training-step exception swallowing so failed optimizer steps are visible and actionable.
- [ ] Fix reward cache/indexing in `CascadeRewardFunction` (duplicate prompt handling and stable per-item indexing).

## Reward alignment
- [ ] Align `reward.py` budget normalization (`INIT_BUDGET`) with environment budgets or pass real init budget per task.
- [ ] Remove/replace stale `repair` action handling in `reward.py` to match env action names (`recover`, etc.).
- [ ] Confirm reward shaping components are consistent between `cascade_environment.py`, `reward.py`, and training reward usage.

## Evaluation reliability
- [ ] Update evaluation to use multi-seed defaults (not single-seed baseline only) for trustworthy score reporting.
- [ ] Avoid masking variance/statistics (e.g., fixed minimum std for single runs) in reports.
- [ ] Re-check score clamping/sanitization paths so they do not hide poor behavior.

## Verification coverage
- [ ] Extend `scripts/verify_training_pipeline.py` to fail on flat-score degeneracy (e.g., persistent `0.500`).
- [ ] Add checks that detect missing task graders for any task in `TASK_CONFIGS`.
- [ ] Add smoke checks validating reward signal diversity across actions for the same observation batch.

## Operational setup
- [ ] Install missing training/test dependencies in the active environment (`torch`, `trl`, `transformers`, `unsloth`, `accelerate`, `peft`, `bitsandbytes`, `pytest`).
- [ ] Run end-to-end GRPO smoke training and capture reward/loss curves once dependencies are installed.
