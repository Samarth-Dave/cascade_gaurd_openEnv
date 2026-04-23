# CascadeGuard: Manual Tuning & Performance Guide

This guide is for the developer to manually tweak, tune, and test the internal variables of CascadeGuard to squeeze out maximum RL performance or environment realism.

## 1. Tuning the Reward Signal (`cascade_environment.py` & `graders.py`)

**Where to check:** 
- `cascade_environment.py: _compute_reward()`
- `graders.py: grade()` and individual grader components.

**What to tune:**
- **Wait Penalty:** Currently, progressive wait penalties punish the agent for sitting idle. If the agent acts *too chaotically* (spending too much budget too fast), decrease the wait penalty scale.
- **Cascade Debt Pressure:** If the agent ignores long-standing failures, increase the `-0.04/step` penalty.
- **Hospital Weighting:** In `graders.py`, adjust how heavily hospital health impacts the final score. If you want the agent to prioritize absolute grid survival over human safety, lower the hospital weights.

**How to test:** 
- Run `python inference.py` and inspect the terminal reward outputs. Ensure the rewards appropriately differentiate good strategies from bad ones.

## 2. Tuning the GRPO Training Hyperparameters (`grpo_train.py`)

**Where to check:** 
- The `TrainingArguments` block in `grpo_train.py` (if using TRL/Unsloth).

**What to tune:**
- **`learning_rate`:** If the loss curve is collapsing to 0 immediately or bouncing erratically, decrease the LR (e.g., from `2e-5` to `5e-6`).
- **`num_generations`:** Increasing from 4 to 8 will provide richer reward variance per batch, improving stability but doubling training time/VRAM usage.
- **LoRA Config (`r`, `lora_alpha`):** If the model is not learning the syntax of the CoT prompt, increase `r` (rank) from 16 to 32 to give the adapter more capacity to act smartly over JSON structures.
- **KL Penalty:** To stop the policy moving too far from the reference model (preventing gibberish output), increase the `beta` (KL coeff).

**How to test:**
- Look at the `figs/reward_curve.png` after running `plot_curves.py`. The curve should steadily increase. A flat line means the hyperparameters are broken.

## 3. Curriculum Learning Pacing (`curriculum_scheduler.py`)

**Where to check:**
- Promotion criteria in `curriculum_scheduler.py`.

**What to tune:**
- **Rolling Mean Window:** If the agent graduates to harder tasks out of luck, increase the required window (e.g., must clear the threshold for the last 5 episodes instead of 3).
- **Thresholds:** If the agent struggles continuously on `task_hard`, slightly lower the graduation threshold from `task_medium` so it isn't over-trained on medium before seeing hard mechanics.

## 4. Adversarial Attacker Aggression (`adversarial_attacker.py`)

**Where to check:**
- `AdversarialAttacker` initialization logic.

**What to tune:**
- **`aggression` scale:** Adjust the `0.0` to `1.0` value parameter. Too high, and the defender has *no mathematical sequence of moves* to survive, plateauing the learning curve. Too low, and the environment defaults back to being deterministic/easy.

**How to test:**
- Run the baseline evaluator in the multi-agent setup, logging `attacker_efficiency` against `defender_score` from `grade_adversarial()`. Ensure the win rate is balanced (e.g., ~50-60% defender survival).
