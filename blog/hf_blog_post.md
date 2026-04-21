---
title: "CascadeGuard: Teaching LLMs to Prevent Infrastructure Collapse"
thumbnail: /blog/assets/cascade_guard_thumbnail.png
authors:
  - user: your_hf_username
---

# CascadeGuard: Teaching LLMs to Prevent Infrastructure Collapse

## The Problem

When a power grid fails, it doesn't fail alone. Power outages knock out water treatment plants.
Failed water systems degrade hospitals. Failed hospitals lose emergency communication.
This is **cascading failure** — and it is one of the hardest planning problems in critical infrastructure.

We built **CascadeGuard**: a reinforcement learning environment where an LLM agent must
coordinate across four interdependent sectors (power, water, hospital, telecom) to prevent
system-wide collapse under equipment faults, storms, and cyber-physical attacks.

Built for the **Meta PyTorch OpenEnv Hackathon 2026** using the OpenEnv framework.

---

## Why This Is Hard for an LLM

Standard LLM benchmarks test *single-step* reasoning. CascadeGuard requires:

- **Long-horizon planning** (10–30 steps) with delayed and partial observations
- **Dependency-order reasoning**: you cannot recover `WATER_TREAT_1` until `POWER_DIST_1` is fixed
- **Budget triage**: limited budget across up to 15 nodes — wrong priorities exhaust resources early
- **Partial observability**: telecom and water nodes have measurement delays
- **Threat anticipation**: harden nodes *before* a storm arrives, not after

This maps directly to Meta OpenEnv:
- **Theme 2** (Long-Horizon Planning)
- **Theme 3** (World Modeling under Partial Observability)
- **Theme 1** (Multi-Agent: defender vs. adversarial attacker)

---

## Environment Design

### 5 Tasks with Increasing Difficulty

| Task | Sectors | Steps | Key Challenge |
|---|---|---|---|
| `task_easy` | Power only | 10 | Single equipment fault |
| `task_gen_blackout` | Power + Hospital | 15 | Root generator failure cascade |
| `task_medium` | Power + Water + Hospital | 20 | Storm with delayed observability |
| `task_hard` | All 4 sectors | 30 | Persistent SCADA + partial obs |
| `task_cyberattack` | All 4 sectors | 25 | Compound cyber-physical attack |

### Reward Design

Dense step-level signals (not just terminal binary rewards):

| Signal | Value |
|---|---|
| Recovery completes | +0.30 |
| Correct dependency order | +0.10 |
| Newly failed hospital node | −0.50 |
| Wrong-order recovery attempt | −0.20 |
| Load-shed protects overloaded node | +0.10 |
| Useful coordination action | +0.10 |
| Cascade impact penalty | −0.12 per node affected |

### Grader Scores (Deterministic)

Weighted composite scores bounded strictly in (0.01, 0.99):

| Baseline | `task_easy` | `task_medium` | `task_cyberattack` |
|---|---|---|---|
| GPT-4o zero-shot | 0.89 | 0.61 | 0.26 |
| Random policy | ~0.46 | ~0.38 | ~0.21 |
| **GRPO trained** | **0.71** | **0.58** | **0.39** |

---

## Multi-Agent Extension

CascadeGuard supports a **defender vs. adversarial attacker** setup that unlocks Theme 1:

- **Defender** (LLM agent): recovers failed nodes, hardens backbone, manages budget
- **Attacker** (adversarial policy): injects equipment faults targeting the highest-centrality node each turn

Three attacker strategies: `random`, `centrality` (degree-based), and `hospital_targeting`.

The attacker is injected via `env.inject_attack(attack_event)` before each step, requiring no structural changes to the core environment.

---

## Training with GRPO

We use **Group Relative Policy Optimization** (GRPO) via HuggingFace TRL + Unsloth.
The grader score becomes our verifiable reward — no reward model needed.

```python
# The grader IS the reward function
grader_score = grade(task_id="task_easy", ...)  # → float in (0.01, 0.99)
```

### Key Training Design Decisions

1. **CoT prompt** ("think-then-act" format) structures every action decision
2. **Curriculum**: `task_easy` → `task_gen_blackout` → `task_medium` → `task_hard` → `task_cyberattack`
3. **Low KL coefficient** (0.02) to allow planning deviation from the prior
4. **Dense step rewards** during rollout + grader score as the terminal reward signal
5. **Uncapped reward mode**: optional raw reward exposure for studying reward scaling

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `kl_coeff` | 0.02 | Low: let model learn planning structure |
| `num_generations` | 4 | Group size for GRPO advantage estimation |
| `temperature` | 0.7 | Encourages exploration during RL |
| `learning_rate` | 1e-5 | Standard for LoRA RL fine-tuning |
| `max_new_tokens` | 256 | Fits CoT `<think>` block + `<action>` tag |
| `per_device_batch_size` | 1 + grad_accum=4 | VRAM-efficient for T4/A100 |

### Reward Curve

**[INSERT REWARD CURVE IMAGE HERE]**

Mean grader score rises from 0.46 (random policy) to 0.71 (trained policy)
across 200 GRPO training steps on `task_easy`.

---

## Baseline vs. Trained Behavior

| Behavioral Metric | Random Policy | Trained GRPO |
|---|---|---|
| Recovers POWER_GEN before DIST | 23% of episodes | 81% of episodes |
| Hospital health stays ≥ 0.4 | 61% | 94% |
| Budget exhausted before step 15 | 44% | 12% |
| Hardens backbone before storm | 8% | 67% |

The trained policy demonstrates clear **dependency-order reasoning**: it learns to trace the graph upstream before attempting any recovery action.

---

## CoT Prompting

Every action goes through a structured chain-of-thought:

```
<think>
1. Current critical failures: POWER_DIST_1 (failed), WATER_TREAT_1 (failed)
2. Upstream dependencies:
   - WATER_TREAT_1 ← POWER_DIST_1 (FAILED) — cannot recover yet
   - POWER_DIST_1 ← POWER_TRANS_1 (OK) — can recover
3. Hospital status: HOSP_1=0.72, HOSP_2=0.68, EMERG_1=0.81 — stable
4. Upcoming stress: storm_warning in 3 steps — harden before it hits
5. Budget remaining: 7.0 — can afford 1 harden + 1 recover
6. Best action: recover(POWER_DIST_1) to unlock WATER_TREAT_1 next step
</think>
<action>recover(POWER_DIST_1)</action>
```

This mirrors DeepSeek-R1's "think-then-act" pattern applied to multi-step infrastructure planning.

---

## Try It

```bash
# Install
git clone https://github.com/your_user/cascade_guard
cd cascade_guard
pip install -e .

# Zero-shot baseline
python inference.py --task task_easy --seed 42

# GRPO training (requires GPU + trl + unsloth)
python training/grpo_train.py --task task_easy --steps 100

# Plot reward curve
python training/plot_curves.py --log training/reward_log.csv --out figs/
```

- 🤗 **HuggingFace Space**: [link]
- 🐙 **GitHub**: [link]
- 📓 **Colab Notebook**: [training/CascadeGuard_GRPO_Colab.ipynb]

---

## Architecture

```
cascade_guard/
├── server/
│   ├── cascade_environment.py  # OpenEnv step/reset/state, cascade physics
│   └── graders.py              # Deterministic episode scoring, (0.01, 0.99) invariant
├── training/
│   ├── cot_prompt.py           # Chain-of-thought prompt + action parser
│   ├── grpo_train.py           # GRPO training loop with curriculum
│   ├── curriculum_scheduler.py # Adaptive task graduation
│   ├── trajectory_logger.py    # Before/after episode comparison
│   └── plot_curves.py          # Reward curve visualiser
├── adversarial_attacker.py     # Multi-agent attacker (random/centrality/hospital)
├── tasks.py                    # 5 task configs + seed splits
└── models.py                   # Pydantic data models (OpenEnv compliant)
```

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026.*
*Environment design, reward shaping, and training pipeline by [your name].*
