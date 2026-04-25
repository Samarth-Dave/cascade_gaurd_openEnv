# Model Selection — pre-training decision

> **Goal**: Pick the right base model BEFORE spending T4 hours on SFT + GRPO.
> A bad model choice cannot be fixed by training; a good model choice often
> halves training time and doubles final reward.

## Why this exists

We were burning Colab T4 cycles on `Qwen/Qwen2.5-0.5B-Instruct` GRPO without
asking the more important question: **does this base model have the inductive
bias to learn the cascade-failure task at all?** GRPO amplifies what the base
model already knows; if the base can't follow JSON format or reason about a
graph topology zero-shot, fine-tuning won't save it.

This folder contains:

| File | Purpose |
|---|---|
| `ENV_AUDIT.md` | Honest assessment of the environment (the 40%-of-grade-rubric criterion). Should the env even be optimized against? Yes. |
| `compare_models.py` | Standalone Python harness that loads each candidate in 4-bit, runs zero-shot rollouts, and ranks them by `quality / cost`. |
| `results/leaderboard.json` | Output of the harness (created on first run). |
| this file | Methodology + how to interpret results + decision rules. |

## Methodology

**Decision rule**: rank candidates by

```text
fitness = (mean_episode_reward + 30) * valid_action_rate / (mean_tokens_per_step * mean_seconds_per_step)
```

Why this formula:

- `mean_episode_reward + 30` — the `+30` shift keeps the metric monotonic even when models score negative. We care about ordering across candidates, not absolute fitness magnitude.
- `valid_action_rate` — the fraction of generated actions that pass legal-action validation. A model that outputs unparseable JSON or hallucinates node IDs is useless regardless of "intent".
- `mean_tokens_per_step` — generation cost per decision. Tiny models (0.5B) generating 200-token rationales lose to 3B models generating 30-token JSON-only.
- `mean_seconds_per_step` — wall-clock cost. Reflects both model size and tokens generated.

**A `pareto_table.md`** is also printed (full leaderboard sorted by fitness, with quality and cost columns visible) so you can override the recommendation if you weight quality differently.

### What we measure (per model, per task)

| Metric | What it tells us |
|---|---|
| `mean_reward` | Average sum-of-step-rewards over the episode. Primary quality signal. |
| `median_reward` | Robust quality (1 catastrophic episode doesn't sink the score) |
| `valid_rate` | `chosen ∈ legal_actions` rate. Format + reasoning combined. |
| `parse_rate` | Just JSON parseability. If `parse_rate > valid_rate` substantially, the model formats fine but reasons poorly. |
| `mean_tokens` | Tokens generated per step. Lower = cheaper per decision. |
| `mean_seconds` | Wall-clock per step on T4. |
| `vram_gb` | Peak VRAM during loading + first inference. |

### What we DON'T measure (and why)

- **SFT improvement**: Adds 30+ minutes per model. We assume that GRPO + SFT will roughly preserve relative ordering of models — true in practice, especially for similar-architecture families.
- **Token efficiency under SFT**: After SFT the model is forced to produce concise JSON, so token counts drop. Zero-shot tokens are a worst-case proxy.
- **Train-from-scratch tokens**: Not relevant — all candidates are pretrained instruction-tuned models.

## How to run

### From Colab (recommended)

There's a dedicated Cell M near the bottom of `training/CascadeGuard_TRL_Colab_fixed_(1) (1).ipynb` that runs the harness.

You can also call it directly:

```bash
!cd /content/cascade_guard && python temp/model_selection/compare_models.py
```

Defaults:
- 7 candidate models
- 6 tasks × 2 episodes each = 12 rollouts per model
- 8–12 step cap per task → ~100 steps per model
- ~5 minutes per model on T4 → 30–45 minutes total

### Quicker smoke test (3 candidates, 1 episode)

```bash
!python temp/model_selection/compare_models.py \
    --candidates Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --episodes 1 \
    --max-new-tokens 48
```

Should finish in ~10 minutes.

### Local (CPU only)

Edit `_load_model` to skip `BitsAndBytesConfig` and use `torch_dtype=torch.float32`. Realistically, this is too slow for the 3B+ candidates — Colab is the right venue.

## Candidate justification

| Candidate | Why include | Expected character |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | Current baseline | Fast, may struggle with structured reasoning |
| `Qwen/Qwen2.5-1.5B-Instruct` | One tier up, same family | Sweet spot for many small-RL tasks |
| `Qwen/Qwen2.5-3B-Instruct` | Upper bound on Qwen2.5 family that fits T4 GRPO | Best raw reasoning, slowest |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Different architecture lineage | Sometimes better at concise structured output |
| `meta-llama/Llama-3.2-1B-Instruct` | Different family entirely | Strong instruction-following at small scale; **gated — needs HF_TOKEN** |
| `microsoft/Phi-3.5-mini-instruct` | Reasoning-heavy training data | Strong on multi-step logic; biggest of the set (~3.8B) |
| `Qwen/Qwen3-1.7B` | Newest Qwen generation | Improved reasoning over Qwen2.5 at same param count |

## Decision rules — how to read the leaderboard

After the run completes, look at the printed table:

```text
MODEL                                ep_reward   med_R  valid%  parse%  tok/step  sec/step   vram   fitness
========================================================================================================================
Qwen/Qwen2.5-1.5B-Instruct              -2.34   -1.80   78.5%   91.2%        45      0.28   2.1G    0.6234
Qwen/Qwen2.5-3B-Instruct                -1.10   -0.50   85.7%   95.1%        62      0.51   3.4G    0.4128
HuggingFaceTB/SmolLM2-1.7B-Instruct     -3.45   -3.10   65.3%   88.0%        51      0.35   2.3G    0.3811
...
```

Rules of thumb:

1. **Pick the fitness winner UNLESS**:
   - Its `valid_rate < 50%` — it can't follow the protocol; SFT might fix this but it's a risk.
   - Its `mean_reward` is more than `2.0` worse than the second-place by `mean_reward` — fitness penalizes cost too much in this case.
2. **If two candidates are within ±5% on fitness**, pick the one with the lower `vram_gb` — leaves more room for `num_generations` during GRPO.
3. **If `valid_rate ≪ parse_rate`** for the winner — the model formats JSON but picks illegal actions. SFT on the heuristic teacher will likely fix this. Continue with the candidate.
4. **If ALL candidates have `valid_rate < 30%`** — the prompt itself is the problem, not the models. Inspect `_build_user_prompt` in `compare_models.py`.

## What to do after picking

1. Update `MODEL` in the Colab notebook (cell that defines `MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'`).
2. If the winner is **larger** than current (e.g., 1.5B instead of 0.5B), reduce `GENERATIONS` in GRPOConfig from 6 → 4 to keep VRAM headroom.
3. If the winner is **smaller** (unlikely given current is 0.5B), increase `GENERATIONS` to 8 for more reward signal.
4. Re-run Cell A (signal sanity) to confirm the env is still well-shaped against the new model.
5. Run Cell B (diversity check) AFTER SFT — pre-SFT diversity is meaningless for an untrained model.

## Files saved

- `results/leaderboard.json` — raw per-episode results + per-model aggregates. Useful for plotting / regression.
- `results/winner.txt` — single line with the winning model ID (the harness does NOT write this; you do, after reviewing the leaderboard).
