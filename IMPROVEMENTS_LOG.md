# CascadeGuard Improvements Log

A versioned record of every improvement sprint.

---

## v2.1 - 2026-04-21 | GRPO Training Alignment + Local Notebook

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
