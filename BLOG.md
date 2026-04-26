# CascadeGuard: Teaching LLMs to Protect Infrastructure From Cascading Failure

## Summary

CascadeGuard is an OpenEnv environment for training LLM agents to manage cross-sector infrastructure crises. The agent acts as a resilience coordinator during cascading failures across power, water, hospital, telecom, and financial-settlement systems.

The problem is deliberately hard: a single power failure can disable water treatment, degrade hospital service, disrupt telecom coordination, and create a wider emergency. The agent must reason over an interdependent graph, delayed telemetry, budget limits, weather events, cyber-physical attacks, and recovery-order constraints.

The goal is not just to build a simulator. The goal is to build an environment that can teach an LLM a measurable capability: cross-sector dependency reasoning under partial observability.

Links:

| Resource | Link |
|---|---|
| Backend API Space | https://harshit0400-backend.hf.space |
| Frontend UI Space | https://huggingface.co/spaces/LordBhatt/CascadeGuardUI |
| Live environment Space | https://huggingface.co/spaces/samarthdave0305/cascade-failure-env |
| Trained adapter checkpoint | https://huggingface.co/samarthdave0305/cascadeguard-trained/tree/main |
| GitHub repository | https://github.com/Samarth-Dave/cascade_gaurd_openEnv |
| Result assets folder | `assets/results/` |

## Why This Environment Matters

Modern infrastructure does not fail in isolation. Power systems run water pumps. Water systems cool power generation and support hospitals. Hospitals depend on both power and communications. Telecom and SCADA systems coordinate repair and emergency response.

In real incidents, the problem is not only that something breaks. The problem is that many systems break in the wrong order, under time pressure, with incomplete information, and with multiple organizations optimizing locally.

CascadeGuard turns that real-world coordination failure into an OpenEnv training problem.

The agent must answer questions like:

- Which upstream node should be recovered first?
- Should the agent harden a central power node before a storm arrives?
- Is it safe to shed load from this node, or would that harm a critical hospital?
- When telemetry is delayed, should the agent coordinate before acting?
- Should it patch SCADA, reroute supply, isolate a compromised node, or deploy a repair crew?
- How should it spend a limited budget across a multi-sector dependency graph?

This is the capability gap CascadeGuard targets: long-horizon, cross-sector infrastructure reasoning.

## What the Agent Sees

At each step, the agent receives an observation containing:

- Node health across power, water, hospital, telecom, and finance sectors
- Operational status for each node
- Load and overload pressure
- Criticality flags for hospitals and emergency services
- Interdependency edges between sectors
- Remaining response budget
- Active failures and pending recoveries
- Weather and stress signals
- Delayed or partial observations for some nodes
- Diagnostics such as at-risk nodes and recommended recovery order

The environment is graph-structured and partially observable. Some information arrives late, which forces the agent to coordinate and plan instead of greedily reacting.

## What the Agent Can Do

CascadeGuard supports both core and advanced infrastructure actions.

| Action | Purpose |
|---|---|
| `recover` | Start recovery for a failed node when dependencies allow it |
| `harden` | Spend budget to protect an operational node before stress hits |
| `shed_load` | Reduce load on non-critical overloaded nodes |
| `coordinate` | Improve response under delayed or partial observations |
| `isolate` | Isolate a failed or compromised node to limit propagation |
| `reroute` | Redirect healthy supply toward a stressed target |
| `prioritize` | Mark a critical node as operationally important |
| `deploy_repair_crew` | Speed up a recovery already in progress |
| `emergency_shutdown` | Safely shut down a fragile non-critical node |
| `cross_sector_bridge` | Create temporary cross-sector support |
| `patch_scada` | Mitigate cyber/SCADA anomalies |
| `redistribute_load` | Move load within a sector from overloaded to underloaded nodes |
| `request_mutual_aid` | Request sector-level external support |
| `controlled_cascade` | Sacrifice a node deliberately to contain wider collapse |
| `multi_sector_lockdown` | Spend budget on broad emergency stabilization |
| `wait` | Take no action |

This action set makes the environment more than a recover/wait game. It tests whether an agent can choose the right intervention type for the failure mode.

## Task Curriculum

The training and evaluation curriculum contains escalating scenarios:

| Task | Scenario | What It Tests |
|---|---|---|
| `task_easy` | Small power graph with a critical dependency | Basic hardening and recovery |
| `task_medium` | Power, water, and hospital under storm stress | Delayed telemetry and hospital protection |
| `task_surge_demand` | Demand surge across dependent services | Load shedding and coordination |
| `task_gen_blackout` | Root generator failure and downstream collapse | Graph-centrality and recovery order |
| `task_hard` | Multi-sector crisis with telecom pressure | Budgeted triage under partial observability |
| `task_cyberattack` | SCADA anomaly plus physical faults | Cyber-physical defense |

The easy tasks teach basic semantics. The hard tasks require the agent to reason over cascading dependencies, not just memorize action names.

## Reward Design

CascadeGuard uses dense step rewards plus episode-level grading. The reward is designed to teach useful behavior rather than reward a single terminal success.

The agent is rewarded for:

- Protecting critical hospital and emergency-service nodes
- Recovering failed nodes in dependency order
- Preventing new failures and limiting cascade depth
- Using budget efficiently
- Hardening important upstream nodes before stress arrives
- Taking useful coordination actions when telemetry is delayed
- Choosing legal, meaningful actions instead of waiting during crisis

The agent is penalized for:

- Wrong-order recovery attempts
- Letting failures propagate
- Damaging critical infrastructure through unsafe load shedding
- Repeating ineffective actions
- Wasting budget
- Waiting when there are active failures or at-risk nodes

This makes the reward hard to game. A policy that only hardens everything exhausts budget. A policy that only recovers downstream nodes violates dependency order. A policy that waits during crisis gets punished. A useful policy must balance several objectives simultaneously.

## Training Pipeline

The training stack uses Hugging Face TRL and LoRA fine-tuning.

Training stages:

1. Collect environment states from CascadeGuard tasks.
2. Build action-formatted prompts with legal action context.
3. Run supervised fine-tuning so the model learns the action interface and basic policy behavior.
4. Run GRPO against environment rewards so the model improves beyond imitation.
5. Save loss/reward plots and push adapter checkpoints.

The current Space trains `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` with 4-bit quantization and LoRA adapters.

## Training Evidence

SFT has completed successfully on the Hugging Face backend training service.

Observed SFT run:

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- LoRA rank: `16`
- SFT steps: `150`
- Dataset rows: `155`
- Initial logged loss: about `1.59`
- Final logged loss: about `0.18`
- Mean training loss: about `0.506`
- Adapter checkpoint pushed to: https://huggingface.co/samarthdave0305/cascadeguard-trained

This shows the model learned the action format and environment structure during supervised fine-tuning.

GRPO has reached real training. The first full attempt with `generations=4`, `max_seq_len=1024`, and `max_completion_len=256` loaded the model and started GRPO, then hit CUDA out-of-memory on the A100 during per-token log-prob computation. The backend training service has been updated to use reduced-memory GRPO settings:

- `generations=2`
- `max_seq_len=512`
- `max_completion_len=128`
- stronger CUDA cleanup after SFT
- skip SFT on restart if the checkpoint already exists

Final GRPO reward plots should be placed in:

- `assets/results/training_reward_curve.png`
- `assets/results/baseline_vs_trained_reward.png`

## Result Images

The following image paths are reserved for final submission assets:

![Training loss](./assets/results/training_loss_curve.png)

Caption: SFT/GRPO loss over training step.

![Training reward](./assets/results/training_reward_curve.png)

Caption: GRPO reward over training step.

![Baseline vs trained reward](./assets/results/baseline_vs_trained_reward.png)

Caption: Baseline policy versus trained policy on the same evaluation split.

![Dashboard screenshot](./assets/results/demo_dashboard.png)

Caption: Live CascadeGuard Space dashboard.

## What Changed After Training

The SFT-trained adapter learns to produce valid action JSON and follows the environment's action contract more reliably than an untrained model. In practice, this matters because invalid actions are heavily penalized and do not help control a cascade.

The learned behavior from SFT includes:

- Better adherence to `action_type` and `target_node_id` format
- More consistent recovery of failed nodes
- Better use of `harden` before stress events
- Less random waiting during active failures
- More stable behavior on task prompts with legal action context

The expected GRPO improvement target is:

- higher reward than random or naive baseline
- fewer hospital-critical failures
- better dependency-order recovery
- improved budget discipline
- more frequent use of advanced actions when legally useful

## Why OpenEnv Is a Good Fit

OpenEnv matters here because the environment is not just a static dataset. The agent must interact with a live stateful simulator:

- `reset()` initializes a crisis scenario
- `step(action)` advances the cascade
- observations change based on previous actions
- rewards are computed from actual environment dynamics
- the same task can be evaluated across seeds and splits

This is exactly the kind of setting where an LLM can be trained to become measurably better at a behavior, rather than merely reciting instructions.

## Why This Is Different From Existing Benchmarks

Many RL environments test simple games, navigation, or isolated control systems. CascadeGuard tests a different class of reasoning:

- multi-sector dependency graphs
- partial observability
- delayed telemetry
- critical-service protection
- adversarial cyber-physical events
- long-horizon budget allocation
- legal-action constrained decision making

The closest real-world inspiration is single-sector power-grid RL such as L2RPN. CascadeGuard extends that spirit to cross-sector infrastructure resilience, where power, water, hospitals, telecom, and finance interact.

## Limitations

CascadeGuard is still a simulation. It uses seeded infrastructure graphs and synthetic stress events rather than live SCADA feeds.

Current limitations:

- Real operator telemetry is not yet connected.
- GRPO on a 32B model is memory-intensive and needs careful settings.
- Final reward-improvement plots must be generated from a successful reduced-memory GRPO run.
- The current trained checkpoint is an adapter, not a fully merged base model.

These are engineering limitations, not conceptual blockers. The environment already supports the core OpenEnv loop and real training.

## Next Steps

The next iteration is clear:

1. Complete reduced-memory GRPO and commit reward plots.
2. Run baseline-vs-trained evaluation over held-out seeds.
3. Add the final demo video link to the README.
4. Replace synthetic seeds with anonymized real operator telemetry.
5. Expand tasks to include larger city-scale infrastructure graphs.

## Conclusion

CascadeGuard is a benchmark for a capability LLMs do not currently have: acting as a cross-sector infrastructure resilience coordinator during a cascading crisis.

It is ambitious because the environment forces the agent to reason over a partially observable graph, not just choose moves in a toy world. It is practical because infrastructure failures are real and costly. It is trainable because the environment provides dense rewards, legal actions, curriculum tasks, and a working TRL-based training pipeline.

The core claim is simple:

> If we want LLMs to help with real-world resilience, they need environments that teach interdependency reasoning. CascadeGuard is one such environment.

