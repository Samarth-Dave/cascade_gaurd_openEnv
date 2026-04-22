"""
train_grpo.py — CascadeGuard GRPO Training Script
==================================================
Single-file replacement for CascadeGuard_GRPO_Colab.ipynb.
Works locally with Unsloth (installed via the official installer).

INSTALLATION (Windows — Anaconda Prompt or PowerShell):
  irm https://unsloth.ai/install.ps1 | iex           # installs Unsloth + torch auto
  # Then install the project deps:
  conda activate unsloth_env   (or whatever env Unsloth made)
  pip install "trl>=0.24.0" "datasets>=4.2" "peft>=0.14" "accelerate>=1.10" pandas matplotlib "fastapi>=0.115" "uvicorn>=0.30" "pydantic>=2.0" "openenv-core[core]>=0.2.2" mergekit
  cd "C:\\Users\\Samarth Dave\\Desktop\\metaV1\\cascade_guard"
  pip install -e .

INSTALLATION (Linux / WSL / macOS):
  curl -fsSL https://unsloth.ai/install.sh | sh
  pip install "trl>=0.24.0" "datasets>=4.2" "peft>=0.14" "accelerate>=1.10" pandas matplotlib "fastapi>=0.115" "uvicorn>=0.30" "pydantic>=2.0" mergekit
  cd /path/to/cascade_gaurd_openEnv
  pip install -e .

USAGE:
  # Full training run (recommended)
  python train_grpo.py

  # Quick smoke test (10 steps, 32 states)
  python train_grpo.py --smoke-test

  # Custom config
  python train_grpo.py --steps 500 --states 512 --lora-r 16 --model Qwen/Qwen2.5-3B-Instruct

  # Skip eval after training
  python train_grpo.py --no-eval

  # Resume from checkpoint
  python train_grpo.py --resume-from cascadeguard_grpo_local/checkpoint-100

  # Custom HF cache location (model is downloaded once, reused forever)
  python train_grpo.py --smoke-test --cache-dir D:\\hf_models
"""

# ── Stub factory for optional trl deps ───────────────────────────────────────
#
# trl 0.24.0 imports mergekit and llm_blender at module-load time.
# llm_blender 0.0.2 is broken with transformers v5 (removed TRANSFORMERS_CACHE).
# Simple stubs used to fail on Python 3.12 with "llm_blender.__spec__ is None"
# because Python's import machinery requires __spec__, __path__, and __package__
# to be set correctly on any module used as a package.
#
# This factory sets all required dunder attributes so the import system never
# raises.  Safe to keep even after installing the real packages.
# ─────────────────────────────────────────────────────────────────────────────
import importlib.machinery as _imm, types as _t, sys as _s
import os

from sympy import false
import torch

# 🔴 Disable compilation (CRITICAL for Windows)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
def _make_stub(dotted_name: str, attrs: dict = None):
    """Return a fully-formed stub module safe for Python 3.12 import machinery."""
    if dotted_name in _s.modules:
        return _s.modules[dotted_name]
    mod = _t.ModuleType(dotted_name)
    # Required by Python 3.12: __spec__ must not be None for subpackage imports
    spec = _imm.ModuleSpec(dotted_name, loader=None, origin="<stub>")
    spec.submodule_search_locations = []   # marks it as a package
    mod.__spec__    = spec
    mod.__package__ = dotted_name
    mod.__path__    = []                   # also required for packages
    mod.__loader__  = None
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    _s.modules[dotted_name] = mod
    return mod

# mergekit ────────────────────────────────────────────────────────────────────
try:
    import mergekit  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _cfg = _make_stub("mergekit.config", {"MergeConfiguration": object})
    _mk  = _make_stub("mergekit",        {"config": _cfg,
                                          "MergeConfiguration": object})

# llm_blender ─────────────────────────────────────────────────────────────────
# llm_blender imports TRANSFORMERS_CACHE which was removed in transformers v5.
# We stub every sub-path trl/judges.py touches so `import llm_blender` and
# `from llm_blender.blender.blender import Blender` both succeed silently.
try:
    import llm_blender  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _bu  = _make_stub("llm_blender.blender.blender_utils")
    _bl  = _make_stub("llm_blender.blender.blender",   {"Blender": object})
    _blp = _make_stub("llm_blender.blender",
                      {"Blender": object, "blender": _bl, "blender_utils": _bu})
    _make_stub("llm_blender", {"Blender":  object,
                               "blender":  _blp})
try:
    import weave  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _make_stub("weave")

del _imm, _t, _s, _make_stub

# ── Standard library ─────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Third-party (installed by Unsloth installer + pip above) ─────────────
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cascadeguard")


def _ensure_trainer_compatibility(model):
    """
    TRL's GRPOTrainer expects `model.warnings_issued` to exist.
    Some PEFT / Unsloth wrapper stacks do not expose it, so we seed it on
    the wrapper and common nested base-model objects before trainer init.
    """
    for obj in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if obj is None:
            continue
        if not hasattr(obj, "warnings_issued"):
            try:
                setattr(obj, "warnings_issued", {})
            except Exception:
                pass
        if hasattr(obj, "config") and getattr(obj.config, "use_cache", None) is not False:
            try:
                obj.config.use_cache = False
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════
# 1.  CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CascadeGuard GRPO local training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    # Default switched to 0.5B — downloads ~400 MB instead of 1.53 GB.
    # Use --model unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit to go back.
    p.add_argument("--model",
                   default="unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
                   help="HuggingFace model ID to fine-tune")
    p.add_argument("--no-unsloth",  action="store_true",
                   help="Disable Unsloth; use standard Transformers+PEFT instead")
    p.add_argument("--hf-token",    default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace token (only needed for gated models)")

    # Cache — model weights are downloaded once and reused on every subsequent run
    p.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.expanduser("~"), ".cache", "cascadeguard_hf"),
        help=(
            "Local directory for HuggingFace model cache. "
            "Model is downloaded once; subsequent runs load from disk."
        ),
    )

    # LoRA
    p.add_argument("--lora-r",      type=int, default=16,
                   help="LoRA rank  (16=fast/low-mem, 32=more capacity)")
    p.add_argument("--lora-alpha",  type=int, default=None,
                   help="LoRA alpha (default: 2 × lora-r)")

    # Training
    p.add_argument("--steps",       type=int, default=300,
                   help="Total GRPO training steps")
    p.add_argument("--states",      type=int, default=256,
                   help="Number of training state samples to collect")
    p.add_argument("--generations", type=int, default=8,
                   help="GRPO group size (completions per prompt)")
    p.add_argument("--lr",          type=float, default=5e-6,
                   help="Learning rate")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-completion-len", type=int, default=128)

    # Tasks
    p.add_argument("--tasks", nargs="+",
                   default=["task_easy","task_medium","task_gen_blackout",
                            "task_hard","task_cyberattack"],
                   help="Task IDs to train on")
    p.add_argument("--eval-tasks", nargs="+", default=None,
                   help="Task IDs to evaluate on (default: same as --tasks)")
    p.add_argument("--eval-seeds",  type=int, default=2,
                   help="Validation seeds per task during evaluation")

    # Output
    p.add_argument("--output-dir",  default="cascadeguard_grpo_local")
    p.add_argument("--results-csv", default="cascadeguard_eval_results.csv")
    p.add_argument("--resume-from", default=None,
                   help="Path to a checkpoint directory to resume training")

    # Shortcuts
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick sanity run: 10 steps, 32 states")
    p.add_argument("--no-eval",     action="store_true",
                   help="Skip post-training evaluation")
    p.add_argument("--repo-root",   default=os.environ.get("CASCADEGUARD_REPO", ""),
                   help="Path to the cascade_gaurd_openEnv repo root")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# 2.  PROJECT ROOT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════

def find_project_root(hint: str = "") -> Path:
    """
    Walk up from CWD (or use hint) to find the repo root that contains
    cascade_guard/server/cascade_environment.py.
    """
    if hint:
        candidate = Path(hint).expanduser().resolve()
        if (candidate / "cascade_guard" / "server" / "cascade_environment.py").exists():
            return candidate
        if (candidate / "server" / "cascade_environment.py").exists():
            return candidate.parent
        raise RuntimeError(
            f"CASCADEGUARD_REPO '{hint}' does not look like the repo root.\n"
            "Expected to find:  cascade_guard/server/cascade_environment.py"
        )
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "cascade_guard" / "server" / "cascade_environment.py").exists():
            return candidate
    raise RuntimeError(
        "Cannot find the cascade_gaurd_openEnv repo.\n"
        "Fix options:\n"
        "  1. cd into the repo root before running this script\n"
        "  2. set CASCADEGUARD_REPO=C:\\path\\to\\cascade_gaurd_openEnv\n"
        "  3. pass --repo-root C:\\path\\to\\cascade_gaurd_openEnv"
    )


# ══════════════════════════════════════════════════════════════════════════
# 3.  ENVIRONMENT IMPORTS  (done after sys.path is set)
# ══════════════════════════════════════════════════════════════════════════

def import_cascade(project_root: Path):
    """Insert the repo on sys.path then import all cascade_guard symbols."""
    repo_str = str(project_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from cascade_guard.models import CascadeAction
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    from cascade_guard.tasks import TASK_CONFIGS, TASK_SEED_SPLITS
    from cascade_guard.training.cot_prompt import (
        build_system_prompt,
        build_user_prompt,
        parse_action_from_response,
    )
    return CascadeAction, CascadeEnvironment, TASK_CONFIGS, TASK_SEED_SPLITS, \
           build_system_prompt, build_user_prompt, parse_action_from_response


# ══════════════════════════════════════════════════════════════════════════
# 4.  ENVIRONMENT SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════

def run_env_sanity_check(CascadeEnvironment, CascadeAction, TASK_SEED_SPLITS, tasks):
    log.info("Running environment sanity check …")
    for task_id in tasks:
        env = CascadeEnvironment()
        obs = env.reset(
            task_id=task_id,
            seed=TASK_SEED_SPLITS[task_id]["train"][0],
            reward_mode="grpo",
            training_mode=True,
        )
        legal = env.get_legal_actions()
        obs = env.step(CascadeAction(action_type="wait", target_node_id=None))
        meta = obs.metadata or {}
        log.info(
            "  %-25s  nodes=%-3d  legal=%-3d  reward=%.4f  mode=%s",
            task_id, len(obs.nodes), len(legal),
            float(obs.reward), meta.get("reward_mode", "?"),
        )
    log.info("Sanity check passed.")


# ══════════════════════════════════════════════════════════════════════════
# 5.  HEURISTIC POLICY  (baseline + state collector)
# ══════════════════════════════════════════════════════════════════════════

def make_heuristic_policy(CascadeAction):
    def heuristic_policy(obs):
        diag = obs.diagnostics
        # 1. Follow diagnostics recovery order
        if diag and diag.recommended_recovery_order:
            return CascadeAction(
                action_type="recover",
                target_node_id=diag.recommended_recovery_order[0],
            )
        # 2. Recover failed nodes (critical first)
        failed = [
            n for n in obs.nodes
            if not n.is_operational and n.node_id not in set(obs.pending_recoveries)
        ]
        if failed:
            failed.sort(key=lambda n: (not n.is_critical, n.health))
            return CascadeAction(action_type="recover", target_node_id=failed[0].node_id)
        # 3. Coordinate delayed observations
        delayed = [n for n in obs.nodes if n.observation_delayed]
        if delayed and obs.budget_remaining >= 1.0:
            delayed.sort(key=lambda n: (not n.is_critical, n.health))
            return CascadeAction(action_type="coordinate", target_node_id=delayed[0].node_id)
        # 4. Harden at-risk nodes before weather hits
        at_risk = []
        if diag and diag.at_risk_nodes:
            node_map = {n.node_id: n for n in obs.nodes}
            at_risk = [node_map[nid] for nid in diag.at_risk_nodes if nid in node_map]
        if obs.budget_remaining >= 2.0 and (
            obs.weather_forecast != "clear" or obs.step < 5
        ):
            candidates = [
                n for n in (at_risk or obs.nodes)
                if n.is_operational and not n.is_hardened
            ]
            if candidates:
                candidates.sort(key=lambda n: (not n.is_critical, n.health))
                return CascadeAction(action_type="harden", target_node_id=candidates[0].node_id)
        # 5. Shed load on overloaded non-critical nodes
        overloaded = [
            n for n in obs.nodes
            if n.is_operational
            and not n.is_critical
            and n.sector != "hospital"
            and n.load > 0.85
        ]
        if overloaded:
            overloaded.sort(key=lambda n: n.load, reverse=True)
            return CascadeAction(action_type="shed_load", target_node_id=overloaded[0].node_id)
        # 6. Default: wait
        return CascadeAction(action_type="wait", target_node_id=None)

    return heuristic_policy


# ══════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy_name: str,
    policy_fn,
    eval_tasks: List[str],
    seeds_per_task: int,
    CascadeEnvironment,
    TASK_CONFIGS: dict,
    TASK_SEED_SPLITS: dict,
) -> pd.DataFrame:
    rows = []
    for task_id in eval_tasks:
        seeds = TASK_SEED_SPLITS[task_id]["validation"][:seeds_per_task]
        for seed in seeds:
            env = CascadeEnvironment()
            obs = env.reset(task_id=task_id, seed=seed, scenario_split="validation")
            invalid_actions = 0
            critical_failures = 0
            rewards = []
            latencies = []
            actions_log = []
            done = False

            while not done:
                t0 = time.perf_counter()
                out = policy_fn(obs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if isinstance(out, tuple):
                    action, elapsed_ms = out
                else:
                    action = out

                latencies.append(float(elapsed_ms))
                actions_log.append({
                    "action_type": action.action_type,
                    "target_node_id": action.target_node_id,
                })
                obs = env.step(action)
                rewards.append(float(obs.reward))
                meta = obs.metadata or {}
                if not meta.get("action_valid", True):
                    invalid_actions += 1
                if obs.diagnostics:
                    critical_failures += len(obs.diagnostics.critical_failures)
                done = bool(obs.done)

            state = env.state
            avg_sector = sum(state.sector_health.values()) / max(len(state.sector_health), 1)
            rows.append({
                "policy":             policy_name,
                "task_id":            task_id,
                "seed":               seed,
                "score":              float(state.score),
                "success":            bool(avg_sector >= 0.5 and invalid_actions == 0),
                "steps":              int(state.step_count),
                "mean_reward":        statistics.mean(rewards) if rewards else 0.0,
                "invalid_actions":    invalid_actions,
                "invalid_action_rate": invalid_actions / max(len(rewards), 1),
                "budget_spent":       float(TASK_CONFIGS[task_id]["budget"] - state.budget_remaining),
                "critical_failures":  critical_failures,
                "avg_response_ms":    statistics.mean(latencies) if latencies else 0.0,
                "actions":            json.dumps(actions_log),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 7.  TRAINING STATE COLLECTION
# ══════════════════════════════════════════════════════════════════════════

def collect_training_states(
    num_states: int,
    train_tasks: List[str],
    TASK_SEED_SPLITS: dict,
    CascadeEnvironment,
    heuristic_policy,
    build_system_prompt,
    build_user_prompt,
) -> List[dict]:
    SYSTEM_PROMPT = (
        build_system_prompt(grpo_mode=True)
        + "\nReturn exactly one <action>...</action> tag."
    )

    def make_training_prompt(obs, env=None):
        base = SYSTEM_PROMPT + "\n\n" + build_user_prompt(obs)
        if env is not None and hasattr(env, "get_legal_actions"):
            legal_parts = []
            for act in env.get_legal_actions()[:12]:
                atype  = act.get("action_type", "wait")
                target = act.get("target_node_id")
                legal_parts.append(f"{atype}({target if target else 'null'})")
            base += "\n\nLegal actions now: " + " | ".join(legal_parts)
        return base

    rows: List[dict] = []
    log.info("Collecting %d training states from tasks: %s", num_states, train_tasks)

    for task_id in train_tasks:
        for seed in TASK_SEED_SPLITS[task_id]["train"]:
            env = CascadeEnvironment()
            obs = env.reset(
                task_id=task_id,
                seed=seed,
                scenario_split="train",
                reward_mode="grpo",
                training_mode=True,
            )
            history: List[dict] = []
            done = False

            while not done and len(rows) < num_states:
                rows.append({
                    "prompt":       make_training_prompt(obs, env),
                    "task_id":      task_id,
                    "seed":         int(seed),
                    "history_json": json.dumps(history),
                })
                action = heuristic_policy(obs)
                history.append({
                    "action_type":   action.action_type,
                    "target_node_id": action.target_node_id,
                })
                obs = env.step(action)
                done = bool(obs.done)

            if len(rows) >= num_states:
                log.info("  Collected %d states.", len(rows))
                return rows

    log.info("  Collected %d states.", len(rows))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 8.  GRPO REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def make_grpo_reward_fn(CascadeEnvironment, CascadeAction, parse_action_from_response):
    """
    Returns a reward function compatible with TRL GRPOTrainer.
    Closed over the cascade_guard symbols so it can be passed directly.
    """
    def _completion_to_text(completion) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            last = completion[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
        return str(completion)

    def _ensure_list(value, n: int, default):
        if value is None:
            return [default] * n
        if isinstance(value, (str, int, float)):
            return [value] * n
        return list(value)

    def cascade_grpo_reward(
        prompts, completions,
        task_id=None, seed=None, history_json=None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        n = len(completions)
        task_ids  = _ensure_list(task_id,      n, "task_easy")
        seeds     = _ensure_list(seed,         n, 42)
        histories = _ensure_list(history_json, n, "[]")

        for completion, tid, sd, hist_text in zip(completions, task_ids, seeds, histories):
            # ── Reconstruct env state ──────────────────────────────────
            env = CascadeEnvironment()
            obs = env.reset(
                task_id=str(tid),
                seed=int(sd),
                scenario_split="train",
                reward_mode="grpo",
                training_mode=True,
            )
            try:
                history = json.loads(hist_text or "[]")
            except Exception:
                history = []
            for item in history:
                try:
                    obs = env.step(CascadeAction(**item))
                except Exception:
                    break

            # ── Parse generated action ─────────────────────────────────
            text   = _completion_to_text(completion)
            action = parse_action_from_response(text, obs)
            legal  = env.get_legal_actions() if hasattr(env, "get_legal_actions") else []
            legal_types = {a["action_type"] for a in legal}

            # ── Step and get raw reward ────────────────────────────────
            next_obs = env.step(action)
            meta     = next_obs.metadata or {}
            reward   = float(meta.get("raw_reward", 0.0))

            # ── Format shaping ─────────────────────────────────────────
            if "<action>" in text.lower():
                reward += 0.08          # correct tag format
            else:
                reward -= 0.15          # missing tag

            # ── Validity shaping ──────────────────────────────────────
            if not meta.get("action_valid", True):
                reward -= 0.40          # illegal action
                if legal_types - {"wait"}:
                    reward -= 0.15      # better moves existed

            # ── Lazy-wait penalty ─────────────────────────────────────
            if action.action_type == "wait":
                better = [a for a in legal if a["action_type"] != "wait"]
                if   len(better) >= 4: reward -= 0.50
                elif len(better) >= 2: reward -= 0.30
                elif len(better) >= 1: reward -= 0.15

            # ── Recovery bonus ────────────────────────────────────────
            if action.action_type == "recover" and meta.get("action_valid", True):
                reward += 0.25

            # ── Infrastructure health bonus ───────────────────────────
            sector_summary = env._compute_sector_summary()
            avg_health = sum(sector_summary.values()) / max(len(sector_summary), 1)
            reward += 0.10 * (avg_health - 0.5)

            rewards.append(float(max(-3.0, min(3.0, reward))))

        # ── Group-normalise (GRPO baseline subtraction) ───────────────
        if len(rewards) >= 2:
            mu = statistics.mean(rewards)
            sd = statistics.pstdev(rewards)
            if sd > 0.01:
                rewards = [(r - mu) / sd for r in rewards]
            else:
                rewards = [0.0] * len(rewards)

        return rewards

    return cascade_grpo_reward


# ══════════════════════════════════════════════════════════════════════════
# 9.  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_model(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    max_seq_length: int,
    use_unsloth: bool,
    hf_token: str,
    cache_dir: Optional[str] = None,
):
    import torch
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    token = hf_token or None

    # Unsloth's RL guide recommends fp16 for GRPO / RL, and it handles the
    # remaining precision plumbing internally. Keep the whole stack on one
    # dtype to avoid Half/Float mismatches in custom kernels.
    torch_dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    backend = None
    model = tokenizer = peft_config = None

    # ── Try Unsloth first ──────────────────────────────────────────────
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            log.info("Loading model with Unsloth 4-bit QLoRA …")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=torch.float16,
                token=token,
                cache_dir=cache_dir,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=lora_alpha,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            backend = "unsloth_4bit"
            log.info("✓ Unsloth 4-bit QLoRA loaded.")
        except Exception as exc:
            log.warning("Unsloth load failed (%s) — falling back.", exc)
            model = None

    # ── Fallback: standard Transformers + PEFT ─────────────────────────
    if model is None:
        log.info("Loading model with standard Transformers + PEFT …")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, cache_dir=cache_dir
        )
        mkw: dict = {"token": token}
        if cache_dir is not None:
            mkw["cache_dir"] = cache_dir

        if torch.cuda.is_available():
            mkw["device_map"]  = "auto"
            mkw["dtype"] = torch_dtype
            try:
                mkw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
                log.info("  4-bit BitsAndBytes quantisation enabled.")
            except Exception as exc:
                log.warning("  BitsAndBytes unavailable (%s) — loading fp16/bf16.", exc)
        else:
            mkw["dtype"] = torch.float32
            log.warning("  No GPU detected — training will be very slow.")

        mkw = {k: v for k, v in mkw.items() if v is not None}
        model = AutoModelForCausalLM.from_pretrained(model_name, **mkw)
        _ensure_trainer_compatibility(model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )
        backend = "transformers_peft"
        log.info("✓ Transformers+PEFT loaded.")

    # ── Common tokenizer fixes ─────────────────────────────────────────
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer, peft_config, backend, torch_dtype


# ══════════════════════════════════════════════════════════════════════════
# 10.  GRPO TRAINING
# ══════════════════════════════════════════════════════════════════════════

def run_training(
    model,
    tokenizer,
    peft_config,
    backend: str,
    train_ds,
    reward_fn,
    args: argparse.Namespace,
    torch_dtype,
):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    log.info("Starting GRPO training — backend: %s", backend)
    log.info(
        "  Steps=%d  States=%d  Generations=%d  LR=%g",
        args.steps, args.states, args.generations, args.lr,
    )

    precision_kwargs = {
        "bf16": False,
        "fp16": False,
    }

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.generations,
        gradient_accumulation_steps=1,
        num_generations=args.generations,
        max_completion_length=args.max_completion_len,
        temperature=0.7,
        logging_steps=1,
        save_steps=max(args.steps, 1),
        report_to="none",
        remove_unused_columns=False,
        **precision_kwargs,
    )

    # TRL assumes `model.warnings_issued` exists; ensure it does for both
    # plain Transformers+PEFT and Unsloth wrapper stacks.
    _ensure_trainer_compatibility(model)

    # GRPOTrainer accepts peft_config=None (for Unsloth where PEFT is already applied)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,         # None for Unsloth
    )

    checkpoint = args.resume_from or None
    trainer.train(resume_from_checkpoint=checkpoint)

    save_path = f"{args.output_dir}/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    log.info("✓ Checkpoint saved → %s", save_path)
    return trainer


# ══════════════════════════════════════════════════════════════════════════
# 11.  POST-TRAINING EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model, tokenizer, args,
    eval_tasks, heuristic_policy,
    build_system_prompt, build_user_prompt, parse_action_from_response,
    CascadeEnvironment, TASK_CONFIGS, TASK_SEED_SPLITS,
):
    import torch

    SYSTEM_PROMPT = (
        build_system_prompt(grpo_mode=True)
        + "\nReturn exactly one <action>...</action> tag."
    )

    def make_prompt(obs):
        return SYSTEM_PROMPT + "\n\n" + build_user_prompt(obs)

    def grpo_policy(obs):
        prompt  = make_prompt(obs)
        inputs  = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=args.max_seq_len,
        ).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_action_from_response(text, obs), elapsed_ms

    log.info("Evaluating heuristic baseline …")
    heuristic_df = evaluate_policy(
        "heuristic", heuristic_policy, eval_tasks, args.eval_seeds,
        CascadeEnvironment, TASK_CONFIGS, TASK_SEED_SPLITS,
    )

    log.info("Evaluating trained GRPO policy …")
    grpo_df = evaluate_policy(
        "grpo_local", grpo_policy, eval_tasks, args.eval_seeds,
        CascadeEnvironment, TASK_CONFIGS, TASK_SEED_SPLITS,
    )

    all_results = pd.concat([heuristic_df, grpo_df], ignore_index=True)
    summary = (
        all_results.groupby("policy")
        .agg(
            runs              =("score",              "count"),
            mean_score        =("score",              "mean"),
            success_rate      =("success",            "mean"),
            invalid_action_rate=("invalid_action_rate","mean"),
            mean_budget_spent =("budget_spent",       "mean"),
            mean_crit_failures=("critical_failures",  "mean"),
            avg_response_ms   =("avg_response_ms",    "mean"),
        )
        .reset_index()
    )

    # ── Save ──────────────────────────────────────────────────────────
    all_results.to_csv(args.results_csv, index=False)
    summary.to_csv("cascadeguard_eval_summary.csv", index=False)
    log.info("Results saved → %s", args.results_csv)

    # ── Print to terminal ─────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== Per-Episode Results ===")
    print(all_results.drop(columns=["actions"]).to_string(index=False))
    print("\n=== Policy Summary ===")
    print(summary.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        ax = summary.plot(
            kind="bar", x="policy", y="mean_score",
            legend=False, figsize=(7, 4), ylim=(0, 1),
        )
        ax.set_title("CascadeGuard: GRPO vs Heuristic")
        ax.set_ylabel("Mean score")
        plt.tight_layout()
        plt.savefig("cascadeguard_policy_score.png", dpi=160)
        log.info("Plot saved → cascadeguard_policy_score.png")
    except Exception as exc:
        log.warning("Plot skipped: %s", exc)

    return all_results, summary


# ══════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Pin HuggingFace cache so weights are only downloaded once ──────
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["HF_HOME"]               = args.cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"]    = args.cache_dir
    log.info("HF cache dir: %s", args.cache_dir)

    # ── Go offline if model already cached (avoids WinError 10054) ─────
    # Transformers 5.x Qwen2 tokenizer calls is_base_mistral() which makes a
    # live HTTP request to HuggingFace EVEN when the model is fully cached.
    # On Windows this triggers "[WinError 10054] connection forcibly closed".
    # Setting HF_HUB_OFFLINE=1 forces ALL HF calls to use only local cache.
    # We only enable this when the model is already present — first run still
    # needs network access to do the initial download.
    _hub_dir = Path(args.cache_dir) / "hub"
    _model_last = args.model.split("/")[-1].lower()
    _cached = _hub_dir.exists() and any(
        p.is_dir() for p in _hub_dir.glob("models--*")
        if _model_last in p.name.lower()
    )
    if _cached:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log.info("Model found in cache → offline mode (no HF network calls). "
                 "Delete --cache-dir to force re-download.")
    else:
        log.info("Model not in cache yet → online mode (will download once).")

    # ── Smoke-test overrides ───────────────────────────────────────────
    if args.smoke_test:
        log.info("Smoke-test mode: steps=10, states=32, seeds=1")
        args.steps   = 10
        args.states  = 32
        args.eval_seeds = 1

    # ── LoRA alpha default ─────────────────────────────────────────────
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2

    # ── Eval tasks default ────────────────────────────────────────────
    if args.eval_tasks is None:
        args.eval_tasks = args.tasks

    # ── Find repo and set up sys.path ──────────────────────────────────
    project_root = find_project_root(args.repo_root)
    log.info("Project root: %s", project_root)
    os.chdir(project_root)

    # ── Import cascade_guard symbols ───────────────────────────────────
    (
        CascadeAction, CascadeEnvironment, TASK_CONFIGS,
        TASK_SEED_SPLITS, build_system_prompt, build_user_prompt,
        parse_action_from_response,
    ) = import_cascade(project_root)

    log.info("cascade_guard imported OK.")

    # ── Environment sanity check ───────────────────────────────────────
    run_env_sanity_check(
        CascadeEnvironment, CascadeAction, TASK_SEED_SPLITS, args.tasks
    )

    # ── Build heuristic policy ─────────────────────────────────────────
    heuristic_policy = make_heuristic_policy(CascadeAction)

    # ── Collect training states ────────────────────────────────────────
    rows = collect_training_states(
        num_states=args.states,
        train_tasks=args.tasks,
        TASK_SEED_SPLITS=TASK_SEED_SPLITS,
        CascadeEnvironment=CascadeEnvironment,
        heuristic_policy=heuristic_policy,
        build_system_prompt=build_system_prompt,
        build_user_prompt=build_user_prompt,
    )

    from datasets import Dataset
    train_ds = Dataset.from_list(rows)
    log.info("Training dataset: %s", train_ds)

    # ── Build reward function ──────────────────────────────────────────
    reward_fn = make_grpo_reward_fn(
        CascadeEnvironment, CascadeAction, parse_action_from_response
    )

    # ── Reward smoke test ──────────────────────────────────────────────
    test_reward = reward_fn(
        prompts=[rows[0]["prompt"]],
        completions=["<action>harden(POWER_GEN_1)</action>"],
        task_id=[rows[0]["task_id"]],
        seed=[rows[0]["seed"]],
        history_json=[rows[0]["history_json"]],
    )
    log.info("Reward smoke test: %s", test_reward)

    # ── Load model ─────────────────────────────────────────────────────
    use_unsloth = not args.no_unsloth
    model, tokenizer, peft_config, backend, torch_dtype = load_model(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_len,
        use_unsloth=use_unsloth,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    # ── Train ──────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_training(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        backend=backend,
        train_ds=train_ds,
        reward_fn=reward_fn,
        args=args,
        torch_dtype=torch_dtype,
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    if not args.no_eval:
        run_evaluation(
            model=model,
            tokenizer=tokenizer,
            args=args,
            eval_tasks=args.eval_tasks,
            heuristic_policy=heuristic_policy,
            build_system_prompt=build_system_prompt,
            build_user_prompt=build_user_prompt,
            parse_action_from_response=parse_action_from_response,
            CascadeEnvironment=CascadeEnvironment,
            TASK_CONFIGS=TASK_CONFIGS,
            TASK_SEED_SPLITS=TASK_SEED_SPLITS,
        )
    else:
        log.info("Eval skipped (--no-eval).")

    log.info("Done. Outputs in: %s/", args.output_dir)


if __name__ == "__main__":
    main()