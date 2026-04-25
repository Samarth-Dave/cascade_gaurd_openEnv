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
    cd "C:\\path\\to\\repo"
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

    # Optional behavior-cloning warm-start before GRPO
    python train_grpo.py --sft-warmstart-steps 180

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
from collections import Counter
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    p.add_argument("--steps",       type=int, default=800,
                   help="Total GRPO training steps")
    p.add_argument("--states",      type=int, default=256,
                   help="Number of training state samples to collect")
    p.add_argument("--generations", type=int, default=8,
                   help="GRPO group size (completions per prompt)")
    p.add_argument("--lr",          type=float, default=8e-6,
                   help="Learning rate")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-completion-len", type=int, default=32)
    p.add_argument("--sft-warmstart-steps", type=int, default=180,
                   help="Optional supervised warm-start steps before GRPO (0 disables)")
    p.add_argument("--sft-warmstart-lr", type=float, default=1.5e-5,
                   help="Learning rate for optional SFT warm-start")
    p.add_argument("--action-only-prompts", action=argparse.BooleanOptionalAction, default=True,
                   help="Use strict action-only output contract in GRPO/SFT prompts")
    p.add_argument("--debug-parseability", action="store_true",
                   help="Sample model completions and log parseability snapshots")
    p.add_argument("--debug-parseability-samples", type=int, default=8,
                   help="Number of prompt samples for parseability snapshots")
    p.add_argument("--debug-hard-rollouts", action="store_true",
                   help="Dump step-level rollout logs for hard tasks and policy comparisons")
    p.add_argument("--debug-hard-rollout-seeds", type=int, default=1,
                   help="Validation seeds per hard task when --debug-hard-rollouts is enabled")
    p.add_argument("--reward-stats-every", type=int, default=20,
                   help="Log pre-normalization reward mean/std every N reward calls (0 disables)")

    # Tasks
    p.add_argument("--tasks", nargs="+",
                   default=[
                       "task_easy",
                       "task_medium",
                       "task_surge_demand",
                       "task_gen_blackout",
                       "task_hard",
                       "task_cyberattack",
                   ],
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
    server/cascade_environment.py.
    """
    if hint:
        candidate = Path(hint).expanduser().resolve()
        if (candidate / "server" / "cascade_environment.py").exists():
            return candidate
        raise RuntimeError(
            f"CASCADEGUARD_REPO '{hint}' does not look like the repo root.\n"
            "Expected to find:  server/cascade_environment.py"
        )
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "server" / "cascade_environment.py").exists():
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
    """Import all project symbols with package-first fallback."""
    repo_str = str(project_root)
    parent_str = str(project_root.parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)

    try:
        from cascade_guard.models import CascadeAction
        from cascade_guard.server.cascade_environment import CascadeEnvironment
        from cascade_guard.tasks import TASK_CONFIGS, TASK_SEED_SPLITS
        from cascade_guard.training.cot_prompt import (
            build_system_prompt,
            build_user_prompt,
            make_training_prompt,
            parse_action_from_response,
        )
    except ModuleNotFoundError:
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from models import CascadeAction
        from server.cascade_environment import CascadeEnvironment
        from tasks import TASK_CONFIGS, TASK_SEED_SPLITS
        from training.cot_prompt import (
            build_system_prompt,
            build_user_prompt,
            make_training_prompt,
            parse_action_from_response,
        )
    return CascadeAction, CascadeEnvironment, TASK_CONFIGS, TASK_SEED_SPLITS, \
           build_system_prompt, build_user_prompt, make_training_prompt, parse_action_from_response


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


def make_coverage_policy(CascadeAction, base_policy, coverage_target: dict):
    """
    Wraps a base heuristic policy so that underrepresented action types
    are injected into the SFT dataset until minimum coverage is met.

    coverage_target: {action_type: min_count}  — from ALL_ACTIONS config above.
    """
    counts = {a: 0 for a in coverage_target}
    RARE_ACTIONS = [
        "deploy_repair_crew", "emergency_shutdown", "patch_scada",
        "redistribute_load", "request_mutual_aid", "controlled_cascade",
        "multi_sector_lockdown", "reroute", "isolate",
    ]

    def policy(obs, env=None):
        nonlocal counts

        # Check if any rare action is below minimum
        needs_coverage = [
            a for a in RARE_ACTIONS
            if counts.get(a, 0) < coverage_target.get(a, 8)
        ]

        if needs_coverage:
            import random
            action_type = random.choice(needs_coverage)
            # Pick a valid target from obs if needed
            node_states = getattr(obs, "node_states", None)
            if node_states is None:
                nodes = [n.node_id for n in obs.nodes] if hasattr(obs, "nodes") else []
            else:
                nodes = list(node_states.keys())
            target = nodes[0] if nodes else None

            # Two-argument actions
            if action_type in ("reroute", "redistribute_load"):
                src = nodes[0] if len(nodes) >= 1 else None
                tgt = nodes[1] if len(nodes) >= 2 else nodes[0] if nodes else None
                action = CascadeAction(action_type=action_type,
                                       target_node_id=src,
                                       parameters={"source": src, "target": tgt})
            elif action_type in ("multi_sector_lockdown",):
                action = CascadeAction(action_type=action_type, target_node_id=None)
            else:
                action = CascadeAction(action_type=action_type, target_node_id=target)

            counts[action_type] = counts.get(action_type, 0) + 1
            return action

        # Default: use original heuristic
        try:
            action = base_policy(obs, env)
        except TypeError:
            action = base_policy(obs)
        at = getattr(action, "action_type", "wait")
        counts[at] = counts.get(at, 0) + 1
        return action

    return policy


def make_training_sub_policies(CascadeAction, fallback_policy):
    """Teacher sub-policies used for diverse state collection."""

    ADVANCED_ACTION_TYPES = {
        "reroute",
        "prioritize",
        "deploy_repair_crew",
        "emergency_shutdown",
        "cross_sector_bridge",
        "patch_scada",
        "redistribute_load",
        "request_mutual_aid",
        "controlled_cascade",
        "multi_sector_lockdown",
    }

    def _node_degree(obs) -> Dict[str, int]:
        degree: Dict[str, int] = {n.node_id: 0 for n in obs.nodes}
        for e in obs.edges:
            degree[e.source_id] = degree.get(e.source_id, 0) + 1
            degree[e.target_id] = degree.get(e.target_id, 0) + 1
        return degree

    def _legal_action_to_obj(legal_action: Dict[str, Any]):
        params: Dict[str, Any] = {}
        for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
            if key in legal_action:
                params[key] = legal_action[key]
        nested = legal_action.get("parameters") or {}
        if isinstance(nested, dict):
            for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
                if key in nested:
                    params[key] = nested[key]
        kwargs: Dict[str, Any] = {
            "action_type": str(legal_action.get("action_type", "wait")),
            "target_node_id": legal_action.get("target_node_id"),
        }
        if params:
            kwargs["parameters"] = params
        return CascadeAction(**kwargs)

    def _pick_legal_action(
        legal_actions: List[Dict[str, Any]],
        action_type: str,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        for candidate in legal_actions:
            if candidate.get("action_type") != action_type:
                continue
            if predicate is not None and not predicate(candidate):
                continue
            try:
                return _legal_action_to_obj(candidate)
            except Exception:
                continue
        return None

    def _advanced_intervention(obs, legal_actions: Optional[List[Dict[str, Any]]]):
        legal_actions = legal_actions or []
        if not legal_actions:
            return None

        diag = obs.diagnostics
        pressure = float(diag.system_pressure) if diag else 0.0
        node_map = {n.node_id: n for n in obs.nodes}
        degree = _node_degree(obs)
        failed_nodes = [n for n in obs.nodes if not n.is_operational]
        failed_sectors = {n.sector for n in failed_nodes}
        recover_legal_exists = any(a.get("action_type") == "recover" for a in legal_actions)

        if obs.task_id == "task_cyberattack":
            patch_action = _pick_legal_action(
                legal_actions,
                "patch_scada",
                predicate=lambda a: str(a.get("target_node_id", "")).startswith("TELECOM"),
            )
            if patch_action is None:
                patch_action = _pick_legal_action(legal_actions, "patch_scada")
            # In cyberattack scenarios patch_scada should dominate early decisions.
            if patch_action is not None and (obs.step <= 14 or pressure >= 0.30 or len(failed_nodes) >= 1):
                return patch_action

        if obs.task_id == "task_gen_blackout":
            pending_set = set(obs.pending_recoveries)
            if pending_set:
                deploy_action = _pick_legal_action(
                    legal_actions,
                    "deploy_repair_crew",
                    predicate=lambda a: (
                        a.get("target_node_id") in pending_set
                        and (
                            str(a.get("target_node_id", "")).startswith("POWER_GEN")
                            or str(a.get("target_node_id", "")).startswith("POWER_TRANS")
                            or (a.get("target_node_id") in node_map and node_map[a.get("target_node_id")].is_critical)
                        )
                    ),
                )
                if deploy_action is not None:
                    return deploy_action

            hospital_risk = any(
                (n.sector == "hospital") and ((not n.is_operational) or n.health < 0.65)
                for n in obs.nodes
            )
            if hospital_risk:
                reroute_to_hospital = _pick_legal_action(
                    legal_actions,
                    "reroute",
                    predicate=lambda a: (
                        (a.get("parameters") or {}).get("target") in node_map
                        and node_map[(a.get("parameters") or {}).get("target")].sector == "hospital"
                    ),
                )
                if reroute_to_hospital is not None:
                    return reroute_to_hospital

                bridge_action = _pick_legal_action(legal_actions, "cross_sector_bridge")
                if bridge_action is not None:
                    return bridge_action

        if obs.pending_recoveries:
            deploy_action = _pick_legal_action(
                legal_actions,
                "deploy_repair_crew",
                predicate=lambda a: (a.get("target_node_id") in set(obs.pending_recoveries)),
            )
            if deploy_action is not None and (
                pressure >= 0.40
                or any((not n.is_operational) and n.is_critical for n in obs.nodes)
            ):
                return deploy_action

        backbone_failed = any(
            (not n.is_operational)
            and ("GEN" in n.node_id or "TRANS" in n.node_id or "DIST" in n.node_id)
            for n in obs.nodes
        )
        if backbone_failed:
            reroute_action = _pick_legal_action(
                legal_actions,
                "reroute",
                predicate=lambda a: (
                    (a.get("parameters") or {}).get("target") in node_map
                    and (
                        node_map[(a.get("parameters") or {}).get("target")].is_critical
                        or node_map[(a.get("parameters") or {}).get("target")].sector == "hospital"
                    )
                ),
            )
            if reroute_action is None:
                reroute_action = _pick_legal_action(legal_actions, "reroute")
            if reroute_action is not None:
                return reroute_action

        bridge_action = _pick_legal_action(
            legal_actions,
            "cross_sector_bridge",
            predicate=lambda a: {
                ((a.get("parameters") or {}).get("sector_a") or a.get("target_node_id") or ""),
                (a.get("parameters") or {}).get("sector_b") or "",
            }
            == {"telecom", "hospital"},
        )
        if bridge_action is None:
            bridge_action = _pick_legal_action(legal_actions, "cross_sector_bridge")
        if bridge_action is not None and (
            "hospital" in failed_sectors
            and (
                "telecom" in failed_sectors
                or any(n.observation_delayed for n in obs.nodes if n.sector == "hospital")
            )
        ):
            return bridge_action

        prioritize_action = _pick_legal_action(
            legal_actions,
            "prioritize",
            predicate=lambda a: (
                (a.get("target_node_id") in node_map)
                and node_map[a.get("target_node_id")].is_operational
                and (node_map[a.get("target_node_id")].is_critical or node_map[a.get("target_node_id")].sector == "hospital")
                and node_map[a.get("target_node_id")].health < 0.80
            ),
        )
        if prioritize_action is not None and (pressure >= 0.25 or obs.weather_forecast != "clear"):
            return prioritize_action

        lockdown_action = _pick_legal_action(legal_actions, "multi_sector_lockdown")
        lockdown_pressure_threshold = 0.92 if obs.task_id == "task_hard" else 0.90
        lockdown_failed_threshold = 5 if obs.task_id == "task_hard" else 4
        if (
            lockdown_action is not None
            and pressure >= lockdown_pressure_threshold
            and len(failed_sectors) >= 3
            and len(failed_nodes) >= lockdown_failed_threshold
            and not recover_legal_exists
        ):
            return lockdown_action

        controlled_candidates = [
            a
            for a in legal_actions
            if a.get("action_type") == "controlled_cascade"
            and a.get("target_node_id") in node_map
            and node_map[a.get("target_node_id")].is_operational
            and not node_map[a.get("target_node_id")].is_critical
            and node_map[a.get("target_node_id")].sector != "hospital"
        ]
        if (
            controlled_candidates
            and pressure >= 0.95
            and len(failed_sectors) >= 3
            and len(failed_nodes) >= 5
            and not recover_legal_exists
            and obs.task_id != "task_cyberattack"
        ):
            controlled_candidates.sort(
                key=lambda a: (
                    degree.get(str(a.get("target_node_id")), 0),
                    node_map[str(a.get("target_node_id"))].health,
                )
            )
            try:
                return _legal_action_to_obj(controlled_candidates[0])
            except Exception:
                return None

        if pressure >= 0.80:
            emergency_action = _pick_legal_action(
                legal_actions,
                "emergency_shutdown",
                predicate=lambda a: (
                    (a.get("target_node_id") in node_map)
                    and not node_map[a.get("target_node_id")].is_critical
                    and node_map[a.get("target_node_id")].sector != "hospital"
                ),
            )
            if emergency_action is not None:
                return emergency_action

        return None

    def proactive_hardener(obs, legal_actions: Optional[List[Dict[str, Any]]] = None):
        advanced = _advanced_intervention(obs, legal_actions)
        if advanced is not None:
            return advanced

        if obs.budget_remaining >= 2.0 and obs.step < 4:
            degree = _node_degree(obs)
            candidates = [
                n for n in obs.nodes if n.is_operational and not n.is_hardened
            ]
            if candidates:
                candidates.sort(
                    key=lambda n: (
                        not n.is_critical,
                        -degree.get(n.node_id, 0),
                        n.health,
                    )
                )
                return CascadeAction(
                    action_type="harden",
                    target_node_id=candidates[0].node_id,
                )
        return fallback_policy(obs)

    def load_manager(obs, legal_actions: Optional[List[Dict[str, Any]]] = None):
        advanced = _advanced_intervention(obs, legal_actions)
        if advanced is not None:
            return advanced

        overloaded = [
            n
            for n in obs.nodes
            if n.is_operational
            and not n.is_critical
            and n.sector != "hospital"
            and n.load > 0.80
        ]
        if overloaded:
            overloaded.sort(key=lambda n: n.load, reverse=True)
            return CascadeAction(
                action_type="shed_load",
                target_node_id=overloaded[0].node_id,
            )

        delayed = [n for n in obs.nodes if n.observation_delayed]
        if delayed and obs.budget_remaining >= 1.0:
            delayed.sort(key=lambda n: (not n.is_critical, n.health))
            return CascadeAction(
                action_type="coordinate",
                target_node_id=delayed[0].node_id,
            )
        return fallback_policy(obs)

    def recovery_prioritizer(obs, legal_actions: Optional[List[Dict[str, Any]]] = None):
        advanced = _advanced_intervention(obs, legal_actions)
        if advanced is not None:
            return advanced

        diag = obs.diagnostics
        if diag and diag.recommended_recovery_order:
            return CascadeAction(
                action_type="recover",
                target_node_id=diag.recommended_recovery_order[0],
            )

        failed = [
            n
            for n in obs.nodes
            if not n.is_operational and n.node_id not in set(obs.pending_recoveries)
        ]
        if failed:
            failed.sort(key=lambda n: (not n.is_critical, n.health))
            return CascadeAction(action_type="recover", target_node_id=failed[0].node_id)
        return fallback_policy(obs)

    def advanced_crisis_manager(obs, legal_actions: Optional[List[Dict[str, Any]]] = None):
        advanced = _advanced_intervention(obs, legal_actions)
        if advanced is not None:
            return advanced
        return recovery_prioritizer(obs, legal_actions)

    return {
        "advanced_crisis_manager": advanced_crisis_manager,
        "proactive_hardener": proactive_hardener,
        "load_manager": load_manager,
        "recovery_prioritizer": recovery_prioritizer,
    }


# ══════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _looks_parseable_action(text: str) -> bool:
    """
    Returns True only if the output contains a complete <action>...</action> pair.
    A partial opening tag is NOT a parseable action — the model ran out of tokens.
    """
    # Must have opening tag, at least one word character, and closing tag
    if re.search(r'<action>\s*\w+\([^)]*\)\s*</action>', text, re.IGNORECASE):
        return True
    if re.search(r'"action_type"\s*:\s*"\w+"', text, re.IGNORECASE):
        return True
    return False

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
            parse_failures = 0
            critical_failures = 0
            rewards = []
            latencies = []
            actions_log = []
            action_counts: Counter = Counter()
            done = False

            while not done:
                t0 = time.perf_counter()
                try:
                    out = policy_fn(obs, env)
                except TypeError:
                    out = policy_fn(obs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                policy_meta: Dict[str, Any] = {}
                if isinstance(out, tuple):
                    action = out[0]
                    if len(out) >= 2 and isinstance(out[1], (int, float)):
                        elapsed_ms = float(out[1])
                    if len(out) >= 3 and isinstance(out[2], dict):
                        policy_meta = out[2]
                else:
                    action = out

                latencies.append(float(elapsed_ms))
                action_counts[action.action_type] += 1
                obs = env.step(action)
                rewards.append(float(obs.reward))
                meta = obs.metadata or {}
                parse_failed = bool(policy_meta.get("parse_failed", False))
                parse_failed = parse_failed or bool(meta.get("parse_error", False))

                actions_log.append({
                    "step": int(getattr(obs, "step", 0)),
                    "action_type": action.action_type,
                    "target_node_id": action.target_node_id,
                    "reward": float(obs.reward),
                    "action_valid": bool(meta.get("action_valid", True)),
                    "invalid_reason": str(meta.get("invalid_reason", "")),
                    "parse_failed": bool(parse_failed),
                    "critical_failures": len(obs.diagnostics.critical_failures) if obs.diagnostics else 0,
                    "budget_remaining": float(getattr(obs, "budget_remaining", 0.0)),
                    "system_pressure": float(obs.diagnostics.system_pressure) if obs.diagnostics else 0.0,
                })

                if not meta.get("action_valid", True):
                    invalid_actions += 1
                if parse_failed:
                    parse_failures += 1
                if obs.diagnostics:
                    critical_failures += len(obs.diagnostics.critical_failures)
                done = bool(obs.done)

            state = env.state
            avg_sector = sum(state.sector_health.values()) / max(len(state.sector_health), 1)
            total_actions = max(sum(action_counts.values()), 1)
            wait_count = int(action_counts.get("wait", 0))
            action_distribution = {
                action_name: int(count)
                for action_name, count in sorted(action_counts.items())
            }
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
                "wait_rate":          wait_count / total_actions,
                "action_diversity":   len(action_distribution),
                "action_distribution": json.dumps(action_distribution),
                "parse_failures":     parse_failures,
                "parse_failure_rate": parse_failures / total_actions,
                "critical_failures":  critical_failures,
                "avg_response_ms":    statistics.mean(latencies) if latencies else 0.0,
                "actions":            json.dumps(actions_log),
            })
    return pd.DataFrame(rows)


def _is_action_legal_quick(action, legal_actions: List[Dict[str, Any]]) -> bool:
    if not legal_actions:
        return True
    action_params = action.parameters or {}
    for legal in legal_actions:
        if legal.get("action_type") != action.action_type:
            continue
        if legal.get("target_node_id") != action.target_node_id:
            continue

        legal_params: Dict[str, Any] = {}
        nested = legal.get("parameters") or {}
        if isinstance(nested, dict):
            legal_params.update(nested)
        for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
            if key in legal:
                legal_params[key] = legal[key]

        mismatch = False
        for key, value in action_params.items():
            if key in legal_params and legal_params[key] != value:
                mismatch = True
                break
        if not mismatch:
            return True
    return False


def _reconstruct_observation_from_row(row: Dict[str, Any], CascadeEnvironment, CascadeAction):
    task_value = str(row.get("task_id", "task_easy"))
    seed_raw = row.get("seed", 42)
    try:
        seed_value = int(seed_raw)
    except (TypeError, ValueError):
        seed_value = 42

    env = CascadeEnvironment()
    obs = env.reset(
        task_id=task_value,
        seed=seed_value,
        scenario_split="train",
        reward_mode="grpo",
        training_mode=True,
    )

    hist_value = row.get("history_json", "[]")
    try:
        if isinstance(hist_value, str):
            history = json.loads(hist_value or "[]")
        elif isinstance(hist_value, list):
            history = hist_value
        else:
            history = []
    except Exception:
        history = []

    for item in history:
        if not isinstance(item, dict):
            continue
        if "action_type" not in item:
            continue
        try:
            obs = env.step(CascadeAction(**item))
        except Exception:
            break

    return env, obs


def log_parseability_snapshot(
    label: str,
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    CascadeEnvironment,
    CascadeAction,
    parse_action_from_response,
) -> None:
    if not rows:
        return

    sample_limit = int(max(1, args.debug_parseability_samples))
    sample_n = min(sample_limit, len(rows))
    rng = random.Random(2025 + len(rows))
    indices = list(range(len(rows)))
    if len(indices) > sample_n:
        indices = rng.sample(indices, sample_n)

    import torch

    parseable_count = 0
    wait_count = 0
    legal_count = 0
    completion_lengths: List[int] = []
    records: List[Dict[str, Any]] = []

    for idx in indices:
        row = rows[idx]
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue

        env, obs = _reconstruct_observation_from_row(row, CascadeEnvironment, CascadeAction)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parseable = _looks_parseable_action(text)
        action = parse_action_from_response(text, obs)
        legal_actions = env.get_legal_actions() if hasattr(env, "get_legal_actions") else []
        legal = _is_action_legal_quick(action, legal_actions)

        parseable_count += int(parseable)
        wait_count += int(action.action_type == "wait")
        legal_count += int(legal)
        completion_lengths.append(len(text))

        records.append({
            "sample_index": int(idx),
            "task_id": str(row.get("task_id", "")),
            "seed": int(row.get("seed", 0)) if str(row.get("seed", "")).isdigit() else row.get("seed"),
            "parseable": bool(parseable),
            "action_type": action.action_type,
            "action_target": action.target_node_id,
            "action_legal": bool(legal),
            "completion_chars": len(text),
            "completion": text,
        })

    if not records:
        return

    parseable_rate = parseable_count / max(len(records), 1)
    wait_rate = wait_count / max(len(records), 1)
    legal_rate = legal_count / max(len(records), 1)
    median_chars = int(statistics.median(completion_lengths)) if completion_lengths else 0

    debug_dir = Path(args.output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"parseability_{label}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    log.info(
        "Parseability snapshot [%s]: parseable=%.3f legal=%.3f wait=%.3f median_chars=%d (%s)",
        label,
        parseable_rate,
        legal_rate,
        wait_rate,
        median_chars,
        out_path,
    )


def export_hard_rollout_debug(output_dir: str, eval_df: pd.DataFrame, hard_tasks: Sequence[str]) -> None:
    if eval_df.empty:
        return

    hard_set = {str(t) for t in hard_tasks}
    hard_df = eval_df[eval_df["task_id"].astype(str).isin(hard_set)].copy()
    if hard_df.empty:
        return

    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv = debug_dir / "hard_rollouts_episodes.csv"
    episodes_json = debug_dir / "hard_rollouts_episodes.json"
    hard_df.to_csv(episodes_csv, index=False)
    hard_df.to_json(episodes_json, orient="records", indent=2)

    step_rows: List[Dict[str, Any]] = []
    for _, row in hard_df.iterrows():
        actions_payload = row.get("actions", "[]")
        try:
            actions = json.loads(actions_payload) if isinstance(actions_payload, str) else actions_payload
        except Exception:
            actions = []
        if not isinstance(actions, list):
            continue

        for step_entry in actions:
            if not isinstance(step_entry, dict):
                continue
            step_rows.append({
                "policy": row.get("policy"),
                "task_id": row.get("task_id"),
                "seed": row.get("seed"),
                "episode_score": row.get("score"),
                "step": step_entry.get("step"),
                "action_type": step_entry.get("action_type"),
                "target_node_id": step_entry.get("target_node_id"),
                "reward": step_entry.get("reward"),
                "action_valid": step_entry.get("action_valid"),
                "invalid_reason": step_entry.get("invalid_reason"),
                "parse_failed": step_entry.get("parse_failed"),
                "critical_failures": step_entry.get("critical_failures"),
                "budget_remaining": step_entry.get("budget_remaining"),
                "system_pressure": step_entry.get("system_pressure"),
            })

    if step_rows:
        steps_df = pd.DataFrame(step_rows)
        steps_csv = debug_dir / "hard_rollouts_steps.csv"
        steps_json = debug_dir / "hard_rollouts_steps.json"
        steps_df.to_csv(steps_csv, index=False)
        steps_df.to_json(steps_json, orient="records", indent=2)
        log.info(
            "Hard-rollout debug exported: episodes=%d steps=%d (%s)",
            len(hard_df),
            len(step_rows),
            steps_csv,
        )
    else:
        log.info("Hard-rollout debug exported: episodes=%d (no step entries parsed)", len(hard_df))


# ══════════════════════════════════════════════════════════════════════════
# 7.  TRAINING STATE COLLECTION
# ══════════════════════════════════════════════════════════════════════════

def collect_training_states(
    num_states: int,
    train_tasks: List[str],
    TASK_SEED_SPLITS: dict,
    CascadeAction,
    CascadeEnvironment,
    heuristic_policy,
    make_training_prompt_fn,
    action_only_prompts: bool,
) -> List[dict]:
    rows: List[dict] = []
    EPSILON_EXPLORE = 0.35
    EPSILON_STEPS = 8
    ADVANCED_EXPLORE_BIAS = 0.70
    MAX_CONSECUTIVE_WAITS = 3
    ADVANCED_ACTION_TYPES = {
        "reroute",
        "prioritize",
        "deploy_repair_crew",
        "emergency_shutdown",
        "cross_sector_bridge",
        "patch_scada",
        "redistribute_load",
        "request_mutual_aid",
        "controlled_cascade",
        "multi_sector_lockdown",
    }
    sub_policies = make_training_sub_policies(CascadeAction, heuristic_policy)
    policy_names = sorted(sub_policies.keys())

    def _legal_action_to_obj(legal_action: Dict[str, Any]):
        params: Dict[str, Any] = {}
        for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
            if key in legal_action:
                params[key] = legal_action[key]
        kwargs: Dict[str, Any] = {
            "action_type": str(legal_action.get("action_type", "wait")),
            "target_node_id": legal_action.get("target_node_id"),
        }
        if params:
            kwargs["parameters"] = params
        return CascadeAction(**kwargs)

    def _action_to_training_tag(action) -> str:
        params = action.parameters or {}
        if action.action_type == "reroute":
            source = params.get("source") or action.target_node_id or "SOURCE"
            target = params.get("target") or "TARGET"
            return f"<action>reroute({source},{target})</action>"
        if action.action_type == "cross_sector_bridge":
            sector_a = params.get("sector_a") or action.target_node_id or "SECTOR_A"
            sector_b = params.get("sector_b") or "SECTOR_B"
            return f"<action>cross_sector_bridge({sector_a},{sector_b})</action>"
        if action.action_type == "redistribute_load":
            node_a = params.get("node_a") or action.target_node_id or "NODE_A"
            node_b = params.get("node_b") or "NODE_B"
            return f"<action>redistribute_load({node_a},{node_b})</action>"
        if action.action_type == "request_mutual_aid":
            sector = params.get("sector") or action.target_node_id or "SECTOR"
            return f"<action>request_mutual_aid({sector})</action>"
        target = action.target_node_id if action.target_node_id else "null"
        return f"<action>{action.action_type}({target})</action>"

    def _is_safe_advanced_legal_action(legal_action: Dict[str, Any], obs) -> bool:
        action_type = str(legal_action.get("action_type", "wait"))
        target = legal_action.get("target_node_id")
        node_map = {n.node_id: n for n in obs.nodes}

        if action_type == "controlled_cascade":
            if target not in node_map:
                return False
            node = node_map[str(target)]
            return node.is_operational and not node.is_critical and node.sector != "hospital"

        if action_type == "emergency_shutdown" and target in node_map:
            node = node_map[str(target)]
            return not node.is_critical and node.sector != "hospital"

        if action_type == "multi_sector_lockdown":
            failed = [n for n in obs.nodes if not n.is_operational]
            failed_sectors = {n.sector for n in failed}
            pressure = float(obs.diagnostics.system_pressure) if obs.diagnostics else 0.0
            return pressure >= 0.65 and len(failed) >= 2 and len(failed_sectors) >= 2

        return True

    log.info("Collecting %d training states from tasks: %s", num_states, train_tasks)

    for task_id in train_tasks:
        for seed in TASK_SEED_SPLITS[task_id]["train"]:
            seed_basis = int(seed) + sum(ord(ch) for ch in task_id)
            episode_rng = random.Random(seed_basis)
            teacher_name = policy_names[seed_basis % len(policy_names)]
            teacher_policy = sub_policies[teacher_name]

            env = CascadeEnvironment()
            obs = env.reset(
                task_id=task_id,
                seed=seed,
                scenario_split="train",
                reward_mode="grpo",
                training_mode=True,
            )
            history: List[dict] = []
            step_idx = 0
            consecutive_waits = 0
            done = False

            while not done and len(rows) < num_states:
                legal = env.get_legal_actions() if hasattr(env, "get_legal_actions") else []
                non_wait = [a for a in legal if a.get("action_type") != "wait"]

                try:
                    action = teacher_policy(obs, legal)
                except TypeError:
                    action = teacher_policy(obs)
                except Exception:
                    action = heuristic_policy(obs)

                advanced_non_wait = [
                    a
                    for a in non_wait
                    if a.get("action_type") in ADVANCED_ACTION_TYPES
                    and _is_safe_advanced_legal_action(a, obs)
                ]

                if step_idx < EPSILON_STEPS and non_wait and episode_rng.random() < EPSILON_EXPLORE:
                    if advanced_non_wait and episode_rng.random() < ADVANCED_EXPLORE_BIAS:
                        action = _legal_action_to_obj(episode_rng.choice(advanced_non_wait))
                    else:
                        action = _legal_action_to_obj(episode_rng.choice(non_wait))

                if consecutive_waits >= MAX_CONSECUTIVE_WAITS and non_wait:
                    override_pool = advanced_non_wait if advanced_non_wait else non_wait
                    action = _legal_action_to_obj(episode_rng.choice(override_pool))
                    consecutive_waits = 0

                action_payload = {
                    "action_type": action.action_type,
                    "target_node_id": action.target_node_id,
                    "parameters": action.parameters or {},
                }
                rows.append({
                    "prompt":       make_training_prompt_fn(
                        obs,
                        env=env,
                        grpo_mode=True,
                        action_only=action_only_prompts,
                    ),
                    "task_id":      task_id,
                    "seed":         int(seed),
                    "teacher_policy": teacher_name,
                    "history_json": json.dumps(history),
                    "teacher_action_json": json.dumps(action_payload),
                    "teacher_action_text": _action_to_training_tag(action),
                })

                if action.action_type == "wait":
                    consecutive_waits += 1
                else:
                    consecutive_waits = 0

                history.append({
                    "action_type":   action.action_type,
                    "target_node_id": action.target_node_id,
                    "parameters": action.parameters or {},
                })
                obs = env.step(action)
                done = bool(obs.done)
                step_idx += 1

            if len(rows) >= num_states:
                log.info("  Collected %d states.", len(rows))
                return rows

    log.info("  Collected %d states.", len(rows))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 8.  GRPO REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def make_grpo_reward_fn(CascadeEnvironment, CascadeAction, parse_action_from_response, args: Optional[argparse.Namespace] = None):
    """
    Returns a reward function compatible with TRL GRPOTrainer.
    Closed over project symbols so it can be passed directly.
    """
    heuristic_teacher = make_heuristic_policy(CascadeAction)
    teacher_policies = make_training_sub_policies(CascadeAction, heuristic_teacher)
    reward_call_count = 0
    reward_stats_every = int(getattr(args, "reward_stats_every", 0)) if args is not None else 0

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

    def _legal_action_to_obj(legal_action: Dict[str, Any]):
        params: Dict[str, Any] = {}
        for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
            if key in legal_action:
                params[key] = legal_action[key]
        nested = legal_action.get("parameters") or {}
        if isinstance(nested, dict):
            for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
                if key in nested:
                    params[key] = nested[key]
        kwargs: Dict[str, Any] = {
            "action_type": str(legal_action.get("action_type", "wait")),
            "target_node_id": legal_action.get("target_node_id"),
        }
        if params:
            kwargs["parameters"] = params
        return CascadeAction(**kwargs)

    def _action_payload_to_obj(action_payload: Any):
        try:
            if isinstance(action_payload, str):
                parsed = json.loads(action_payload or "{}")
            elif isinstance(action_payload, dict):
                parsed = action_payload
            else:
                parsed = {}
        except Exception:
            parsed = {}

        if not isinstance(parsed, dict):
            return None
        if "action_type" not in parsed:
            return None

        try:
            return CascadeAction(
                action_type=str(parsed.get("action_type", "wait")),
                target_node_id=parsed.get("target_node_id"),
                parameters=parsed.get("parameters") or {},
            )
        except Exception:
            return None

    def _reconstruct_env(task_value: Any, seed_value: Any, hist_value: Any):
        env = CascadeEnvironment()
        obs = env.reset(
            task_id=str(task_value),
            seed=int(seed_value),
            scenario_split="train",
            reward_mode="grpo",
            training_mode=True,
        )
        try:
            if isinstance(hist_value, str):
                history = json.loads(hist_value or "[]")
            elif isinstance(hist_value, list):
                history = hist_value
            else:
                history = []
        except Exception:
            history = []

        for item in history:
            if not isinstance(item, dict):
                continue
            try:
                obs = env.step(CascadeAction(**item))
            except Exception:
                break
        return env, obs

    def _is_action_legal(action, legal_actions: List[Dict[str, Any]]) -> bool:
        if not legal_actions:
            return True
        action_params = action.parameters or {}
        for legal in legal_actions:
            if legal.get("action_type") != action.action_type:
                continue
            if legal.get("target_node_id") != action.target_node_id:
                continue

            legal_params: Dict[str, Any] = {}
            nested = legal.get("parameters") or {}
            if isinstance(nested, dict):
                legal_params.update(nested)
            for key in ("source", "target", "node_a", "node_b", "sector_a", "sector_b", "sector"):
                if key in legal:
                    legal_params[key] = legal[key]

            mismatch = False
            for key, value in action_params.items():
                if key in legal_params and legal_params[key] != value:
                    mismatch = True
                    break
            if not mismatch:
                return True
        return False

    def _select_teacher_action(
        obs,
        legal_actions: List[Dict[str, Any]],
        teacher_action_payload: Any,
        teacher_policy_name: Any,
    ):
        teacher_action = _action_payload_to_obj(teacher_action_payload)
        if teacher_action is None:
            policy = teacher_policies.get(str(teacher_policy_name), heuristic_teacher)
            try:
                teacher_action = policy(obs, legal_actions)
            except TypeError:
                teacher_action = policy(obs)
            except Exception:
                teacher_action = heuristic_teacher(obs)

        if not _is_action_legal(teacher_action, legal_actions):
            non_wait = [a for a in legal_actions if a.get("action_type") != "wait"]
            if non_wait:
                try:
                    return _legal_action_to_obj(non_wait[0])
                except Exception:
                    pass
            return CascadeAction(action_type="wait", target_node_id=None)
        return teacher_action

    def cascade_grpo_reward(
        prompts, completions,
        task_id=None,
        seed=None,
        history_json=None,
        teacher_action_json=None,
        teacher_policy=None,
        **kwargs,
    ) -> List[float]:
        """Teacher-relative GRPO reward with strict parseability pressure."""
        nonlocal reward_call_count
        reward_call_count += 1
        rewards: List[float] = []
        action_log: List[str] = []
        n = len(completions)
        task_ids = _ensure_list(task_id, n, "task_easy")
        seeds = _ensure_list(seed, n, 42)
        histories = _ensure_list(history_json, n, "[]")
        teacher_actions = _ensure_list(teacher_action_json, n, "{}")
        teacher_policy_names = _ensure_list(teacher_policy, n, "advanced_crisis_manager")

        for completion, tid, sd, hist_text, teacher_action_text, teacher_policy_name in zip(
            completions,
            task_ids,
            seeds,
            histories,
            teacher_actions,
            teacher_policy_names,
        ):
            agent_env, agent_obs = _reconstruct_env(tid, sd, hist_text)
            teacher_env, teacher_obs = _reconstruct_env(tid, sd, hist_text)

            text = _completion_to_text(completion)
            parseable = _looks_parseable_action(text)
            has_action_tag = bool(
                re.search(r"<action>\s*\w+\([^)]*\)\s*</action>", text, re.IGNORECASE)
            )
            has_function_call = bool(
                re.search(
                    (
                        r"\b(harden|recover|isolate|shed_load|coordinate|wait"
                        r"|reroute|prioritize|deploy_repair_crew|emergency_shutdown"
                        r"|cross_sector_bridge|patch_scada|redistribute_load"
                        r"|request_mutual_aid|controlled_cascade|multi_sector_lockdown)\([^)]*\)"
                    ),
                    text,
                    re.IGNORECASE,
                )
            )
            has_json_action = bool(
                re.search(
                    r'"action_type"\s*:\s*"\w+".*?"target_node_id"\s*:',
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
            )

            agent_action = parse_action_from_response(text, agent_obs)
            action_log.append(f"{agent_action.action_type}({agent_action.target_node_id})")
            agent_legal = agent_env.get_legal_actions() if hasattr(agent_env, "get_legal_actions") else []
            teacher_legal = teacher_env.get_legal_actions() if hasattr(teacher_env, "get_legal_actions") else []
            teacher_action = _select_teacher_action(
                teacher_obs,
                teacher_legal,
                teacher_action_text,
                teacher_policy_name,
            )

            agent_next_obs = agent_env.step(agent_action)
            teacher_next_obs = teacher_env.step(teacher_action)
            agent_meta = agent_next_obs.metadata or {}

            task_name = str(tid)
            critical_before = (
                len(agent_obs.diagnostics.critical_failures)
                if getattr(agent_obs, "diagnostics", None)
                else 0
            )
            critical_after = (
                len(agent_next_obs.diagnostics.critical_failures)
                if getattr(agent_next_obs, "diagnostics", None)
                else 0
            )
            failed_before = sum(1 for n in agent_obs.nodes if not n.is_operational)
            failed_after = sum(1 for n in agent_next_obs.nodes if not n.is_operational)

            score_agent = float(agent_env.state.score)
            score_teacher = float(teacher_env.state.score)
            score_delta = score_agent - score_teacher

            raw_agent = float(agent_meta.get("raw_reward", agent_next_obs.reward))
            raw_teacher = float((teacher_next_obs.metadata or {}).get("raw_reward", teacher_next_obs.reward))
            step_delta = raw_agent - raw_teacher

            # Main objective: causal improvement relative to teacher action from the same state.
            reward = 6.0 * score_delta + 0.35 * step_delta

            # Parseability is non-negotiable; unparseable completions receive a large penalty.
            if not parseable:
                reward -= 1.65

            if has_action_tag and has_function_call:
                reward += 0.20
            elif has_json_action:
                reward += 0.12
            else:
                reward -= 0.28

            legal_types = {a["action_type"] for a in agent_legal}
            if not agent_meta.get("action_valid", True):
                reward -= 0.45
                if legal_types - {"wait"}:
                    reward -= 0.15

            if agent_action.action_type == "wait":
                better = [a for a in agent_legal if a["action_type"] != "wait"]
                if len(better) >= 4:
                    reward -= 0.32
                elif len(better) >= 2:
                    reward -= 0.24
                elif len(better) >= 1:
                    reward -= 0.14

            if agent_action.action_type == "recover" and agent_meta.get("action_valid", True):
                reward += 0.10

            agent_budget_total = float(getattr(agent_env, "_budget_total", 10.0))
            if agent_budget_total > 0:
                agent_spent = agent_budget_total - float(getattr(agent_env, "_budget", agent_budget_total))
                teacher_spent = agent_budget_total - float(getattr(teacher_env, "_budget", agent_budget_total))

                spent_frac = agent_spent / agent_budget_total
                teacher_spent_frac = teacher_spent / agent_budget_total
                spent_delta = (agent_spent - teacher_spent) / agent_budget_total

                budget_delta_weight = 0.10
                utilization_weight = 0.06
                if task_name in {"task_surge_demand", "task_cyberattack", "task_hard"}:
                    budget_delta_weight = 0.24
                    utilization_weight = 0.12
                elif task_name == "task_gen_blackout":
                    budget_delta_weight = 0.18
                    utilization_weight = 0.10
                elif task_name == "task_easy":
                    budget_delta_weight = 0.05
                    utilization_weight = 0.03

                reward += budget_delta_weight * spent_delta

                util_denom = 0.60 if task_name in {"task_surge_demand", "task_cyberattack", "task_hard", "task_gen_blackout"} else 0.45
                reward += utilization_weight * min(spent_frac / max(util_denom, 1e-6), 1.0)

                # Penalize severe under-spend when the teacher must spend to protect score.
                if teacher_spent_frac >= 0.35 and spent_frac < teacher_spent_frac * 0.70:
                    under_ratio = (teacher_spent_frac - spent_frac) / max(teacher_spent_frac, 1e-6)
                    reward -= 0.24 * min(under_ratio, 1.5)

                # Penalize overspending past a bounded margin over teacher spend.
                allowed_spend = min(teacher_spent_frac + (0.12 if task_name != "task_easy" else 0.08), 0.92)
                if spent_frac > allowed_spend:
                    overspend_ratio = (spent_frac - allowed_spend) / max(1.0 - allowed_spend, 1e-6)
                    reward -= 0.28 * min(overspend_ratio, 1.5)

            # Task-specific shaping hooks.
            if task_name == "task_surge_demand":
                if critical_after < critical_before:
                    reward += 0.22
                if failed_after < failed_before:
                    reward += 0.10
                if agent_action.action_type in {"request_mutual_aid", "deploy_repair_crew", "prioritize", "reroute"}:
                    reward += 0.08
                if agent_action.action_type == "wait" and critical_before > 0:
                    reward -= 0.26

            if task_name == "task_easy":
                if agent_action.action_type in {"multi_sector_lockdown", "controlled_cascade", "emergency_shutdown"}:
                    reward -= 0.20
                if critical_after <= critical_before and agent_meta.get("action_valid", True):
                    reward += 0.05

            if task_name in {"task_hard", "task_gen_blackout", "task_cyberattack"}:
                if agent_action.action_type in {"reroute", "cross_sector_bridge"} and critical_after < critical_before:
                    reward += 0.12
                if task_name == "task_cyberattack" and agent_action.action_type == "patch_scada":
                    reward += 0.16
                if task_name in {"task_hard", "task_gen_blackout"} and agent_action.action_type == "deploy_repair_crew":
                    reward += 0.08

            rewards.append(float(max(-4.0, min(4.0, reward))))

        # ── Diversity diagnostic ──────────────────────────────────────
        # Log action diversity and reward variance so we can detect
        # diversity collapse (all completions → same action → 0 advantage).
        if reward_stats_every > 0 and rewards and (reward_call_count % reward_stats_every == 0):
            raw_mu = statistics.mean(rewards)
            raw_sd = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            unique_actions = len(set(action_log)) if action_log else 0
            log.info(
                "GRPO reward  call=%d  n=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f  "
                "unique_actions=%d/%d  actions=%s",
                reward_call_count,
                len(rewards),
                raw_mu, raw_sd, min(rewards), max(rewards),
                unique_actions, len(rewards),
                action_log[:8],
            )
            if raw_sd < 1e-4:
                log.warning(
                    "  ⚠ Near-zero reward std (%.6f) — all completions likely "
                    "produced the same action. Increase temperature or "
                    "num_generations.",
                    raw_sd,
                )

        # NOTE: Do NOT group-normalize here. TRL's GRPOTrainer already
        # computes per-group advantages = (reward - group_mean) / group_std.
        # Normalizing before returning makes TRL see mean≈0 rewards at every
        # step (masking real signal) and collapses near-zero-variance groups
        # to exactly 0.

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
        gradient_accumulation_steps=2,
        num_generations=args.generations,
        max_completion_length=args.max_completion_len,
        temperature=0.55,
        warmup_ratio=0.08,
        lr_scheduler_type="cosine",
        max_grad_norm=0.30,
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


def run_sft_warmstart(model, tokenizer, train_rows: List[dict], args: argparse.Namespace) -> None:
    """Optional behavior-cloning warm-start before GRPO."""
    if args.sft_warmstart_steps <= 0:
        return

    if not train_rows:
        log.warning("SFT warm-start skipped: no training rows were collected.")
        return

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        log.warning("SFT warm-start unavailable (%s); continuing with GRPO only.", exc)
        return

    sft_rows: List[Dict[str, str]] = []
    for row in train_rows:
        prompt = str(row.get("prompt", "")).strip()
        teacher_action_text = str(row.get("teacher_action_text", "<action>wait(null)</action>")).strip()
        if not prompt:
            continue
        sft_rows.append({"text": f"{prompt}\n{teacher_action_text}"})

    if not sft_rows:
        log.warning("SFT warm-start skipped: prompts were empty after preprocessing.")
        return

    sft_ds = Dataset.from_list(sft_rows)
    sft_output_dir = f"{args.output_dir}/sft_warmstart"
    log.info("Starting optional SFT warm-start: steps=%d rows=%d", args.sft_warmstart_steps, len(sft_rows))

    sft_args = SFTConfig(
        output_dir=sft_output_dir,
        max_steps=args.sft_warmstart_steps,
        learning_rate=args.sft_warmstart_lr,
        per_device_train_batch_size=max(1, min(args.generations, 4)),
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=max(args.sft_warmstart_steps, 1),
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_ds,
        processing_class=tokenizer,
        dataset_text_field="text",
    )
    trainer.train()
    log.info("✓ SFT warm-start complete.")


# ══════════════════════════════════════════════════════════════════════════
# 11.  POST-TRAINING EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model, tokenizer, args,
    eval_tasks, heuristic_policy,
    make_training_prompt_fn, parse_action_from_response,
    CascadeEnvironment, CascadeAction, TASK_CONFIGS, TASK_SEED_SPLITS,
):
    import torch

    def grpo_policy(obs, env=None):
        prompt = make_training_prompt_fn(
            obs,
            env=env,
            grpo_mode=True,
            action_only=args.action_only_prompts,
        )
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
        parse_failed = not _looks_parseable_action(text)
        return parse_action_from_response(text, obs), elapsed_ms, {"parse_failed": parse_failed}

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

    if args.debug_hard_rollouts:
        hard_tasks = [
            t for t in eval_tasks
            if t in {"task_gen_blackout", "task_hard", "task_cyberattack"}
        ]
        if hard_tasks:
            debug_frames = [heuristic_df, grpo_df]
            debug_seeds = max(1, min(args.debug_hard_rollout_seeds, args.eval_seeds))
            teacher_policies = make_training_sub_policies(CascadeAction, heuristic_policy)

            for policy_name in ("advanced_crisis_manager", "proactive_hardener"):
                teacher_policy = teacher_policies.get(policy_name)
                if teacher_policy is None:
                    continue

                def _teacher_policy(obs, env=None, _policy=teacher_policy):
                    legal_actions = env.get_legal_actions() if env is not None and hasattr(env, "get_legal_actions") else None
                    try:
                        return _policy(obs, legal_actions)
                    except TypeError:
                        return _policy(obs)

                teacher_df = evaluate_policy(
                    policy_name,
                    _teacher_policy,
                    hard_tasks,
                    debug_seeds,
                    CascadeEnvironment,
                    TASK_CONFIGS,
                    TASK_SEED_SPLITS,
                )
                debug_frames.append(teacher_df)

            debug_eval_df = pd.concat(debug_frames, ignore_index=True)
            export_hard_rollout_debug(args.output_dir, debug_eval_df, hard_tasks)
        else:
            log.info("Hard-rollout debug enabled but no hard tasks are present in eval task list.")

    all_results = pd.concat([heuristic_df, grpo_df], ignore_index=True)
    summary = (
        all_results.groupby("policy")
        .agg(
            runs              =("score",              "count"),
            mean_score        =("score",              "mean"),
            success_rate      =("success",            "mean"),
            invalid_action_rate=("invalid_action_rate","mean"),
            mean_budget_spent =("budget_spent",       "mean"),
            mean_wait_rate    =("wait_rate",          "mean"),
            mean_parse_failure_rate=("parse_failure_rate", "mean"),
            mean_crit_failures=("critical_failures",  "mean"),
            avg_response_ms   =("avg_response_ms",    "mean"),
        )
        .reset_index()
    )

    def _aggregate_action_distribution(values: pd.Series) -> str:
        total: Counter = Counter()
        for v in values:
            try:
                parsed = json.loads(v) if isinstance(v, str) else {}
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                for k, cnt in parsed.items():
                    total[str(k)] += int(cnt)
        return json.dumps(dict(sorted(total.items())))

    task_summary = (
        all_results.groupby(["policy", "task_id"])
        .agg(
            runs=("score", "count"),
            mean_score=("score", "mean"),
            wait_rate=("wait_rate", "mean"),
            parse_failure_rate=("parse_failure_rate", "mean"),
            mean_budget_spent=("budget_spent", "mean"),
            action_distribution=("action_distribution", _aggregate_action_distribution),
        )
        .reset_index()
    )

    # ── Save ──────────────────────────────────────────────────────────
    all_results.to_csv(args.results_csv, index=False)
    summary.to_csv("cascadeguard_eval_summary.csv", index=False)
    task_summary.to_csv("cascadeguard_eval_task_summary.csv", index=False)
    log.info("Results saved → %s", args.results_csv)

    # ── Print to terminal ─────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== Per-Episode Results ===")
    print(all_results.drop(columns=["actions"]).to_string(index=False))
    print("\n=== Policy Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Per-Task Summary ===")
    print(task_summary.to_string(index=False))

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

    # ── Import project symbols ─────────────────────────────────────────
    (
        CascadeAction, CascadeEnvironment, TASK_CONFIGS,
        TASK_SEED_SPLITS, build_system_prompt, build_user_prompt,
        make_training_prompt_fn,
        parse_action_from_response,
    ) = import_cascade(project_root)

    log.info("Project modules imported OK.")

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
        CascadeAction=CascadeAction,
        CascadeEnvironment=CascadeEnvironment,
        heuristic_policy=heuristic_policy,
        make_training_prompt_fn=make_training_prompt_fn,
        action_only_prompts=args.action_only_prompts,
    )

    from datasets import Dataset
    train_ds = Dataset.from_list(rows)
    log.info("Training dataset: %s", train_ds)

    # ── Build reward function ──────────────────────────────────────────
    reward_fn = make_grpo_reward_fn(
        CascadeEnvironment,
        CascadeAction,
        parse_action_from_response,
        args=args,
    )

    # ── Reward smoke test ──────────────────────────────────────────────
    test_reward = reward_fn(
        prompts=[rows[0]["prompt"]],
        completions=["<action>harden(POWER_GEN_1)</action>"],
        task_id=[rows[0]["task_id"]],
        seed=[rows[0]["seed"]],
        history_json=[rows[0]["history_json"]],
        teacher_action_json=[rows[0].get("teacher_action_json", "{}")],
        teacher_policy=[rows[0].get("teacher_policy", "advanced_crisis_manager")],
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

    # ── Optional SFT warm-start ───────────────────────────────────────
    run_sft_warmstart(model=model, tokenizer=tokenizer, train_rows=rows, args=args)

    if args.debug_parseability:
        log_parseability_snapshot(
            label="pre_grpo",
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            args=args,
            CascadeEnvironment=CascadeEnvironment,
            CascadeAction=CascadeAction,
            parse_action_from_response=parse_action_from_response,
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

    if args.debug_parseability:
        log_parseability_snapshot(
            label="post_grpo",
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            args=args,
            CascadeEnvironment=CascadeEnvironment,
            CascadeAction=CascadeAction,
            parse_action_from_response=parse_action_from_response,
        )

    # ── Evaluate ───────────────────────────────────────────────────────
    if not args.no_eval:
        run_evaluation(
            model=model,
            tokenizer=tokenizer,
            args=args,
            eval_tasks=args.eval_tasks,
            heuristic_policy=heuristic_policy,
            make_training_prompt_fn=make_training_prompt_fn,
            parse_action_from_response=parse_action_from_response,
            CascadeEnvironment=CascadeEnvironment,
            CascadeAction=CascadeAction,
            TASK_CONFIGS=TASK_CONFIGS,
            TASK_SEED_SPLITS=TASK_SEED_SPLITS,
        )
    else:
        log.info("Eval skipped (--no-eval).")

    log.info("Done. Outputs in: %s/", args.output_dir)


if __name__ == "__main__":
    main()