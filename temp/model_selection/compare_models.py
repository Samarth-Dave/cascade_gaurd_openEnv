"""Zero-shot model selection harness for CascadeGuard.

Loads each candidate base model in 4-bit, runs a small set of zero-shot
rollouts on each task, and reports a leaderboard scored by:

    fitness = mean_episode_reward * valid_action_rate / (mean_tokens * mean_seconds)

Higher fitness = more reward per token-second of compute.

USAGE (from repo root, e.g. inside Colab):

    !python temp/model_selection/compare_models.py \\
        --episodes 2 \\
        --max-steps-per-task 10

To restrict candidates:

    !python temp/model_selection/compare_models.py \\
        --candidates Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct

Results stream incrementally to:
    temp/model_selection/results/leaderboard.json

The script is conservative about VRAM: it loads ONE model at a time,
runs all evals, then frees memory before the next model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the cascade_guard package importable when launched from repo root or
# from Colab where the package is pip-installed.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError as e:
    raise SystemExit(
        "transformers + bitsandbytes are required. Install via:\n"
        "    pip install -U transformers accelerate bitsandbytes\n"
        f"original error: {e}"
    )

try:
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    from cascade_guard.models import CascadeAction
    from cascade_guard.tasks import TASK_CONFIGS
except ModuleNotFoundError:
    # Fallback: treat repo root as the package root (legacy layout).
    from server.cascade_environment import CascadeEnvironment  # type: ignore
    from models import CascadeAction  # type: ignore
    from tasks import TASK_CONFIGS  # type: ignore


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Thorough candidate set (user-selected). All are instruction-tuned and
# load fine in 4-bit on a single T4.
DEFAULT_CANDIDATES: List[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen3-1.7B",
]

DEFAULT_TASKS: List[str] = list(TASK_CONFIGS.keys())

# Tight max-step caps per task — we want a fast, comparable evaluation,
# not a perfect score. Caps below the env's natural max_steps are fine
# because we measure RELATIVE performance across models on identical states.
DEFAULT_MAX_STEPS_PER_TASK: Dict[str, int] = {
    "task_easy":           8,
    "task_medium":        10,
    "task_hard":          12,
    "task_gen_blackout":   8,
    "task_cyberattack":   10,
    "task_surge_demand":   8,
}

SYSTEM_PROMPT = (
    "You are an infrastructure resilience agent. "
    "Output ONLY a single JSON object on one line. "
    "Do not include any prose, markdown, or explanation. "
    'Example: {"action_type":"recover","target_node_id":"POWER_GEN_1"}'
)


# ---------------------------------------------------------------------------
# Prompting / parsing
# ---------------------------------------------------------------------------

def _summarize_obs(obs: Any) -> str:
    """Compact, single-string observation summary."""
    fail = ",".join(getattr(obs, "active_failures", []) or []) or "-"
    pend = ",".join(getattr(obs, "pending_recoveries", []) or []) or "-"
    sectors = " ".join(
        f"{k[:3]}={v:.2f}" for k, v in (getattr(obs, "sector_summary", {}) or {}).items()
    )
    notable = []
    for n in getattr(obs, "nodes", []) or []:
        if (n.health < 0.7 or n.load > 0.8 or (not n.is_operational)
                or n.observation_delayed):
            tag = ""
            if not n.is_operational:
                tag += " FAIL"
            if n.health < 0.7:
                tag += f" h={n.health:.2f}"
            if n.load > 0.8:
                tag += f" L={n.load:.2f}"
            if n.is_critical:
                tag += " CRIT"
            notable.append(f"{n.node_id}{tag}")
    notable_s = "; ".join(notable[:6]) if notable else "(all healthy)"
    return (
        f"Sectors:{sectors} Fail:{fail} Pend:{pend} "
        f"Budget:{obs.budget_remaining:.0f}\n"
        f"Notable: {notable_s}"
    )


def _build_user_prompt(obs: Any, step: int, max_steps: int, legal: List[Dict]) -> str:
    legal_s = " | ".join(
        f"{a['action_type']}:{a.get('target_node_id') or 'null'}" for a in legal[:12]
    )
    return (
        f"Step {step}/{max_steps}\n"
        f"{_summarize_obs(obs)}\n"
        f"LegalActions: {legal_s}\n"
        "Pick ONE legal action. Output JSON now:"
    )


def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    raw = (raw or "").strip()
    # Strip </think>, markdown fences, etc.
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)
        if len(raw) >= 2:
            body = raw[1]
            if body.startswith("json"):
                body = body[4:]
            raw = body.strip()
        else:
            raw = ""
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        d = json.loads(raw[s : e + 1])
    except Exception:
        return None
    if "action_type" not in d:
        return None
    return {
        "action_type": str(d.get("action_type", "wait")),
        "target_node_id": d.get("target_node_id"),
        "parameters": d.get("parameters") or {},
    }


def _is_legal(action: Dict[str, Any], legal: List[Dict]) -> bool:
    sig = (action.get("action_type"), action.get("target_node_id"))
    return any((la.get("action_type"), la.get("target_node_id")) == sig for la in legal)


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def _load_model(model_id: str, hf_token: Optional[str]):
    """Load model in 4-bit on cuda. Returns (model, tokenizer, vram_gb_peak)."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    mdl.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram = 0.0
    return mdl, tok, vram


def _generate(model, tok, system_prompt: str, user_prompt: str,
              max_new_tokens: int) -> Tuple[str, int, float]:
    """Returns (decoded_text, n_new_tokens, seconds)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        ids = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    except Exception:
        text = system_prompt + "\n" + user_prompt + "\n"
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,                # greedy → reproducible cost measurement
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    n_new = int(out.shape[1] - ids.shape[1])
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text, n_new, dt


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(task_id: str, seed: int, model, tok,
                 max_steps: int, max_new_tokens: int) -> Dict[str, Any]:
    env = CascadeEnvironment()

    # ✅ BUG 1 FIX: remove scenario_split, SAVE the returned obs
    obs = env.reset(
        task_id=task_id,
        seed=seed,
        reward_mode="grpo",
        training_mode=True,
    )

    total_reward = 0.0
    n_valid = 0
    n_parseable = 0
    n_steps = 0
    tokens_used: List[int] = []
    times: List[float] = []

    for step in range(max_steps):
        # ✅ BUG 2 FIX: use obs returned from reset/step, not env._build_observation()
        # ✅ BUG 3 FIX: get legal actions from obs attribute, not env.get_legal_actions()
        legal = getattr(obs, "legal_actions", []) or []

        prompt = _build_user_prompt(obs, step, max_steps, legal)
        raw, n_tok, dt = _generate(model, tok, SYSTEM_PROMPT, prompt, max_new_tokens)
        tokens_used.append(n_tok)
        times.append(dt)

        parsed = _parse_action(raw)
        if parsed is not None:
            n_parseable += 1
            if _is_legal(parsed, legal):
                action_dict = parsed
                n_valid += 1
            else:
                action_dict = legal[0] if legal else {"action_type": "wait", "target_node_id": None}
        else:
            action_dict = legal[0] if legal else {"action_type": "wait", "target_node_id": None}

        # env.step() returns the NEXT obs — save it
        obs = env.step(CascadeAction(
            action_type=action_dict.get("action_type", "wait"),
            target_node_id=action_dict.get("target_node_id"),
            parameters=action_dict.get("parameters") or {},
        ))
        total_reward += float(getattr(obs, "reward", 0.0) or 0.0)
        n_steps += 1
        if getattr(obs, "done", False) or getattr(obs, "terminated", False):
            break

    final_score = float(getattr(obs, "episode_score", None)
                        or getattr(env, "episode_score", None)
                        or 0.5)
    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": total_reward,
        "final_score": final_score,
        "n_steps": n_steps,
        "n_valid": n_valid,
        "n_parseable": n_parseable,
        "mean_tokens": float(statistics.mean(tokens_used)) if tokens_used else 0.0,
        "mean_seconds": float(statistics.mean(times)) if times else 0.0,
        "total_tokens": int(sum(tokens_used)),
        "total_seconds": float(sum(times)),
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_id: str, tasks: List[str], episodes: int,
                   seeds: List[int], max_steps_per_task: Dict[str, int],
                   max_new_tokens: int, hf_token: Optional[str]) -> Dict[str, Any]:
    bar = "=" * 76
    print(f"\n{bar}")
    print(f"  Loading {model_id}")
    print(bar)

    t_load = time.time()
    try:
        model, tok, vram_gb = _load_model(model_id, hf_token=hf_token)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:160]}")
        return {"model_id": model_id, "load_error": f"{type(e).__name__}: {str(e)[:200]}"}
    load_seconds = time.time() - t_load
    print(f"  loaded in {load_seconds:5.1f}s | VRAM peak: {vram_gb:5.2f} GB")

    episode_results: List[Dict[str, Any]] = []
    try:
        for task_id in tasks:
            cap = max_steps_per_task.get(task_id, 10)
            for ep_i in range(episodes):
                seed = seeds[ep_i % len(seeds)]
                try:
                    res = _run_episode(task_id, seed, model, tok, cap, max_new_tokens)
                    episode_results.append(res)
                    print(
                        f"  {task_id:22s} ep{ep_i + 1}/{episodes}  "
                        f"reward={res['total_reward']:+7.2f}  "
                        f"score={res['final_score']:.3f}  "
                        f"valid={res['n_valid']}/{res['n_steps']}  "
                        f"parse={res['n_parseable']}/{res['n_steps']}  "
                        f"tok/step={res['mean_tokens']:>4.0f}  "
                        f"sec/step={res['mean_seconds']:.2f}"
                    )
                except Exception as e:
                    traceback.print_exc()
                    print(f"  {task_id:22s} ep{ep_i + 1}/{episodes}  ERROR: {str(e)[:120]}")
                    episode_results.append({
                        "task_id": task_id, "seed": seed,
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                    })
    finally:
        del model
        del tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return {
        "model_id": model_id,
        "load_seconds": load_seconds,
        "vram_gb": vram_gb,
        "episodes": episode_results,
    }


# ---------------------------------------------------------------------------
# Aggregation & leaderboard
# ---------------------------------------------------------------------------

def _aggregate(model_result: Dict[str, Any]) -> Dict[str, float]:
    eps = [e for e in model_result.get("episodes", []) if "total_reward" in e]
    if not eps:
        return {
            "mean_reward": 0.0, "median_reward": 0.0,
            "valid_rate": 0.0, "parse_rate": 0.0,
            "mean_tokens": 0.0, "mean_seconds": 0.0,
            "fitness": 0.0, "vram_gb": float(model_result.get("vram_gb", 0.0)),
            "n_episodes": 0,
        }
    rewards = [e["total_reward"] for e in eps]
    n_steps_total = sum(e["n_steps"] for e in eps)
    n_valid_total = sum(e["n_valid"] for e in eps)
    n_parse_total = sum(e["n_parseable"] for e in eps)
    valid_rate = n_valid_total / max(n_steps_total, 1)
    parse_rate = n_parse_total / max(n_steps_total, 1)
    mean_tokens = float(statistics.mean(e["mean_tokens"] for e in eps))
    mean_seconds = float(statistics.mean(e["mean_seconds"] for e in eps))

    # Fitness blend: quality * validity / cost.
    # Use shifted reward so a model that scores 0 doesn't get fitness 0
    # purely because of its reward sign — we still want ordering.
    shifted = max(statistics.mean(rewards) + 30.0, 0.5)   # +30 baseline shift
    cost = max(mean_tokens, 1.0) * max(mean_seconds, 0.05)
    fitness = (shifted * max(valid_rate, 0.05)) / cost

    return {
        "mean_reward": float(statistics.mean(rewards)),
        "median_reward": float(statistics.median(rewards)),
        "valid_rate": float(valid_rate),
        "parse_rate": float(parse_rate),
        "mean_tokens": mean_tokens,
        "mean_seconds": mean_seconds,
        "fitness": float(fitness),
        "vram_gb": float(model_result.get("vram_gb", 0.0)),
        "n_episodes": len(eps),
    }


def _print_leaderboard(all_results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 124)
    print(
        f"{'MODEL':40s} {'ep_reward':>10s} {'med_R':>8s} "
        f"{'valid%':>7s} {'parse%':>7s} {'tok/step':>9s} {'sec/step':>9s} "
        f"{'vram':>6s} {'fitness':>10s}"
    )
    print("=" * 124)
    rows: List[Tuple[float, str, Dict[str, float]]] = []
    for r in all_results:
        if r.get("load_error"):
            print(f"{r['model_id']:40s}  LOAD FAILED: {r['load_error'][:70]}")
            continue
        agg = _aggregate(r)
        rows.append((agg["fitness"], r["model_id"], agg))
    rows.sort(reverse=True)
    for fit, mid, agg in rows:
        print(
            f"{mid:40s} {agg['mean_reward']:+10.2f} {agg['median_reward']:+8.2f} "
            f"{agg['valid_rate'] * 100:>6.1f}% {agg['parse_rate'] * 100:>6.1f}% "
            f"{agg['mean_tokens']:>9.0f} {agg['mean_seconds']:>9.2f} "
            f"{agg['vram_gb']:>5.1f}G {fit:>10.4f}"
        )
    print("=" * 124)
    if rows:
        winner = rows[0]
        print(f"\nWINNER (by fitness): {winner[1]}")
        print(f"  mean_reward={winner[2]['mean_reward']:+.2f}  "
              f"valid_rate={winner[2]['valid_rate'] * 100:.1f}%  "
              f"sec/step={winner[2]['mean_seconds']:.2f}")
        print("  → use this as MODEL_NAME in the Colab notebook for SFT + GRPO.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES,
                        help="HF model IDs to evaluate.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help="Task IDs to roll out on.")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes per (model, task).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 256],
                        help="Pool of seeds; cycled per episode.")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generation budget per step.")
    parser.add_argument("--max-steps-per-task", type=int, default=None,
                        help="Override DEFAULT_MAX_STEPS_PER_TASK uniformly.")
    parser.add_argument("--output", default="temp/model_selection/results/leaderboard.json")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"),
                        help="HF auth token for gated models (Llama-3.x).")
    args = parser.parse_args()

    if args.max_steps_per_task is not None:
        max_steps_map = {t: args.max_steps_per_task for t in args.tasks}
    else:
        max_steps_map = {
            t: DEFAULT_MAX_STEPS_PER_TASK.get(t, 10) for t in args.tasks
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Candidates: {len(args.candidates)}")
    print(f"Tasks:      {len(args.tasks)}  ({', '.join(args.tasks)})")
    print(f"Episodes:   {args.episodes}  per (model, task)")
    print(f"Caps:       {max_steps_map}")
    print(f"Output:     {out_path}")
    if not args.hf_token:
        print("NOTE: no --hf-token / HF_TOKEN found. Gated models (Llama-3.2) may fail to load.")

    all_results: List[Dict[str, Any]] = []
    for cand in args.candidates:
        try:
            result = evaluate_model(
                cand, args.tasks, args.episodes, args.seeds,
                max_steps_per_task=max_steps_map,
                max_new_tokens=args.max_new_tokens,
                hf_token=args.hf_token,
            )
        except Exception as e:
            traceback.print_exc()
            result = {"model_id": cand, "load_error": f"{type(e).__name__}: {str(e)[:200]}"}
        all_results.append(result)

        # Save incrementally so an interrupted run still leaves usable data.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  → leaderboard checkpoint saved to {out_path}")

    _print_leaderboard(all_results)
    print(f"\nFull JSON results: {out_path}")


if __name__ == "__main__":
    main()
