"""Local model comparison harness for CascadeGuard.

Runs a fast zero-shot evaluation of small LLMs against the CascadeEnvironment
and prints a fitness leaderboard.  Designed to work on CPU or GPU — no 4-bit
quantization is forced (bitsandbytes is not required unless you have CUDA).

QUICK START (from the repo root):

    python temp/model_selection/local_compare.py

Override defaults:

    python temp/model_selection/local_compare.py \\
        --candidates Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \\
        --episodes 1 \\
        --max-steps 5

Results saved to:
    temp/model_selection/results/local_leaderboard.json
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

# ---------------------------------------------------------------------------
# Repo-root path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent          # …/cascade_guard
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Optional heavy deps — fail with a clear message
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    raise SystemExit(
        "PyTorch is required.\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit(
        "transformers is required.\n"
        "  pip install -U transformers accelerate"
    )

# ---------------------------------------------------------------------------
# CascadeGuard imports
# ---------------------------------------------------------------------------
try:
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    from cascade_guard.models import CascadeAction
    from cascade_guard.tasks import TASK_CONFIGS
except ModuleNotFoundError:
    try:
        from server.cascade_environment import CascadeEnvironment  # type: ignore
        from models import CascadeAction                           # type: ignore
        from tasks import TASK_CONFIGS                             # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import CascadeGuard modules: {exc}\n"
            "Run this script from the repo root:\n"
            "  cd /path/to/cascade_guard\n"
            "  python temp/model_selection/local_compare.py"
        )

# ---------------------------------------------------------------------------
# Defaults — small / fast for local runs
# ---------------------------------------------------------------------------

# These are tiny enough to run on CPU in a reasonable time.
DEFAULT_CANDIDATES: List[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

DEFAULT_TASKS: List[str] = list(TASK_CONFIGS.keys())

# Tight caps — we want quick relative comparison, not a perfect score.
DEFAULT_MAX_STEPS: int = 6

SYSTEM_PROMPT = (
    "You are an infrastructure resilience agent. "
    "Output ONLY a single JSON object on one line. "
    "Do not include any prose, markdown, or explanation. "
    'Example: {"action_type":"recover","target_node_id":"POWER_GEN_1"}'
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_id: str, device: str, hf_token: Optional[str]):
    """Load tokenizer + model.  Uses 4-bit on CUDA, float16 on MPS, float32 on CPU."""
    print(f"  device={device}", end="")
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "token": hf_token,
    }

    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb
            load_kwargs["device_map"] = "auto"
            print(" [4-bit]", end="")
        except ImportError:
                load_kwargs["dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                print(" [float16 — bitsandbytes not found]", end="")
    elif device == "mps":
        load_kwargs["dtype"] = torch.float16
        load_kwargs["device_map"] = "mps"
        print(" [float16]", end="")
    else:
        # CPU — use float32 (bfloat16 can help on modern CPUs but is less universal)
        load_kwargs["dtype"] = torch.float32
        print(" [float32 — CPU, will be SLOW for large models]", end="")

    print()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    mdl = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    mdl.eval()

    vram_gb = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return mdl, tok, vram_gb


# ---------------------------------------------------------------------------
# Observation summary
# ---------------------------------------------------------------------------

def _summarize_obs(obs: Any) -> str:
    fail = ",".join(getattr(obs, "active_failures", []) or []) or "-"
    pend = ",".join(getattr(obs, "pending_recoveries", []) or []) or "-"
    sectors = " ".join(
        f"{k[:3]}={v:.2f}"
        for k, v in (getattr(obs, "sector_summary", {}) or {}).items()
    )
    notable = []
    for n in getattr(obs, "nodes", []) or []:
        if n.health < 0.7 or n.load > 0.8 or (not n.is_operational) or n.observation_delayed:
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
        f"Budget:{obs.budget_remaining:.0f}\nNotable: {notable_s}"
    )


def _build_prompt(obs: Any, step: int, max_steps: int, legal: List[Dict]) -> str:
    legal_s = " | ".join(
        f"{a['action_type']}:{a.get('target_node_id') or 'null'}" for a in legal[:12]
    )
    return (
        f"Step {step}/{max_steps}\n"
        f"{_summarize_obs(obs)}\n"
        f"LegalActions: {legal_s}\n"
        "Pick ONE legal action. Output JSON now:"
    )


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    raw = (raw or "").strip()
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        body = parts[1] if len(parts) >= 2 else ""
        if body.startswith("json"):
            body = body[4:]
        raw = body.strip()
    s, e = raw.find("{"), raw.rfind("}")
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
# Generation
# ---------------------------------------------------------------------------

def _generate(model, tok, system_prompt: str, user_prompt: str,
              max_new_tokens: int, device: str) -> Tuple[str, int, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        ids = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        # Some transformers versions return a BatchEncoding instead of a tensor
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
    except Exception:
        text = system_prompt + "\n" + user_prompt + "\n"
        ids = tok(text, return_tensors="pt").input_ids

    # Move to model device if not using device_map="auto"
    target_device = next(model.parameters()).device
    ids = ids.to(target_device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    n_new = int(out.shape[1] - ids.shape[1])
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text, n_new, dt


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(task_id: str, seed: int, model, tok,
                 max_steps: int, max_new_tokens: int, device: str) -> Dict[str, Any]:
    env = CascadeEnvironment()
    obs = env.reset(task_id=task_id, seed=seed, reward_mode="grpo", training_mode=True)

    total_reward = 0.0
    n_valid = n_parseable = n_steps = 0
    tokens_used: List[int] = []
    times: List[float] = []

    for step in range(max_steps):
        legal = getattr(obs, "legal_actions", []) or []
        prompt = _build_prompt(obs, step, max_steps, legal)
        raw, n_tok, dt = _generate(model, tok, SYSTEM_PROMPT, prompt, max_new_tokens, device)
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

        obs = env.step(CascadeAction(
            action_type=action_dict.get("action_type", "wait"),
            target_node_id=action_dict.get("target_node_id"),
            parameters=action_dict.get("parameters") or {},
        ))
        total_reward += float(getattr(obs, "reward", 0.0) or 0.0)
        n_steps += 1
        if getattr(obs, "done", False) or getattr(obs, "terminated", False):
            break

    final_score = float(
        getattr(obs, "episode_score", None)
        or getattr(env, "episode_score", None)
        or 0.5
    )
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

def evaluate_model(
    model_id: str,
    tasks: List[str],
    episodes: int,
    seeds: List[int],
    max_steps: int,
    max_new_tokens: int,
    device: str,
    hf_token: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    bar = "=" * 76
    print(f"\n{bar}")
    print(f"  Loading: {model_id}")
    print(bar)

    t_load = time.time()
    try:
        model, tok, vram_gb = _load_model(model_id, device, hf_token)
    except Exception as e:
        traceback.print_exc()
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:160]}")
        return {"model_id": model_id, "load_error": f"{type(e).__name__}: {str(e)[:200]}"}

    load_seconds = time.time() - t_load
    print(f"  Loaded in {load_seconds:.1f}s  |  VRAM peak: {vram_gb:.2f} GB")

    episode_results: List[Dict[str, Any]] = []
    total_combos = len(tasks) * episodes
    done_combos = 0

    try:
        for task_id in tasks:
            for ep_i in range(episodes):
                seed = seeds[ep_i % len(seeds)]
                done_combos += 1
                prefix = f"  [{done_combos:>2d}/{total_combos}] {task_id:22s} ep{ep_i + 1}"
                print(f"{prefix}  running…", end="\r", flush=True)
                try:
                    res = _run_episode(task_id, seed, model, tok, max_steps, max_new_tokens, device)
                    episode_results.append(res)
                    line = (
                        f"{prefix}  "
                        f"reward={res['total_reward']:+7.2f}  "
                        f"score={res['final_score']:.3f}  "
                        f"valid={res['n_valid']}/{res['n_steps']}  "
                        f"parse={res['n_parseable']}/{res['n_steps']}  "
                        f"tok/step={res['mean_tokens']:>4.0f}  "
                        f"sec/step={res['mean_seconds']:.2f}"
                    )
                    if verbose:
                        print(line)
                    else:
                        print(line[:120])
                except Exception as e:
                    print()
                    traceback.print_exc()
                    print(f"{prefix}  ERROR: {type(e).__name__}: {str(e)[:100]}")
                    episode_results.append({
                        "task_id": task_id, "seed": seed,
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                    })
    finally:
        del model, tok
        gc.collect()
        if device == "cuda":
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
    n_steps_total  = sum(e["n_steps"]      for e in eps)
    n_valid_total  = sum(e["n_valid"]      for e in eps)
    n_parse_total  = sum(e["n_parseable"]  for e in eps)
    valid_rate     = n_valid_total  / max(n_steps_total, 1)
    parse_rate     = n_parse_total  / max(n_steps_total, 1)
    mean_tokens    = float(statistics.mean(e["mean_tokens"]  for e in eps))
    mean_seconds   = float(statistics.mean(e["mean_seconds"] for e in eps))

    shifted = max(statistics.mean(rewards) + 30.0, 0.5)
    cost    = max(mean_tokens, 1.0) * max(mean_seconds, 0.05)
    fitness = (shifted * max(valid_rate, 0.05)) / cost

    return {
        "mean_reward":   float(statistics.mean(rewards)),
        "median_reward": float(statistics.median(rewards)),
        "valid_rate":    float(valid_rate),
        "parse_rate":    float(parse_rate),
        "mean_tokens":   mean_tokens,
        "mean_seconds":  mean_seconds,
        "fitness":       float(fitness),
        "vram_gb":       float(model_result.get("vram_gb", 0.0)),
        "n_episodes":    len(eps),
    }


def _print_leaderboard(all_results: List[Dict[str, Any]]) -> None:
    W = 128
    print("\n" + "=" * W)
    print(
        f"{'MODEL':40s} {'ep_reward':>10s} {'med_R':>8s} "
        f"{'valid%':>7s} {'parse%':>7s} {'tok/step':>9s} {'sec/step':>9s} "
        f"{'vram':>6s} {'fitness':>10s}"
    )
    print("=" * W)

    rows: List[Tuple[float, str, Dict[str, float]]] = []
    for r in all_results:
        if r.get("load_error"):
            print(f"  {r['model_id']:40s}  LOAD FAILED: {r['load_error'][:70]}")
            continue
        agg = _aggregate(r)
        rows.append((agg["fitness"], r["model_id"], agg))

    rows.sort(reverse=True)
    for rank, (fit, mid, agg) in enumerate(rows, start=1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        n_ok  = agg["n_episodes"]
        print(
            f"{medal} {mid:38s} {agg['mean_reward']:+10.2f} {agg['median_reward']:+8.2f} "
            f"{agg['valid_rate'] * 100:>6.1f}% {agg['parse_rate'] * 100:>6.1f}% "
            f"{agg['mean_tokens']:>9.0f} {agg['mean_seconds']:>9.2f} "
            f"{agg['vram_gb']:>5.1f}G {fit:>10.4f}  (n={n_ok})"
        )

    print("=" * W)

    if rows:
        winner_fit, winner_id, winner_agg = rows[0]
        print(f"\n  WINNER: {winner_id}")
        print(
            f"    mean_reward = {winner_agg['mean_reward']:+.2f}  "
            f"valid_rate = {winner_agg['valid_rate'] * 100:.1f}%  "
            f"sec/step = {winner_agg['mean_seconds']:.2f}  "
            f"fitness = {winner_fit:.4f}"
        )
        print("  → Use this model as MODEL_NAME in the Colab GRPO notebook.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates", nargs="+", default=DEFAULT_CANDIDATES,
        help="HuggingFace model IDs to compare.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASKS,
        help="Task IDs from TASK_CONFIGS to evaluate on.",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per (model, task). Default 1 for a quick local run.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 7],
        help="Seed pool; cycled per episode.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max env steps per episode. Default {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Generation budget per step.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device: cuda / mps / cpu. Auto-detected if not set.",
    )
    parser.add_argument(
        "--hf-token", default=os.getenv("HF_TOKEN"),
        help="HuggingFace auth token (needed for gated models).",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "temp" / "model_selection" / "results" / "local_leaderboard.json"),
        help="Where to save the JSON results.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full-length episode lines.",
    )
    args = parser.parse_args()

    device = args.device or _select_device()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("  CascadeGuard — Local Model Comparison")
    print("=" * 76)
    print(f"  Device     : {device}")
    print(f"  Candidates : {len(args.candidates)}")
    for c in args.candidates:
        print(f"               {c}")
    print(f"  Tasks      : {', '.join(args.tasks)}")
    print(f"  Episodes   : {args.episodes} per (model, task)")
    print(f"  Max steps  : {args.max_steps}")
    print(f"  Output     : {out_path}")
    if not args.hf_token:
        print("  NOTE: no --hf-token / HF_TOKEN. Gated models (Llama) may fail.")
    print()

    all_results: List[Dict[str, Any]] = []
    for i, cand in enumerate(args.candidates, start=1):
        print(f"\n[{i}/{len(args.candidates)}] {cand}")
        try:
            result = evaluate_model(
                model_id=cand,
                tasks=args.tasks,
                episodes=args.episodes,
                seeds=args.seeds,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                device=device,
                hf_token=args.hf_token,
                verbose=args.verbose,
            )
        except Exception as e:
            traceback.print_exc()
            result = {"model_id": cand, "load_error": f"{type(e).__name__}: {str(e)[:200]}"}

        all_results.append(result)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Checkpoint saved → {out_path}")

    _print_leaderboard(all_results)
    print(f"Full JSON results → {out_path}\n")


if __name__ == "__main__":
    main()
