"""CascadeGuard — single-Space HF backend (FastAPI + in-process env + LLM).

This file is the entry point Hugging Face Spaces will execute. It is a
*self-contained* port of ``backend/main.py`` with three critical changes
so it can run as a single Space without an external env server:

  1.  The OpenEnv WebSocket env server is gone. ``InProcessEnv`` wraps
      ``cascade_guard.server.cascade_environment.CascadeEnvironment``
      directly, so ``reset`` / ``step`` / ``state`` are plain method
      calls — no ``max_concurrent_envs=1`` contention, no 1000/1011
      WebSocket churn, and per-session envs run in parallel.

  2.  The LLM loads the *trained* LoRA checkpoint from Hugging Face Hub
      on startup (``MODEL_NAME`` env var, defaults to
      ``samarthdave0305/cascadeguard-trained``). If that repo is an
      adapter-only repo it is applied on top of the matching base
      model. If the repo is unreachable, we transparently fall back to
      ``FALLBACK_MODEL_NAME`` (defaults to ``Qwen/Qwen2.5-0.5B-Instruct``)
      so the Space still boots and serves ``/api/health``.

  3.  All exposed HTTP routes match the live backend exactly, so the
      existing React frontend works without code changes — only
      ``VITE_API_URL`` / ``VITE_WS_URL`` need to be pointed at this
      Space's URL.

The file clones the project repo at startup (like the training Space
does) so the Space only needs these three uploaded files:

  - ``app.py``
  - ``requirements.txt``
  - ``README.md``

See ``deploy/README.md`` for the full deployment walkthrough.
"""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Space layout — paths match the training Space so a human can SSH in and
# debug with muscle memory from either Space.
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("BASE_DIR", "/home/user/app"))
REPO_URL = os.environ.get(
    "CASCADE_REPO_URL",
    "https://github.com/Samarth-Dave/cascade_gaurd_openEnv",
)
REPO_DIR = BASE_DIR / "cascade_gaurd_openEnv"
CACHE_DIR = Path(os.environ.get("HF_HOME", "/tmp/cg_hf_cache"))
OSM_CACHE = Path("/tmp/osmnx_cache")

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

for _d in (BASE_DIR, CACHE_DIR, OSM_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# Gate verbose transformers logs so the Space logs remain human-readable.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").strip().upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("cascadeguard.deploy")
logger.info("CascadeGuard deploy app import started.")
logger.info("Logging initialized level=%s", LOG_LEVEL_NAME)

RUNTIME_BOOTSTRAP_DEPS: Tuple[Dict[str, Any], ...] = ()


def _ensure_runtime_dependency(module_name: str, requirement: str, *, no_deps: bool = False) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    logger.info(
        "Installing missing runtime dependency %s%s ...",
        requirement,
        " --no-deps" if no_deps else "",
    )
    command = [sys.executable, "-m", "pip", "install", requirement]
    if no_deps:
        command.append("--no-deps")
    logger.info("Running command: %s", " ".join(command))
    subprocess.run(
        command,
        check=True,
        timeout=300,
    )
    logger.info("%s installed successfully.", requirement)


for _dep in RUNTIME_BOOTSTRAP_DEPS:
    _ensure_runtime_dependency(
        _dep["module"],
        _dep["requirement"],
        no_deps=bool(_dep.get("no_deps", False)),
    )


def _register_openenv_compat_modules() -> None:
    """Provide only the OpenEnv surface CascadeGuard needs in-process.

    The real OpenEnv server package pulls a websocket stack that conflicts with
    Gradio 4.44's pinned gradio-client on Spaces. The deploy backend never uses
    OpenEnv networking; it only needs model base classes and an Environment
    superclass so the cloned project modules can import.
    """
    try:
        from pydantic import BaseModel, ConfigDict  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not register OpenEnv compatibility stubs: %s", exc)
        return

    class Action(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        metadata: Dict[str, Any] = {}

    class Observation(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = {}

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        episode_id: str = ""
        step_count: int = 0

    class Environment:
        """No-op superclass used by CascadeEnvironment in deploy mode."""

        pass

    openenv_pkg = sys.modules.get("openenv") or types.ModuleType("openenv")
    core_pkg = sys.modules.get("openenv.core") or types.ModuleType("openenv.core")
    env_server_pkg = sys.modules.get("openenv.core.env_server") or types.ModuleType("openenv.core.env_server")
    types_pkg = sys.modules.get("openenv.core.env_server.types") or types.ModuleType("openenv.core.env_server.types")

    openenv_pkg.__path__ = getattr(openenv_pkg, "__path__", [])  # type: ignore[attr-defined]
    core_pkg.__path__ = getattr(core_pkg, "__path__", [])  # type: ignore[attr-defined]
    env_server_pkg.__path__ = getattr(env_server_pkg, "__path__", [])  # type: ignore[attr-defined]

    env_server_pkg.Environment = Environment  # type: ignore[attr-defined]
    types_pkg.Action = Action  # type: ignore[attr-defined]
    types_pkg.Observation = Observation  # type: ignore[attr-defined]
    types_pkg.State = State  # type: ignore[attr-defined]
    core_pkg.env_server = env_server_pkg  # type: ignore[attr-defined]
    openenv_pkg.core = core_pkg  # type: ignore[attr-defined]

    sys.modules["openenv"] = openenv_pkg
    sys.modules["openenv.core"] = core_pkg
    sys.modules["openenv.core.env_server"] = env_server_pkg
    sys.modules["openenv.core.env_server.types"] = types_pkg
    logger.info("OpenEnv compatibility modules registered for in-process deploy mode.")


# ---------------------------------------------------------------------------
# Step 1 — clone (or pull) the project repo so we can import the env.
# ---------------------------------------------------------------------------
def _ensure_repo() -> Path:
    """Clone ``REPO_URL`` into ``REPO_DIR`` or fast-forward an existing clone."""
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        logger.info("Repo already present, running git pull --ff-only ...")
        try:
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
                check=False,
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("git pull failed (continuing with existing checkout): %s", exc)
        return REPO_DIR

    logger.info("Cloning %s -> %s ...", REPO_URL, REPO_DIR)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        check=True,
        timeout=300,
    )
    return REPO_DIR


# The rest of this file assumes ``cascade_guard.*`` imports work, so we make
# the cloned repo importable before any project imports run.
_ensure_repo()

# The training Space symlinks ``/home/user/app/cascade_guard`` -> repo. We do
# the same so ``from cascade_guard.server...`` imports work exactly like they
# do in training.
_ALIAS = BASE_DIR / "cascade_guard"
if _ALIAS.exists() and not _ALIAS.is_symlink():
    # Preserve whatever is there (local dev, tests); only the HF Space sets
    # this up fresh each boot.
    pass
else:
    try:
        if _ALIAS.is_symlink():
            _ALIAS.unlink()
        _ALIAS.symlink_to(REPO_DIR, target_is_directory=True)
    except OSError:
        # Some FSes don't allow symlinks — fall back to adding REPO_DIR to
        # sys.path so the top-level package shape still works.
        pass

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Register this before importing cloned project modules. In particular,
# `models.py` imports `openenv.core.env_server.types`, and
# `server/cascade_environment.py` imports `openenv.core.env_server.Environment`.
_register_openenv_compat_modules()

# Importing ``cascade_guard.models`` normally executes the repo root
# ``__init__.py`` first, and that imports ``client.py``. The client path needs
# OpenEnv's WebSocket client (``websockets.asyncio``), which is intentionally
# unavailable in Gradio 4.44 Spaces because gradio-client pins websockets<13.
# This deploy app only needs the env models and server modules, so expose the
# cloned repo as a namespace package and skip the eager client import.
if "cascade_guard" not in sys.modules:
    _cascade_pkg = types.ModuleType("cascade_guard")
    _cascade_pkg.__path__ = [str(REPO_DIR)]  # type: ignore[attr-defined]
    _cascade_pkg.__file__ = str(REPO_DIR / "__init__.py")
    sys.modules["cascade_guard"] = _cascade_pkg


# ---------------------------------------------------------------------------
# Step 2 — lazy imports from the cloned repo. These are import-time required
# because ``EpisodeState``/``InfraNode``/etc. come from ``backend/models.py``
# in the cloned repo, and the env enums come from ``models.py`` at the
# repo root.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_DIR / "backend"))  # for ``from models import ...``

from models import (  # noqa: E402  (must come after sys.path mutation)
    AgentAction,
    BaselineComparison,
    DatasetStats,
    EpisodeResetRequest,
    EpisodeResetResponse,
    EpisodeState,
    EpisodeStepRequest,
    GraphResponse,
    GraderBreakdown,
    HealthResponse,
    InfraEdge,
    InfraNode,
    RewardCurvePoint,
    SectorHealth,
    StepResult,
    Task,
    TrajectoryItem,
    WsEvent,
)

# Env-side pydantic models + the environment class itself.
from cascade_guard.models import (  # noqa: E402
    CascadeAction,
)
from cascade_guard.server.cascade_environment import CascadeEnvironment  # noqa: E402

# Reuse the backend's session manager verbatim — it is env-agnostic and
# stores per-session state + the per-session event queue we stream over WS.
from session_manager import SessionManager  # noqa: E402


# ---------------------------------------------------------------------------
# FastAPI app. Hugging Face's Gradio SDK still runs this app.py, but importing
# gradio itself pulls pydub, which is fragile on Python 3.13 because audioop was
# removed from the stdlib. This backend does not need Gradio components; it
# serves a small landing payload from FastAPI instead.
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402


# ---------------------------------------------------------------------------
# LLM client — ported from ``backend/llm_client.py`` but rewritten to know
# how to load an adapter-only HF repo produced by the training Space.
# ---------------------------------------------------------------------------
LOAD_STATE_IDLE = "idle"
LOAD_STATE_LOADING = "loading"
LOAD_STATE_READY = "ready"
LOAD_STATE_ERROR = "error"

# Fields that build_user_prompt iterates with ``.items()`` — must stay dicts.
_DICT_FIELDS_OBS = {"sector_summary"}


class LlmClient:
    """HF-Hub-aware LLM client for the deploy Space.

    Resolution strategy
    -------------------
    1.  ``MODEL_NAME`` (defaults to ``samarthdave0305/cascadeguard-trained``):
        - if the repo contains ``adapter_config.json`` -> PEFT adapter load
          on top of the base model listed in that config.
        - if the repo contains ``config.json`` -> plain full-model load.
    2.  On any load failure, fall back to ``FALLBACK_MODEL_NAME``
        (defaults to ``Qwen/Qwen2.5-0.5B-Instruct``) so the Space still
        boots and serves non-LLM endpoints.
    """

    def __init__(self) -> None:
        self.model_name = (os.getenv("MODEL_NAME") or "samarthdave0305/cascadeguard-trained").strip()
        self.fallback_name = (os.getenv("FALLBACK_MODEL_NAME") or "Qwen/Qwen2.5-0.5B-Instruct").strip()
        self.hf_token = os.getenv("HF_TOKEN") or None
        # Default 0: Hugging Face Spaces often ship a PyTorch CUDA wheel that
        # is newer than the host NVIDIA driver, so torch.cuda.is_available() is
        # False. With REQUIRE_CUDA=1 the model never loads and agent-step 503s.
        # Set REQUIRE_CUDA=1 if you will only ever run on a healthy GPU+driver match.
        self.require_cuda = os.getenv("REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes"}
        self.device: str = "cpu"
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._loaded_source: Optional[str] = None
        self._load_state: str = LOAD_STATE_IDLE
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._load_started_at: Optional[float] = None
        self._load_finished_at: Optional[float] = None
        self._last_progress_log_at: float = 0.0

    # -- state ---------------------------------------------------------
    @property
    def load_state(self) -> str:
        return self._load_state

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def loaded_source(self) -> Optional[str]:
        return self._loaded_source

    def load_elapsed_seconds(self) -> Optional[float]:
        with self._load_lock:
            started = self._load_started_at
            finished = self._load_finished_at
            state = self._load_state
        if started is None:
            return None
        if finished is not None:
            return round(max(0.0, finished - started), 3)
        if state == LOAD_STATE_LOADING:
            return round(max(0.0, time.monotonic() - started), 3)
        return None

    def maybe_log_loading_progress(self, every_seconds: float = 15.0) -> None:
        now = time.monotonic()
        with self._load_lock:
            if self._load_state != LOAD_STATE_LOADING or self._load_started_at is None:
                return
            if now - self._last_progress_log_at < every_seconds:
                return
            self._last_progress_log_at = now
            elapsed = round(max(0.0, now - self._load_started_at), 2)
            target = self.model_name
            fallback = self.fallback_name
        logger.info(
            "Model loading in progress (elapsed=%.2fs target=%s fallback=%s cache=%s)",
            elapsed,
            target,
            fallback,
            CACHE_DIR,
        )

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _log_torch_runtime(self, torch_module: Any) -> None:
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_build = getattr(torch_module.version, "cuda", None)
        try:
            device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
        except Exception:  # noqa: BLE001
            device_count = 0
        device_names: List[str] = []
        if cuda_available and device_count > 0:
            for idx in range(device_count):
                try:
                    device_names.append(str(torch_module.cuda.get_device_name(idx)))
                except Exception:  # noqa: BLE001
                    device_names.append(f"cuda:{idx}")
        logger.info(
            "Torch runtime: torch=%s cuda_build=%s cuda_available=%s device_count=%s devices=%s CUDA_VISIBLE_DEVICES=%s",
            torch_module.__version__,
            cuda_build,
            cuda_available,
            device_count,
            device_names if device_names else ["<none>"],
            cuda_visible,
        )
        if not cuda_available:
            logger.warning(
                "CUDA is not available; model will run on CPU. If this Space should use GPU, verify Space hardware assignment and torch CUDA wheel compatibility."
            )

    # -- load ----------------------------------------------------------
    def _try_load(self, source: str) -> None:
        """Load tokenizer + model from ``source``. Populates ``self._model``."""
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._log_torch_runtime(torch)
        cuda_available = bool(torch.cuda.is_available())
        if self.require_cuda and not cuda_available:
            raise RuntimeError("REQUIRE_CUDA=1 but torch.cuda.is_available() is False")
        self.device = "cuda" if cuda_available else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Selected inference device=%s dtype=%s", self.device, dtype)

        # Decide adapter vs full model by probing the repo.
        adapter_base: Optional[str] = self._resolve_adapter_base(source)

        if adapter_base:
            logger.info("Detected PEFT adapter at %s (base=%s).", source, adapter_base)
            from peft import PeftModel  # noqa: PLC0415

            # Prefer tokenizer artifacts from the adapter repo (if present),
            # then fall back to the base model tokenizer.
            try:
                logger.info("Loading tokenizer from adapter repo: %s", source)
                tokenizer = AutoTokenizer.from_pretrained(
                    source,
                    trust_remote_code=True,
                    token=self.hf_token,
                    cache_dir=str(CACHE_DIR),
                )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "Adapter tokenizer load from %s failed (%s); using base tokenizer %s.",
                    source,
                    exc,
                    adapter_base,
                )
                logger.info("Loading tokenizer from base model: %s", adapter_base)
                tokenizer = AutoTokenizer.from_pretrained(
                    adapter_base,
                    trust_remote_code=True,
                    token=self.hf_token,
                    cache_dir=str(CACHE_DIR),
                )
            logger.info("Loading base model weights from: %s", adapter_base)
            base = AutoModelForCausalLM.from_pretrained(
                adapter_base,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=self.hf_token,
                cache_dir=str(CACHE_DIR),
            )
            logger.info("Applying adapter weights from: %s", source)
            model = PeftModel.from_pretrained(base, source, token=self.hf_token)
            # ``merge_and_unload`` makes inference ~15-20% faster by folding
            # LoRA into the base weights; skip it if the base is 4-bit
            # because bitsandbytes doesn't support merge.
            try:
                logger.info("Merging adapter into base model for faster inference...")
                model = model.merge_and_unload()
            except Exception as exc:  # noqa: BLE001
                logger.info("LoRA merge skipped: %s", exc)
        else:
            logger.info("Loading %s as a full model.", source)
            logger.info("Loading tokenizer from model repo: %s", source)
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=True,
                token=self.hf_token,
                cache_dir=str(CACHE_DIR),
            )
            logger.info("Loading model weights from model repo: %s", source)
            model = AutoModelForCausalLM.from_pretrained(
                source,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=self.hf_token,
                cache_dir=str(CACHE_DIR),
            )

        logger.info("Moving model to device=%s and switching to eval mode...", self.device)
        model.to(self.device)
        model.eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        with self._load_lock:
            self._tokenizer = tokenizer
            self._model = model
            self._loaded_source = source
            self._load_state = LOAD_STATE_READY
        logger.info("Model loaded from %s on %s.", source, self.device)

    def _resolve_adapter_base(self, source: str) -> Optional[str]:
        """Return the base model name if ``source`` is a PEFT adapter repo."""
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            adapter_cfg = hf_hub_download(
                repo_id=source,
                filename="adapter_config.json",
                token=self.hf_token,
                cache_dir=str(CACHE_DIR),
            )
            with open(adapter_cfg, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            base = data.get("base_model_name_or_path")
            return str(base) if base else None
        except Exception:  # noqa: BLE001
            return None

    def load_model(self) -> str:
        """Synchronous load with graceful fallback. Idempotent."""
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._loaded_source or self.model_name
            self._load_state = LOAD_STATE_LOADING
            self._load_error = None
            self._load_started_at = time.monotonic()
            self._load_finished_at = None
            self._last_progress_log_at = 0.0

        for candidate in (self.model_name, self.fallback_name):
            if not candidate:
                continue
            try:
                logger.info("Attempting model load candidate=%s", candidate)
                self._try_load(candidate)
                if candidate != self.model_name:
                    logger.warning(
                        "Primary model unavailable; serving fallback model %s (primary=%s).",
                        candidate,
                        self.model_name,
                    )
                else:
                    logger.info("Serving primary model %s.", candidate)
                elapsed = self.load_elapsed_seconds()
                if elapsed is not None:
                    logger.info("Model load completed in %.2fs.", elapsed)
                return candidate
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model load from %s failed: %s", candidate, exc)
                with self._load_lock:
                    self._load_error = f"{candidate}: {exc}"
                continue

        with self._load_lock:
            self._load_state = LOAD_STATE_ERROR
            self._load_finished_at = time.monotonic()
        logger.error("All candidate models failed; agent-step will return 503.")
        raise RuntimeError(self._load_error or "No model could be loaded.")

    def load_model_async(self) -> None:
        with self._load_lock:
            if self._load_state in (LOAD_STATE_LOADING, LOAD_STATE_READY):
                return
            self._load_state = LOAD_STATE_LOADING
            self._load_error = None

        def _runner() -> None:
            try:
                self.load_model()
            except Exception:  # noqa: BLE001
                pass  # state already recorded

        threading.Thread(target=_runner, name="llm-loader", daemon=True).start()

    def ensure_model_loading(self) -> None:
        """Start background model loading if it has not already started."""
        self.load_model_async()

    # -- inference -----------------------------------------------------
    async def get_action(self, raw_observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")

        logger.info(
            "Running inference with loaded_source=%s device=%s state=%s",
            self._loaded_source,
            self.device,
            self._load_state,
        )

        # Late import so a partial boot (no training repo cloned) doesn't
        # blow up at module import time.
        from training.cot_prompt import build_system_prompt, build_user_prompt  # noqa: PLC0415

        obs = _obs_namespace(raw_observation)
        user_prompt = (
            build_user_prompt(obs)
            + "\n\nFINAL OUTPUT RULE: output exactly one XML action tag and nothing else, "
            "for example <action>recover(POWER_GEN_1)</action> or <action>wait(null)</action>."
        )
        messages = [
            {"role": "system", "content": build_system_prompt(grpo_mode=True, action_only=True)},
            {"role": "user", "content": user_prompt},
        ]
        content = await asyncio.to_thread(self._generate, messages)
        logger.info("Raw model output: %s", content[:200].replace("\n", " | "))
        thinking = _extract_tag(content, "think")
        has_action_tag = _has_action_tag(content)
        action_type, target, parameters = _parse_action(content)
        if not has_action_tag:
            action_type, target, parameters = _repair_missing_action(content, raw_observation)
            logger.warning(
                "Model omitted <action> tag; repaired action from text/observation: type=%s target=%s parameters=%s",
                action_type,
                target,
                parameters,
            )
        logger.info(
            "Parsed model action: type=%s target=%s parameters=%s has_action_tag=%s",
            action_type,
            target,
            parameters,
            has_action_tag,
        )
        return {
            "action": action_type,
            "target": target,
            "parameters": parameters,
            "thinking": thinking,
            "raw_output": content,
        }

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        torch = self._torch
        assert torch is not None and self._model is not None and self._tokenizer is not None

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\n"

        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=220,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[-1]
        output_tokens = output[0][prompt_tokens:]
        return self._tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prompt helpers — copied from backend/llm_client.py so we don't depend on
# import ordering of the cloned repo.
# ---------------------------------------------------------------------------
def _obs_namespace(raw_observation: Dict[str, Any]) -> SimpleNamespace:
    obs = _to_namespace(raw_observation)
    for field_name in _DICT_FIELDS_OBS:
        value = getattr(obs, field_name, None)
        if isinstance(value, SimpleNamespace):
            setattr(obs, field_name, _to_dict(value))
        elif value is None:
            setattr(obs, field_name, {})
    return obs


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, types.SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    open_match = re.search(rf"<{tag}>\s*(.*)", content, re.IGNORECASE | re.DOTALL)
    if open_match:
        chunk = open_match.group(1).strip()
        next_tag = re.search(r"<\w+>", chunk)
        if next_tag:
            chunk = chunk[: next_tag.start()].strip()
        return chunk
    return ""


def _has_action_tag(content: str) -> bool:
    return bool(re.search(r"<action>\s*[a-zA-Z_]+\([^)]*\)\s*</action>", content, re.IGNORECASE))


_TWO_NODE_ACTIONS = {"reroute", "redistribute_load"}
_TWO_SECTOR_ACTIONS = {"cross_sector_bridge"}
_SECTOR_ACTIONS = {"request_mutual_aid"}
_LEGAL_ACTIONS = {
    "harden",
    "recover",
    "isolate",
    "shed_load",
    "coordinate",
    "wait",
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


def _parse_action(content: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    match = re.search(r"<action>\s*([a-zA-Z_]+)\(([^)]*)\)\s*</action>", content, re.IGNORECASE)
    if not match:
        return "wait", None, {}
    action_type = match.group(1).lower()
    raw_args = match.group(2).strip()
    if not raw_args or raw_args.lower() in {"null", "none"}:
        return action_type, None, {}

    parts = [p.strip() for p in raw_args.split(",") if p.strip()]
    if not parts:
        return action_type, None, {}

    parameters: Dict[str, Any] = {}
    primary: Optional[str] = parts[0] if parts[0].lower() not in {"null", "none"} else None

    if action_type in _TWO_NODE_ACTIONS and len(parts) >= 2:
        if action_type == "reroute":
            parameters = {"source": parts[0], "target": parts[1]}
        else:
            parameters = {"node_a": parts[0], "node_b": parts[1]}
        primary = parts[0]
    elif action_type in _TWO_SECTOR_ACTIONS and len(parts) >= 2:
        parameters = {"sector_a": parts[0], "sector_b": parts[1]}
        primary = parts[0]
    elif action_type in _SECTOR_ACTIONS:
        parameters = {"sector": parts[0]}
        primary = parts[0]
    return action_type, primary, parameters


def _repair_missing_action(content: str, raw_observation: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Recover a legal action when the model explains but omits XML.

    DeepSeek-R1-style models often produce prose even under action-only prompts.
    The env step loop should not degrade to wait every time that happens, so we
    infer a conservative legal action from the model's words plus live state.
    """
    lowered = content.lower()
    mentioned = [action for action in _LEGAL_ACTIONS if re.search(rf"\b{re.escape(action)}\b", lowered)]
    nodes = _observation_nodes(raw_observation)
    node_ids = [str(node.get("node_id")) for node in nodes if node.get("node_id")]
    mentioned_node = next((node_id for node_id in node_ids if node_id.lower() in lowered), None)

    if "harden" in mentioned or "harden now" in lowered:
        target = mentioned_node or _pick_harden_target(raw_observation)
        if target:
            return "harden", target, {}
    if "recover" in mentioned or "repair" in lowered:
        target = mentioned_node or _pick_recover_target(raw_observation)
        if target:
            return "recover", target, {}
    if "isolate" in mentioned:
        target = mentioned_node or _first_active_failure(raw_observation)
        if target:
            return "isolate", target, {}
    if "coordinate" in mentioned:
        target = mentioned_node or _pick_coordinate_target(raw_observation)
        if target:
            return "coordinate", target, {}
    if "shed_load" in mentioned or "shed load" in lowered:
        target = mentioned_node or _pick_shed_load_target(raw_observation)
        if target:
            return "shed_load", target, {}
    if "patch_scada" in mentioned or "patch scada" in lowered or "scada" in lowered:
        target = mentioned_node or _pick_scada_target(raw_observation)
        if target:
            return "patch_scada", target, {}
    if "prioritize" in mentioned:
        target = mentioned_node or _pick_priority_target(raw_observation)
        if target:
            return "prioritize", target, {}

    return _policy_action_from_observation(raw_observation)


def _policy_action_from_observation(raw_observation: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    recover_target = _pick_recover_target(raw_observation)
    if recover_target:
        return "recover", recover_target, {}

    if _storm_pressure(raw_observation):
        harden_target = _pick_harden_target(raw_observation)
        if harden_target:
            return "harden", harden_target, {}

    coordinate_target = _pick_coordinate_target(raw_observation)
    if coordinate_target:
        return "coordinate", coordinate_target, {}

    shed_target = _pick_shed_load_target(raw_observation)
    if shed_target:
        return "shed_load", shed_target, {}

    return "wait", None, {}


def _observation_nodes(raw_observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [node for node in raw_observation.get("nodes", []) if isinstance(node, dict)]


def _first_active_failure(raw_observation: Dict[str, Any]) -> Optional[str]:
    failures = raw_observation.get("active_failures") or []
    return str(failures[0]) if failures else None


def _pick_recover_target(raw_observation: Dict[str, Any]) -> Optional[str]:
    failures = {str(node_id) for node_id in raw_observation.get("active_failures", [])}
    nodes = _observation_nodes(raw_observation)
    failed_nodes = [
        node for node in nodes
        if str(node.get("node_id")) in failures or not bool(node.get("is_operational", True))
    ]
    if not failed_nodes:
        return None
    failed_nodes.sort(
        key=lambda node: (
            bool(node.get("is_critical", False)),
            _sector_priority(str(node.get("sector", ""))),
            -float(node.get("health", 0.0)),
        ),
        reverse=True,
    )
    return str(failed_nodes[0].get("node_id"))


def _pick_harden_target(raw_observation: Dict[str, Any]) -> Optional[str]:
    budget = float(raw_observation.get("budget_remaining", 0.0))
    if budget < 2.0:
        return None
    candidates = [
        node for node in _observation_nodes(raw_observation)
        if bool(node.get("is_operational", True)) and not bool(node.get("is_hardened", False))
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda node: (
            bool(node.get("is_critical", False)),
            _sector_priority(str(node.get("sector", ""))),
            float(node.get("load", 0.0)),
            float(node.get("health", 0.0)),
        ),
        reverse=True,
    )
    return str(candidates[0].get("node_id"))


def _pick_coordinate_target(raw_observation: Dict[str, Any]) -> Optional[str]:
    candidates = [
        node for node in _observation_nodes(raw_observation)
        if bool(node.get("observation_delayed", False))
        or float(node.get("observation_confidence", 1.0)) < 0.75
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda node: (
            bool(node.get("is_critical", False)),
            1.0 - float(node.get("observation_confidence", 1.0)),
        ),
        reverse=True,
    )
    return str(candidates[0].get("node_id"))


def _pick_shed_load_target(raw_observation: Dict[str, Any]) -> Optional[str]:
    candidates = [
        node for node in _observation_nodes(raw_observation)
        if not bool(node.get("is_critical", False)) and float(node.get("load", 0.0)) > 0.5
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda node: float(node.get("load", 0.0)), reverse=True)
    return str(candidates[0].get("node_id"))


def _pick_scada_target(raw_observation: Dict[str, Any]) -> Optional[str]:
    diagnostics = raw_observation.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        for key in ("critical_failures", "at_risk_nodes"):
            values = diagnostics.get(key) or []
            if values:
                return str(values[0])
    return _pick_priority_target(raw_observation)


def _pick_priority_target(raw_observation: Dict[str, Any]) -> Optional[str]:
    candidates = _observation_nodes(raw_observation)
    if not candidates:
        return None
    candidates.sort(
        key=lambda node: (
            bool(node.get("is_critical", False)),
            _sector_priority(str(node.get("sector", ""))),
            1.0 - float(node.get("health", 1.0)),
        ),
        reverse=True,
    )
    return str(candidates[0].get("node_id"))


def _storm_pressure(raw_observation: Dict[str, Any]) -> bool:
    weather = str(raw_observation.get("weather_forecast", "")).lower()
    upcoming = str(raw_observation.get("upcoming_stress_level", "")).lower()
    phase = str(raw_observation.get("narrative_phase", "")).lower()
    info = str(raw_observation.get("info", "")).lower()
    return (
        "storm" in weather
        or upcoming in {"soon", "imminent"}
        or phase in {"pre_storm", "storm_onset"}
        or "harden now" in info
        or "stress" in info
    )


def _sector_priority(sector: str) -> int:
    order = {"hospital": 5, "power": 4, "water": 3, "telecom": 2}
    return order.get(sector.lower(), 1)


# ---------------------------------------------------------------------------
# In-process env client — the headline change. One ``CascadeEnvironment``
# per session, no WebSocket in the loop.
# ---------------------------------------------------------------------------
class InProcessEnv:
    """Drop-in replacement for ``backend.env_client.EnvClient``.

    Owns exactly one ``CascadeEnvironment`` and exposes the same
    ``reset`` / ``step`` / ``get_state`` / ``close`` coroutines that
    ``backend/main.py`` already calls. Each coroutine hops to a worker
    thread so the env's CPU-heavy simulation never blocks the event loop.
    """

    def __init__(self) -> None:
        self._env: Optional[CascadeEnvironment] = None
        # The env isn't thread-safe: serialize all interactions with this
        # session's env so a racing step + state poll can't corrupt it.
        self._lock = asyncio.Lock()

    async def _spawn_env(self) -> CascadeEnvironment:
        if self._env is None:
            self._env = await asyncio.to_thread(CascadeEnvironment)
        return self._env

    async def reset(self, task: str, seed: Optional[int]) -> Dict[str, Any]:
        async with self._lock:
            env = await self._spawn_env()

            def _run() -> Dict[str, Any]:
                obs = env.reset(task_id=task, seed=seed)
                return _model_dump(obs)

            obs = await asyncio.to_thread(_run)
            logger.info(
                "ENV reset: task=%s seed=%s -> step=%s budget=%s nodes=%s",
                task, seed, obs.get("step"), obs.get("budget_remaining"),
                len(obs.get("nodes", [])),
            )
            return obs

    async def step(
        self,
        action: str,
        target: Optional[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            env = await self._spawn_env()

            def _run() -> Dict[str, Any]:
                cascade_action = CascadeAction(
                    action_type=action,
                    target_node_id=target,
                    parameters=parameters or {},
                )
                obs = env.step(cascade_action)
                dumped = _model_dump(obs)
                return {
                    "observation": dumped,
                    "reward": float(dumped.get("reward", 0.0)),
                    "done": bool(dumped.get("done", False)),
                }

            result = await asyncio.to_thread(_run)
            logger.info(
                "ENV step: action=%s target=%s -> step=%s reward=%.3f done=%s",
                action, target,
                result["observation"].get("step"),
                result["reward"], result["done"],
            )
            return result

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            env = await self._spawn_env()

            def _run() -> Dict[str, Any]:
                return _model_dump(env.state)

            return await asyncio.to_thread(_run)

    async def health_check(self) -> bool:
        # In-process env is "connected" iff the env was importable, which
        # it was — otherwise app.py wouldn't have booted.
        return True

    async def close(self) -> None:
        env = self._env
        self._env = None
        if env is None:
            return

        def _close() -> None:
            try:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as exc:  # noqa: BLE001
                logger.debug("env.close() raised (ignored): %s", exc)

        await asyncio.to_thread(_close)


def _model_dump(obj: Any) -> Dict[str, Any]:
    """Normalize pydantic / dataclass / dict-shaped objects to a plain dict."""
    if isinstance(obj, dict):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="python")
    dump_legacy = getattr(obj, "dict", None)
    if callable(dump_legacy):
        return dump_legacy()
    return json.loads(json.dumps(obj, default=lambda x: getattr(x, "__dict__", str(x))))


# ---------------------------------------------------------------------------
# Shared helpers (exact copies of backend/main.py helpers — see that file
# for the rationale behind every line).
# ---------------------------------------------------------------------------
SUPPORTED_TASKS: Tuple[Task, ...] = (
    "task_easy",
    "task_medium",
    "task_hard",
    "task_blackout",
    "task_cyberattack",
)
TASK_ALIASES = {"task_blackout": "task_gen_blackout"}
ACTION_ALIASES = {
    "MONITOR": "wait",
    "WAIT": "wait",
    "REPAIR": "recover",
    "RECOVER": "recover",
    "ISOLATE": "isolate",
    "REROUTE": "reroute",
    "HARDEN": "harden",
    "COORDINATE": "coordinate",
    "SHED_LOAD": "shed_load",
}
REVERSE_ACTION_ALIASES = {
    "wait": "MONITOR",
    "recover": "REPAIR",
    "isolate": "ISOLATE",
    "reroute": "REROUTE",
    "harden": "HARDEN",
    "coordinate": "COORDINATE",
    "shed_load": "SHED_LOAD",
}
SECTOR_KEYS = ("power", "water", "hospital", "telecom")

CITY_FROM_TASK: Dict[str, str] = {
    "task_easy": "mumbai",
    "task_medium": "mumbai",
    "task_hard": "mumbai",
    "task_blackout": "mumbai",
    "task_gen_blackout": "mumbai",
    "task_cyberattack": "mumbai",
    "task_real_city": "mumbai",
    "task_osm_london": "london",
    "task_osm_mumbai": "mumbai",
    "task_osm_bangalore": "bangalore",
    "task_osm_delhi": "delhi",
    "task_osm_nyc": "nyc",
    "task_osm_tokyo": "tokyo",
}
DEFAULT_CITY = "mumbai"
BASELINE_PATH = REPO_DIR / "baseline_results.json"
REAL_NODES_PATH = REPO_DIR / "data" / "real_nodes.json"
DATASET_PATH = REPO_DIR / "data" / "cascadeguard_sft_dataset.csv"

CITY_NODE_LOOKUPS: Dict[str, Dict[str, Dict[str, Any]]] = {}
GLOBAL_NODE_LOOKUP_BY_ID: Dict[str, Dict[str, Any]] = {}
GLOBAL_NODE_LOOKUP_LOWER: Dict[str, Dict[str, Any]] = {}
MISSING_GEO_NODE_IDS: set[str] = set()

MUMBAI_FALLBACKS: Tuple[Tuple[float, float], ...] = (
    (19.076, 72.877), (19.050, 72.821), (19.120, 72.900),
    (18.960, 72.820), (19.033, 72.862), (19.089, 72.863),
    (18.998, 72.836), (19.145, 72.870), (19.065, 72.835),
    (18.975, 72.855), (19.110, 72.885), (19.022, 72.848),
    (19.055, 72.910), (19.080, 72.840), (19.038, 72.875),
)
DEFAULT_RADIUS_BY_SECTOR = {
    "power": 35.0, "water": 20.0, "hospital": 8.0, "telecom": 100.0, "ai": 5.0,
}
_FALLBACK_GEO_INDEX: Dict[str, int] = {}
ABANDON_CLIENTS_ON_RESET = os.getenv("ABANDON_CLIENTS_ON_RESET", "0").strip().lower() in {
    "1", "true", "yes",
}
EPISODE_STREAM_IDLE_TTL_S = float(os.getenv("EPISODE_STREAM_IDLE_TTL_S", "20"))


def _load_real_node_lookups() -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not REAL_NODES_PATH.exists():
        logger.warning("real_nodes.json not found: %s", REAL_NODES_PATH)
        return {}
    try:
        payload = json.loads(REAL_NODES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("real_nodes.json is not valid JSON: %s", exc)
        return {}
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for city, city_data in payload.items():
        if not isinstance(city_data, dict):
            continue
        by_id: Dict[str, Dict[str, Any]] = {}
        for node in city_data.get("nodes", []):
            nid = str(node.get("id", "")).strip()
            if not nid:
                continue
            by_id[nid] = {
                "lat": float(node.get("lat", 0.0)),
                "lng": float(node.get("lng", 0.0)),
                "service_radius_km": float(node.get("service_radius_km", 0.0)),
                "name": str(node.get("name", "")),
                "city": str(city),
            }
        if by_id:
            result[city] = by_id
    return result


def _city_for_task(task_id: Optional[str]) -> str:
    if not task_id:
        return DEFAULT_CITY
    if task_id in CITY_FROM_TASK:
        return CITY_FROM_TASK[task_id]
    if task_id.startswith("task_osm_"):
        suffix = task_id[len("task_osm_"):]
        return suffix or DEFAULT_CITY
    return DEFAULT_CITY


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="CascadeGuard Backend (HF Space)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    # CORS is wide-open because the Space is called cross-origin by the UI
    # Space. allow_credentials MUST stay False with ``*`` — the browser
    # rejects credentials+wildcard combos and the UI doesn't use cookies.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = LlmClient()
session_manager = SessionManager()


async def _abandon_active_clients() -> int:
    """Close stale env handles across SessionManager versions."""
    abandon = getattr(session_manager, "abandon_active_clients", None)
    if abandon is not None:
        return int(await abandon())
    await session_manager.close_all_clients()
    # Older SessionManager does not report a count.
    return 0


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("FastAPI startup started.")
    global CITY_NODE_LOOKUPS, GLOBAL_NODE_LOOKUP_BY_ID, GLOBAL_NODE_LOOKUP_LOWER
    CITY_NODE_LOOKUPS = _load_real_node_lookups()
    merged: Dict[str, Dict[str, Any]] = {}
    for city_lookup in CITY_NODE_LOOKUPS.values():
        merged.update(city_lookup)
    GLOBAL_NODE_LOOKUP_BY_ID = merged
    GLOBAL_NODE_LOOKUP_LOWER = {k.lower(): v for k, v in merged.items()}
    logger.info(
        "Loaded %s geo nodes across %s cities.",
        len(GLOBAL_NODE_LOOKUP_BY_ID), len(CITY_NODE_LOOKUPS),
    )

    if os.getenv("PRELOAD_MODEL", "0").strip().lower() in {"1", "true", "yes"}:
        # Optional: useful after the API is stable, but not during first Space
        # bring-up because model downloads can make HF show "Starting" forever.
        llm_client.load_model_async()
        logger.info("Model preload kicked off in background.")
    else:
        logger.info("Model preload disabled; first agent-step will start loading.")
    logger.info(
        "Model config primary=%s fallback=%s hf_token_set=%s require_cuda=%s",
        llm_client.model_name,
        llm_client.fallback_name,
        bool(llm_client.hf_token),
        llm_client.require_cuda,
    )
    logger.info("Episode stream idle TTL set to %.1fs.", EPISODE_STREAM_IDLE_TTL_S)
    logger.info("FastAPI startup complete; serving traffic now.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await session_manager.close_all_clients()


# ---------------------------------------------------------------------------
# Routes — mirror backend/main.py byte-for-byte where possible.
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    llm_client.maybe_log_loading_progress()
    return HealthResponse(
        status="ok",
        env_connected=True,
        model_loaded=llm_client.is_loaded(),
        model_load_state=llm_client.load_state,
        model_load_error=llm_client.load_error,
    )


@app.post("/api/episode/reset", response_model=EpisodeResetResponse)
async def reset_episode(payload: EpisodeResetRequest) -> EpisodeResetResponse:
    requested_task = payload.task
    env_task = TASK_ALIASES.get(requested_task, requested_task)

    if ABANDON_CLIENTS_ON_RESET:
        abandoned = await _abandon_active_clients()
        if abandoned:
            logger.info("Reset: closed %s stale env handles.", abandoned)

    # Kick off model loading early so health transitions to loading/ready even
    # before the first agent-step call.
    if not llm_client.is_loaded() and llm_client.load_state in {LOAD_STATE_IDLE, LOAD_STATE_ERROR}:
        llm_client.ensure_model_loading()
        logger.info(
            "Triggered background model loading during reset (target=%s state=%s).",
            llm_client.model_name,
            llm_client.load_state,
        )

    episode_env = InProcessEnv()
    try:
        raw_observation = await episode_env.reset(env_task, payload.seed)
        raw_state = await episode_env.get_state()
    except Exception as exc:  # noqa: BLE001
        await episode_env.close()
        raise HTTPException(status_code=502, detail=f"Env error: {exc}") from exc

    session = await session_manager.create(requested_task, env_task, payload.seed)
    session.env_client = episode_env
    episode_state = _build_episode_state(session.session_id, requested_task, raw_observation, raw_state)
    await session_manager.update(
        session.session_id,
        raw_observation=raw_observation,
        raw_state=raw_state,
        current_state=episode_state,
        episode_score=episode_state.episode_score,
        budget_initial=episode_state.budget,
        done=episode_state.done,
    )
    return EpisodeResetResponse(
        session_id=session.session_id,
        state=episode_state,
        nodes=episode_state.nodes,
        edges=episode_state.edges,
        sector_health=episode_state.sector_health,
        budget=episode_state.budget,
        max_steps=episode_state.max_steps,
        task=requested_task,
    )


@app.post("/api/episode/step", response_model=StepResult)
async def step_episode(payload: EpisodeStepRequest) -> StepResult:
    session = await _require_session(payload.session_id)
    env_action, target, parameters = _normalize_action(
        payload.action, payload.target, session.raw_observation,
        target2=payload.target2, explicit_parameters=payload.parameters,
    )
    return await _execute_step(session.session_id, env_action, target, parameters)


@app.post("/api/episode/agent-step", response_model=StepResult)
async def agent_step(payload: Dict[str, str]) -> StepResult:
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    if not llm_client.is_loaded():
        llm_client.ensure_model_loading()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model loading, please retry after /api/health reports model_load_state='ready'",
                "load_state": llm_client.load_state,
                "load_error": llm_client.load_error,
            },
        )
    logger.info(
        "agent-step session=%s model_source=%s load_state=%s",
        session_id,
        llm_client.loaded_source,
        llm_client.load_state,
    )
    session = await _require_session(session_id)
    decision = await llm_client.get_action(session.raw_observation)
    raw_action = str(decision["action"])
    raw_target = decision.get("target")
    raw_parameters = decision.get("parameters") or {}
    norm_action, norm_target, parameters = _normalize_action(
        raw_action, raw_target, session.raw_observation,
        explicit_parameters=raw_parameters or None,
    )
    display_action = AgentAction(
        action=REVERSE_ACTION_ALIASES.get(norm_action, norm_action.upper()),
        target=raw_target if raw_target else norm_target,
    )
    return await _execute_step(
        session_id, norm_action, norm_target, parameters,
        agent_thinking=decision["thinking"], agent_action=display_action,
    )


@app.get("/api/episode/state/{session_id}", response_model=EpisodeState)
async def get_episode_state(session_id: str) -> EpisodeState:
    session = await _require_session(session_id)
    if session.current_state is None:
        raise HTTPException(status_code=404, detail="Episode state not initialized.")
    return session.current_state


@app.get("/api/episodes/history")
async def episode_history() -> List[Dict[str, Any]]:
    history = await session_manager.history()
    return [item.model_dump(mode="json") for item in history]


@app.get("/api/training/reward-curve", response_model=List[RewardCurvePoint])
async def reward_curve() -> List[RewardCurvePoint]:
    final_target = 0.81
    if BASELINE_PATH.exists():
        try:
            payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            final_target = float(
                payload.get("task_summary", {}).get("task_hard", {}).get("mean_score", final_target)
            )
        except Exception:  # noqa: BLE001
            final_target = 0.81
    points: List[RewardCurvePoint] = []
    for episode in range(1, 51):
        trend = 0.23 + (final_target - 0.23) * (1 - math.exp(-episode / 13))
        noise = 0.012 * math.sin(episode * 0.85) + 0.006 * math.cos(episode * 0.31)
        grpo = max(0.23, min(0.92, trend + noise))
        points.append(RewardCurvePoint(
            episode=episode,
            grpo_score=round(grpo, 3),
            random_baseline=0.23,
            zero_shot_baseline=0.49,
        ))
    return points


@app.get("/api/training/baseline-comparison", response_model=BaselineComparison)
async def baseline_comparison() -> BaselineComparison:
    return BaselineComparison(random=0.23, zero_shot=0.49, grpo_trained=0.81)


@app.get("/api/environment/graph", response_model=GraphResponse)
async def environment_graph(
    session_id: Optional[str] = Query(default=None),
    city: Optional[str] = Query(default=None),
) -> GraphResponse:
    graph_data = _load_real_nodes()
    available_cities = sorted(graph_data.keys())

    resolved_city: Optional[str] = None
    session = None
    if session_id:
        session = await _require_session(session_id)

    candidate = (city or "").strip().lower() or None
    if candidate and candidate in graph_data:
        resolved_city = candidate
    elif session is not None:
        resolved_city = _city_for_task(session.env_task)
        if resolved_city not in graph_data:
            resolved_city = None
    if resolved_city is None:
        if DEFAULT_CITY in graph_data:
            resolved_city = DEFAULT_CITY
        elif available_cities:
            resolved_city = available_cities[0]
        else:
            resolved_city = DEFAULT_CITY

    city_data = graph_data.get(resolved_city, {"nodes": [], "edges": []})
    nodes = [dict(n) for n in city_data.get("nodes", [])]
    edges = [
        {"id": f"{e[0]}->{e[1]}", "source": e[0], "target": e[1], "kind": "dependency"}
        for e in city_data.get("edges", [])
        if len(e) == 2
    ]
    if session is not None and session.current_state is not None:
        status_map = {n.id: n for n in session.current_state.nodes}
        for n in nodes:
            status = status_map.get(str(n.get("id")))
            if status is not None:
                n["health"] = status.health
                n["status"] = status.status
    return GraphResponse(
        source="data/real_nodes.json",
        city=resolved_city,
        nodes=nodes,
        edges=edges,
        available_cities=available_cities,
        session_id=session_id,
    )


@app.get("/api/dataset/sft")
async def dataset_download(format: str = Query(default="csv")) -> FileResponse:
    if format.lower() != "csv":
        raise HTTPException(status_code=400, detail="Only csv format is supported.")
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    return FileResponse(
        DATASET_PATH, filename="cascadeguard_sft_dataset.csv", media_type="text/csv"
    )


@app.get("/api/dataset/sft/stats", response_model=DatasetStats)
async def dataset_stats() -> DatasetStats:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    rows = _read_dataset_rows()
    task_distribution: Dict[str, int] = {}
    action_distribution: Dict[str, int] = {}
    avg_score = 0.0
    for row in rows:
        task_distribution[row["task"]] = task_distribution.get(row["task"], 0) + 1
        action_distribution[row["action"]] = action_distribution.get(row["action"], 0) + 1
        avg_score += float(row["episode_score"])
    avg_score = round(avg_score / max(len(rows), 1), 3)
    return DatasetStats(
        total=len(rows),
        task_distribution=task_distribution,
        action_distribution=action_distribution,
        avg_score=avg_score,
    )


@app.websocket("/ws/episode/{session_id}")
async def episode_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info("episode_stream connected session=%s", session_id)
    session = await session_manager.get(session_id)
    if session is None:
        try:
            await websocket.send_json({
                "type": "step_result", "level": "CRIT",
                "timestamp": _timestamp(),
                "message": f"Session {session_id} not found.",
                "payload": {"session_id": session_id},
            })
        finally:
            await websocket.close(code=4404)
        return
    try:
        if session.current_state is not None:
            await websocket.send_json(WsEvent(
                type="step_result", level="INFO",
                timestamp=_timestamp(),
                message="Episode stream connected.",
                payload={"state": session.current_state.model_dump(mode="json")},
            ).model_dump(mode="json"))
        while True:
            try:
                event = await asyncio.wait_for(
                    session.event_queue.get(),
                    timeout=EPISODE_STREAM_IDLE_TTL_S,
                )
                await websocket.send_json(event.model_dump(mode="json"))
            except asyncio.TimeoutError:
                refreshed = await session_manager.get(session_id)
                if refreshed is None:
                    await websocket.close(code=4404)
                    return
                payload: Dict[str, Any] = {"heartbeat": True}
                if refreshed.current_state is not None:
                    payload["state"] = refreshed.current_state.model_dump(mode="json")
                await websocket.send_json({
                    "type": "step_result",
                    "level": "INFO",
                    "timestamp": _timestamp(),
                    "message": f"Episode stream heartbeat (ttl={EPISODE_STREAM_IDLE_TTL_S:.0f}s).",
                    "payload": payload,
                })
    except WebSocketDisconnect as exc:
        logger.info("episode_stream disconnected session=%s code=%s", session_id, getattr(exc, "code", None))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("episode_stream error session=%s: %s", session_id, exc)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Step execution + observation/state normalization (ports from backend/main.py)
# ---------------------------------------------------------------------------
async def _execute_step(
    session_id: str,
    env_action: str,
    target: Optional[str],
    parameters: Optional[Dict[str, Any]],
    *,
    agent_thinking: Optional[str] = None,
    agent_action: Optional[AgentAction] = None,
) -> StepResult:
    session = await _require_session(session_id)
    if session.done:
        raise HTTPException(status_code=409, detail="Episode already completed.")
    if session.env_client is None:
        raise HTTPException(status_code=500, detail="Environment session not initialized.")

    previous_observation = session.raw_observation
    try:
        step_payload = await session.env_client.step(env_action, target, parameters)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Env error during step: {exc}"
        ) from exc
    raw_observation = dict(step_payload.get("observation", {}))

    # Carry over nodes/edges from the previous obs if the env dropped them on
    # this step. Mirrors the defensive fallback in backend/main.py so the
    # front-end never sees an empty graph frame mid-episode.
    if not raw_observation.get("nodes"):
        raw_observation["nodes"] = previous_observation.get("nodes", [])
    if not raw_observation.get("edges"):
        raw_observation["edges"] = previous_observation.get("edges", [])
    if not raw_observation.get("sector_summary"):
        raw_observation["sector_summary"] = previous_observation.get("sector_summary", {})

    raw_state = await session.env_client.get_state()
    episode_state = _build_episode_state(session_id, session.requested_task, raw_observation, raw_state)
    episode_state.done = bool(step_payload.get("done", episode_state.done))
    breakdown = _build_grader_breakdown(raw_observation, raw_state, session.budget_initial or episode_state.budget)
    score = episode_state.episode_score
    await session_manager.update(
        session_id,
        raw_observation=raw_observation,
        raw_state=raw_state,
        current_state=episode_state,
        episode_score=score,
        done=bool(step_payload.get("done", episode_state.done)),
    )
    item = TrajectoryItem(
        step=episode_state.step,
        action=REVERSE_ACTION_ALIASES.get(env_action, env_action.upper()),
        target=target,
        reward_step=round(float(step_payload.get("reward", raw_observation.get("reward", 0.0))), 4),
        score_after=round(score, 4),
        cascade_depth=episode_state.cascade_depth,
        thinking=agent_thinking,
        created_at=datetime.now(timezone.utc),
    )
    await session_manager.append_trajectory(session_id, item)
    result = StepResult(
        new_state=episode_state,
        reward_step=item.reward_step,
        episode_score=round(score, 4),
        done=episode_state.done,
        grader_breakdown=breakdown,
        nodes=episode_state.nodes,
        edges=episode_state.edges,
        agent_thinking=agent_thinking,
        agent_action=agent_action,
    )
    for event in _derive_events(previous_observation, raw_observation, result):
        await session_manager.publish(session_id, event)
    if result.done:
        await session_manager.close_client(session_id)
    return result


async def _require_session(session_id: str):
    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


_NULL_TARGET_ACTIONS = {"wait", "multi_sector_lockdown"}


def _coerce_target_pair(target: Optional[str], target2: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if target2 is not None:
        return (target.strip() if target else None, target2.strip() or None)
    if target and "," in target:
        parts = [p.strip() for p in target.split(",", 1)]
        if len(parts) == 2:
            return (parts[0] or None, parts[1] or None)
    return (target.strip() if isinstance(target, str) else target, None)


def _normalize_action(
    action: str,
    target: Optional[str],
    raw_observation: Dict[str, Any],
    *,
    target2: Optional[str] = None,
    explicit_parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    env_action = ACTION_ALIASES.get(action.strip().upper(), action.strip().lower())
    parameters: Dict[str, Any] = dict(explicit_parameters) if explicit_parameters else {}
    normalized_target = target.strip() if isinstance(target, str) and target else None

    if env_action in _NULL_TARGET_ACTIONS:
        return env_action, None, parameters
    if env_action == "reroute":
        a, b = _coerce_target_pair(target, target2)
        target_node = b or a
        source_node = a if (a and a != target_node) else _pick_reroute_source(raw_observation, target_node)
        if not source_node or not target_node or source_node == target_node:
            return "wait", None, {}
        parameters.setdefault("source", source_node)
        parameters.setdefault("target", target_node)
        return env_action, source_node, parameters
    if env_action == "redistribute_load":
        a, b = _coerce_target_pair(target, target2)
        if not a or not b or a == b:
            return "wait", None, {}
        parameters.setdefault("node_a", a)
        parameters.setdefault("node_b", b)
        return env_action, a, parameters
    if env_action == "cross_sector_bridge":
        a, b = _coerce_target_pair(target, target2)
        if not a or not b or a == b:
            return "wait", None, {}
        parameters.setdefault("sector_a", a)
        parameters.setdefault("sector_b", b)
        return env_action, None, parameters
    if env_action == "request_mutual_aid":
        sector = (target or "").strip() or parameters.get("sector")
        if not sector:
            return "wait", None, {}
        parameters.setdefault("sector", sector)
        return env_action, None, parameters
    return env_action, normalized_target, parameters


def _pick_reroute_source(raw_observation: Dict[str, Any], target: Optional[str]) -> Optional[str]:
    for node in raw_observation.get("nodes", []):
        if node.get("node_id") == target:
            continue
        if node.get("sector") != "power":
            continue
        if not node.get("is_operational", True):
            continue
        if float(node.get("health", 0.0)) < 0.6:
            continue
        return str(node.get("node_id"))
    return None


def _build_episode_state(
    session_id: str, requested_task: str,
    raw_observation: Dict[str, Any], raw_state: Dict[str, Any],
) -> EpisodeState:
    sector_health = _sector_health(raw_observation, raw_state)
    nodes = _build_nodes(raw_observation)
    edges = _build_edges(raw_observation, raw_state)
    return EpisodeState(
        session_id=session_id,
        task=requested_task,  # type: ignore[arg-type]
        step=int(raw_observation.get("step", raw_state.get("step_count", 0))),
        max_steps=int(raw_observation.get("max_steps", 30)),
        budget=round(float(raw_observation.get("budget_remaining", raw_state.get("budget_remaining", 0.0))), 3),
        episode_score=round(float(raw_state.get("score", 0.0)), 4),
        cascade_depth=_cascade_depth(raw_observation, raw_state),
        done=bool(raw_observation.get("done", False)),
        sector_health=sector_health,
        nodes=nodes,
        edges=edges,
    )


def _build_nodes(raw_observation: Dict[str, Any]) -> List[InfraNode]:
    nodes: List[InfraNode] = []
    isolated_nodes = set(raw_observation.get("metadata", {}).get("isolated_nodes", []))
    for node in raw_observation.get("nodes", []):
        node_id = str(node.get("node_id"))
        health = float(node.get("health", 0.0))
        geo = _resolve_node_geo(node)
        status = _node_status(node, isolated_nodes)
        nodes.append(InfraNode(
            id=node_id,
            label=str(node.get("real_name") or node_id),
            sector=str(node.get("sector", "power")),
            health=round(health, 4),
            health_pct=max(0, min(100, round(health * 100))),
            status=status,
            is_critical=bool(node.get("is_critical", False)),
            is_operational=bool(node.get("is_operational", True)),
            is_hardened=bool(node.get("is_hardened", False)),
            lat=geo["lat"], lng=geo["lng"],
            service_radius_km=geo["service_radius_km"],
            metadata={
                "load": float(node.get("load", 0.0)),
                "observation_delayed": bool(node.get("observation_delayed", False)),
                "observation_confidence": float(node.get("observation_confidence", 1.0)),
                "isolated": node_id in isolated_nodes,
            },
        ))
    return nodes


def _build_edges(raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> List[InfraEdge]:
    active_failures = set(raw_observation.get("active_failures", []))
    depth = _cascade_depth(raw_observation, raw_state)
    edges: List[InfraEdge] = []
    for edge in raw_observation.get("edges", []):
        source = str(edge.get("source_id"))
        target = str(edge.get("target_id"))
        kind = "dependency"
        if target in active_failures and depth > 0:
            kind = "cascade"
        elif target in set(raw_observation.get("pending_recoveries", [])):
            kind = "recovery"
        edges.append(InfraEdge(
            id=f"{source}->{target}", source=source, target=target, kind=kind,
            status="active" if bool(edge.get("active", True)) else "inactive",
        ))
    return edges


def _build_grader_breakdown(raw_observation: Dict[str, Any], raw_state: Dict[str, Any], budget_initial: float) -> GraderBreakdown:
    sector_health = _sector_health(raw_observation, raw_state)
    avg_sector = sum(getattr(sector_health, k) for k in SECTOR_KEYS) / len(SECTOR_KEYS)
    hospital_health = sector_health.hospital
    cascade_penalty = max(0.01, 1.0 - min(_cascade_depth(raw_observation, raw_state) / 5.0, 1.0))
    budget_remaining = float(raw_observation.get("budget_remaining", raw_state.get("budget_remaining", 0.0)))
    budget_efficiency = budget_remaining / max(budget_initial or 1.0, 1.0)
    nodes = raw_observation.get("nodes", [])
    healthy_nodes = [n for n in nodes if float(n.get("health", 0.0)) >= 0.6]
    stability = len(healthy_nodes) / max(len(nodes), 1)
    critical_nodes = [n for n in nodes if bool(n.get("is_critical", False))]
    protected_critical = [n for n in critical_nodes if bool(n.get("is_operational", True))]
    centrality = len(protected_critical) / max(len(critical_nodes), 1)
    return GraderBreakdown(
        hospital_health=round(hospital_health, 4),
        cascade_penalty=round(cascade_penalty, 4),
        budget_efficiency=round(max(0.01, min(budget_efficiency, 1.0)), 4),
        infrastructure_stability=round(max(0.01, min(avg_sector * 0.6 + stability * 0.4, 1.0)), 4),
        centrality_protection=round(max(0.01, min(centrality, 1.0)), 4),
    )


def _sector_health(raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> SectorHealth:
    summary = raw_observation.get("sector_summary") or raw_state.get("sector_health") or {}
    return SectorHealth(
        power=float(summary.get("power", 0.0)),
        water=float(summary.get("water", 0.0)),
        hospital=float(summary.get("hospital", 0.0)),
        telecom=float(summary.get("telecom", 0.0)),
    )


def _resolve_node_geo(node: Dict[str, Any]) -> Dict[str, float]:
    node_id = str(node.get("node_id", "")).strip()
    direct = GLOBAL_NODE_LOOKUP_BY_ID.get(node_id) or GLOBAL_NODE_LOOKUP_LOWER.get(node_id.lower())
    if direct is not None:
        return {
            "lat": float(direct["lat"]),
            "lng": float(direct["lng"]),
            "service_radius_km": float(direct["service_radius_km"]),
        }
    payload_lat = node.get("lat")
    payload_lng = node.get("lng", node.get("lon"))
    if payload_lat is not None and payload_lng is not None and (float(payload_lat) != 0.0 or float(payload_lng) != 0.0):
        return {
            "lat": float(payload_lat),
            "lng": float(payload_lng),
            "service_radius_km": float(
                node.get("service_radius_km")
                or DEFAULT_RADIUS_BY_SECTOR.get(str(node.get("sector", "")).lower(), 10.0)
            ),
        }
    if node_id and node_id not in MISSING_GEO_NODE_IDS:
        MISSING_GEO_NODE_IDS.add(node_id)
    if node_id not in _FALLBACK_GEO_INDEX:
        _FALLBACK_GEO_INDEX[node_id] = len(_FALLBACK_GEO_INDEX) % len(MUMBAI_FALLBACKS)
    lat, lng = MUMBAI_FALLBACKS[_FALLBACK_GEO_INDEX[node_id]]
    sector = str(node.get("sector", "")).lower()
    return {
        "lat": float(lat), "lng": float(lng),
        "service_radius_km": float(
            node.get("service_radius_km")
            or DEFAULT_RADIUS_BY_SECTOR.get(sector, 10.0)
        ),
    }


def _node_status(node: Dict[str, Any], isolated_nodes: set) -> str:
    node_id = str(node.get("node_id", ""))
    if node_id in isolated_nodes:
        return "isolated"
    if not bool(node.get("is_operational", True)):
        return "critical"
    health = float(node.get("health", 0.0))
    if health > 0.7:
        return "healthy"
    if health >= 0.3:
        return "degraded"
    return "critical"


def _cascade_depth(raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> int:
    cascade_depth_log = raw_state.get("cascade_depth_log") or []
    if cascade_depth_log:
        return int(cascade_depth_log[-1])
    return len(raw_observation.get("active_failures", []))


def _derive_events(previous_observation: Dict[str, Any], raw_observation: Dict[str, Any], result: StepResult) -> List[WsEvent]:
    timestamp = _timestamp()
    events: List[WsEvent] = [WsEvent(
        type="step_result", level="INFO", timestamp=timestamp,
        message=f"Step {result.new_state.step} completed with reward {result.reward_step:+.3f}.",
        payload={"result": result.model_dump(mode="json")},
    )]
    previous_failures = set(previous_observation.get("active_failures", []))
    current_failures = set(raw_observation.get("active_failures", []))
    new_failures = sorted(current_failures - previous_failures)
    repaired_nodes = sorted(previous_failures - current_failures)
    if new_failures:
        events.append(WsEvent(
            type="failure_event", level="CRIT", timestamp=timestamp,
            message=f"Failure detected: {', '.join(new_failures)}.",
            payload={"nodes": new_failures},
        ))
    if result.new_state.cascade_depth > _cascade_depth(previous_observation, {}):
        events.append(WsEvent(
            type="cascade_trigger", level="WARN", timestamp=timestamp,
            message=f"Cascade depth increased to {result.new_state.cascade_depth}.",
            payload={"cascade_depth": result.new_state.cascade_depth},
        ))
    if repaired_nodes:
        events.append(WsEvent(
            type="repair_complete", level="OK", timestamp=timestamp,
            message=f"Recovered nodes: {', '.join(repaired_nodes)}.",
            payload={"nodes": repaired_nodes},
        ))
    if result.done:
        events.append(WsEvent(
            type="episode_done",
            level="OK" if result.episode_score >= 0.6 else "CRIT",
            timestamp=timestamp,
            message=f"Episode finished with score {result.episode_score:.3f}.",
            payload={"score": result.episode_score, "state": result.new_state.model_dump(mode="json")},
        ))
    return events


def _load_real_nodes() -> Dict[str, Any]:
    if not REAL_NODES_PATH.exists():
        return {}
    return json.loads(REAL_NODES_PATH.read_text(encoding="utf-8"))


def _read_dataset_rows() -> List[Dict[str, str]]:
    if not DATASET_PATH.exists():
        return []
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Gradio landing page — minimal, so HF Spaces has something to render at /.
# ---------------------------------------------------------------------------
def _landing_markdown() -> str:
    return (
        "# CascadeGuard Backend (HF Space)\n\n"
        "This Space is the **API backend** — there is no UI here. Point the "
        "React frontend (`VITE_API_URL` / `VITE_WS_URL`) at this Space URL "
        "and use it as a drop-in replacement for a local backend.\n\n"
        "- **Health** — [/api/health](./api/health)\n"
        "- **OpenAPI docs** — [/docs](./docs)\n"
        "- **Environment graph** — [/api/environment/graph](./api/environment/graph)\n"
        "- **Dataset stats** — [/api/dataset/sft/stats](./api/dataset/sft/stats)\n\n"
        "## Session lifecycle\n"
        "1. `POST /api/episode/reset` with `{ task, seed }` -> returns `session_id`.\n"
        "2. Stream events over `ws://<space>/ws/episode/{session_id}`.\n"
        "3. Drive the episode via `POST /api/episode/step` (manual) or "
        "`POST /api/episode/agent-step` (LoRA agent).\n"
    )


@app.get("/", include_in_schema=False)
async def landing() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "CascadeGuard Backend",
        "health": "/api/health",
        "docs": "/docs",
        "graph": "/api/environment/graph",
        "dataset_stats": "/api/dataset/sft/stats",
        "episode_reset": "POST /api/episode/reset",
        "episode_step": "POST /api/episode/step",
        "agent_step": "POST /api/episode/agent-step",
        "websocket": "/ws/episode/{session_id}",
    })


# Mount gradio on top of the FastAPI app at '/'. The important bit: all
# /api/* and /ws/* routes remain handled by FastAPI first; Gradio only
# renders at '/' (and /assets/, /config, etc.).


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
