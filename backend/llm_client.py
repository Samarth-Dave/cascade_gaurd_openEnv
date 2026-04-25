from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from training.cot_prompt import build_system_prompt, build_user_prompt  # noqa: E402

# Fields that build_user_prompt iterates over with .items() — must stay dicts.
_DICT_FIELDS_OBS = {"sector_summary"}

# Model loading state machine — exposed via /api/health and used by agent-step
# to return a 503 cleanly when the model is still loading at startup.
LOAD_STATE_IDLE = "idle"
LOAD_STATE_LOADING = "loading"
LOAD_STATE_READY = "ready"
LOAD_STATE_ERROR = "error"

logger = logging.getLogger("cascadeguard.llm")


class LlmClient:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.model_path = (model_path or os.getenv("MODEL_PATH") or "./model").strip()
        self.model_name = (model_name or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-0.5B-Instruct").strip()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None
        self._loaded_source: Optional[str] = None
        self._load_state: str = LOAD_STATE_IDLE
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_model_source(self) -> str:
        local_path = Path(self.model_path)
        if local_path.is_dir():
            return str(local_path)
        return self.model_name

    def load_model(self) -> str:
        """Synchronously load tokenizer + model. Idempotent."""
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                self._load_state = LOAD_STATE_READY
                return self._loaded_source or self._resolve_model_source()
            self._load_state = LOAD_STATE_LOADING
            self._load_error = None

        try:
            source = self._resolve_model_source()
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                source,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model.to(self.device)
            model.eval()

            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            with self._load_lock:
                self._tokenizer = tokenizer
                self._model = model
                self._loaded_source = source
                self._load_state = LOAD_STATE_READY
            logger.info("Model loaded from %s on %s", source, self.device)
            return source
        except Exception as exc:  # noqa: BLE001
            with self._load_lock:
                self._load_state = LOAD_STATE_ERROR
                self._load_error = str(exc)
            logger.exception("Model load failed")
            raise

    def load_model_async(self) -> None:
        """Kick off load in a daemon thread so the FastAPI startup doesn't block."""
        with self._load_lock:
            if self._load_state in (LOAD_STATE_LOADING, LOAD_STATE_READY):
                return
            self._load_state = LOAD_STATE_LOADING
            self._load_error = None

        def _runner() -> None:
            try:
                self.load_model()
            except Exception:  # noqa: BLE001
                # load_model already records the error state.
                pass

        threading.Thread(target=_runner, name="llm-loader", daemon=True).start()

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def load_state(self) -> str:
        return self._load_state

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def get_action(self, raw_observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")

        obs = _build_observation(raw_observation)
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(obs)},
        ]
        content = await asyncio.to_thread(self._generate_response, messages)
        # Truncate to keep log lines compact while preserving the full <action> tag.
        logger.info("Raw model output: %s", content[:200].replace("\n", " | "))
        thinking = _extract_tag(content, "think")
        action_type, target, parameters = _parse_action(content)
        logger.info(
            "Parsed action: type=%s target=%s parameters=%s",
            action_type,
            target,
            parameters,
        )
        return {
            "action": action_type,
            "target": target,
            "parameters": parameters,
            "thinking": thinking,
            "raw_output": content,
        }

    def _generate_response(self, messages: list[dict[str, str]]) -> str:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model is not loaded.")

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\n"

        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **encoded,
                # The output contract is <think>…</think><action>…</action>.
                # 60 tokens was too tight: the 0.5B model spent everything on
                # <think>, ran out of budget mid-tag, and never emitted
                # <action>. The parser then fell through to wait/None which
                # the UI rendered as MONITOR every step. 220 tokens reliably
                # covers a brief reasoning summary + the action tag.
                max_new_tokens=220,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_tokens = encoded["input_ids"].shape[-1]
        output_tokens = generated_ids[0][prompt_tokens:]
        return self._tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _to_dict(obj: Any) -> Any:
    """Recursively convert SimpleNamespace -> plain dict.

    Useful when build_user_prompt expects an object with .items() (e.g.
    sector_summary), which SimpleNamespace does not provide.
    """
    if isinstance(obj, types.SimpleNamespace):
        return {key: _to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_dict(value) for key, value in obj.items()}
    return obj


def _build_observation(raw_observation: Dict[str, Any]) -> SimpleNamespace:
    """Convert raw observation to a namespace shaped for build_user_prompt.

    build_user_prompt expects:
      - top-level + nested fields via attribute access (obs.step, n.health, d.critical_failures)
      - obs.sector_summary as a plain dict (so .items() works)
    """
    obs = _to_namespace(raw_observation)
    for field in _DICT_FIELDS_OBS:
        value = getattr(obs, field, None)
        if isinstance(value, SimpleNamespace):
            setattr(obs, field, _to_dict(value))
        elif value is None:
            setattr(obs, field, {})
    return obs


def _extract_tag(content: str, tag: str) -> str:
    """Extract <tag>…</tag> content. Falls back to an unclosed <tag>… block
    so a CoT block that exceeds max_new_tokens still appears in the UI."""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    open_match = re.search(rf"<{tag}>\s*(.*)", content, re.IGNORECASE | re.DOTALL)
    if open_match:
        chunk = open_match.group(1).strip()
        # Stop at the next opening tag if any (e.g. <action>...).
        next_tag = re.search(r"<\w+>", chunk)
        if next_tag:
            chunk = chunk[: next_tag.start()].strip()
        return chunk
    return ""


# Multi-target action helpers — keep aligned with backend/main.py and
# server/cascade_environment.py so the env never receives a malformed action.
_TWO_NODE_ACTIONS = {"reroute", "redistribute_load"}
_TWO_SECTOR_ACTIONS = {"cross_sector_bridge"}
_SECTOR_ACTIONS = {"request_mutual_aid"}


def _parse_action(content: str) -> tuple[str, Optional[str], Dict[str, Any]]:
    """Extract (action_type, primary_target, parameters) from an LLM response.

    Falls back to wait/null on any parse failure so a malformed model output
    can never crash the step loop.
    """
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
        else:  # redistribute_load
            parameters = {"node_a": parts[0], "node_b": parts[1]}
        primary = parts[0]
    elif action_type in _TWO_SECTOR_ACTIONS and len(parts) >= 2:
        parameters = {"sector_a": parts[0], "sector_b": parts[1]}
        primary = parts[0]
    elif action_type in _SECTOR_ACTIONS:
        parameters = {"sector": parts[0]}
        primary = parts[0]

    return action_type, primary, parameters
