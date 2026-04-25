from __future__ import annotations

import asyncio
import os
import re
import sys
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

    def _resolve_model_source(self) -> str:
        local_path = Path(self.model_path)
        if local_path.is_dir():
            return str(local_path)
        return self.model_name

    def load_model(self) -> str:
        if self._model is not None and self._tokenizer is not None:
            return self._loaded_source or self._resolve_model_source()

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

        self._tokenizer = tokenizer
        self._model = model
        self._loaded_source = source
        return source

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def get_action(self, raw_observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")

        obs = _build_observation(raw_observation)
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(obs)},
        ]
        content = await asyncio.to_thread(self._generate_response, messages)
        thinking = _extract_tag(content, "think")
        action_type, target = _parse_action(content)
        return {
            "action": action_type,
            "target": target,
            "thinking": thinking,
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
                max_new_tokens=512,
                do_sample=False,
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
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", content, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_action(content: str) -> tuple[str, Optional[str]]:
    match = re.search(r"<action>\s*([a-zA-Z_]+)\(([^)]*)\)\s*</action>", content, re.IGNORECASE)
    if not match:
        return "wait", None

    action_type = match.group(1).lower()
    raw_target = match.group(2).strip()
    if not raw_target or raw_target.lower() in {"null", "none"}:
        return action_type, None
    first_target = raw_target.split(",")[0].strip()
    return action_type, first_target or None
