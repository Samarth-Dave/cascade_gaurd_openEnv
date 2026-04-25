"""WebSocket client for the OpenEnv CascadeGuard env server.

Why one process-wide WebSocket instead of one per episode?
----------------------------------------------------------
The HF Space runs ``create_app(...)`` with ``max_concurrent_envs=1``. Each
WebSocket connection holds a dedicated env *session slot* on the server.
If episode 1 opens a socket and episode 2 then opens *another* socket while
episode 1 is still alive, episode 2 is rejected at the handshake with
``ConnectionClosedOK 1000``.

The HTTP ``/reset`` and ``/step`` endpoints are unusable for this env
because they instantiate a fresh ``CascadeEnvironment()`` per request and
discard it — every step then returns ``nodes=[]``, ``budget=0``,
``step=1``, ``max_steps=10`` (the ``__init__`` defaults), so failures never
appear and the cascade never propagates.

Solution: keep **one** WebSocket alive for the entire backend process.
Starting a new episode sends ``{"type": "reset"}`` over the existing
socket, which calls ``env.reset()`` on the same ``CascadeEnvironment``
instance. State is preserved between steps, and only one server slot is
ever used.

Public API (contract unchanged for main.py):
  reset(task, seed)         → flat observation dict
  step(action, target, ...) → {"observation": {...}, "reward": float, "done": bool}
  get_state()               → state dict
  health_check()            → bool   (HTTP /health)
  close()                   → no-op for per-session instances; the global
                               socket stays open for the next episode

Module-level helpers:
  ensure_global_socket()    — open the singleton socket (call on startup)
  close_global_socket()     — gracefully close it (call on shutdown)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import httpx

try:
    import websockets
    from websockets.asyncio.client import ClientConnection, connect as ws_connect
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "websockets is required. Install with `pip install websockets`."
    ) from exc

logger = logging.getLogger("cascadeguard.env")

# ---------------------------------------------------------------------------
# Process-wide singletons. Created lazily on the first connect attempt.
# ---------------------------------------------------------------------------
_GLOBAL_WS: Optional[ClientConnection] = None
_GLOBAL_LOCK: asyncio.Lock | None = None
_GLOBAL_WS_URL: Optional[str] = None
_GLOBAL_TIMEOUT: float = 60.0
_MAX_WS_SIZE = 8 * 1024 * 1024


def _get_lock() -> asyncio.Lock:
    global _GLOBAL_LOCK
    if _GLOBAL_LOCK is None:
        _GLOBAL_LOCK = asyncio.Lock()
    return _GLOBAL_LOCK


def _http_to_ws(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme in ("ws", "wss"):
        return base_url
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


async def _open_socket(ws_url: str, timeout: float) -> ClientConnection:
    logger.info("Opening global WebSocket session to %s", ws_url)
    return await ws_connect(
        ws_url,
        ping_interval=20,
        ping_timeout=20,
        max_size=_MAX_WS_SIZE,
        open_timeout=timeout,
    )


async def ensure_global_socket(base_url: Optional[str] = None, timeout: float = 60.0) -> None:
    """Open the singleton WebSocket if it's not already alive.

    Safe to call from FastAPI's startup hook. Errors are logged but not
    raised — a missing socket is reopened lazily on the next request.
    """
    global _GLOBAL_WS, _GLOBAL_WS_URL, _GLOBAL_TIMEOUT
    base = (
        base_url
        or os.getenv("OPENENV_SERVER_URL")
        or "https://samarthdave0305-cascade-failure-env.hf.space"
    ).rstrip("/")
    _GLOBAL_WS_URL = _http_to_ws(base) + "/ws"
    _GLOBAL_TIMEOUT = timeout
    if _GLOBAL_WS is not None:
        return
    try:
        _GLOBAL_WS = await _open_socket(_GLOBAL_WS_URL, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Initial WS connect failed: %s — will retry on first request", exc)


async def close_global_socket() -> None:
    """Send the openenv close message and close the singleton WebSocket."""
    global _GLOBAL_WS
    ws = _GLOBAL_WS
    _GLOBAL_WS = None
    if ws is None:
        return
    try:
        await ws.send(json.dumps({"type": "close"}))
    except Exception:  # noqa: BLE001
        pass
    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass


class EnvClient:
    """Thin per-session wrapper around the process-wide singleton WS.

    ``reset()`` sends a reset message over the existing socket — it does
    NOT open a new connection — so a second episode never trips the
    server's ``max_concurrent_envs=1`` limit.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0) -> None:
        self.base_url = (
            base_url
            or os.getenv("OPENENV_SERVER_URL")
            or "https://samarthdave0305-cascade-failure-env.hf.space"
        ).rstrip("/")
        global _GLOBAL_WS_URL, _GLOBAL_TIMEOUT
        if _GLOBAL_WS_URL is None:
            _GLOBAL_WS_URL = _http_to_ws(self.base_url) + "/ws"
        _GLOBAL_TIMEOUT = max(_GLOBAL_TIMEOUT, timeout)
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10.0,
            headers={"Accept": "application/json"},
        )

    # ------------------------------------------------------------------
    # Internal: send/recv on the global socket
    # ------------------------------------------------------------------

    async def _ensure_socket(self) -> ClientConnection:
        global _GLOBAL_WS, _GLOBAL_WS_URL
        if _GLOBAL_WS is not None:
            return _GLOBAL_WS
        if _GLOBAL_WS_URL is None:
            _GLOBAL_WS_URL = _http_to_ws(self.base_url) + "/ws"
        _GLOBAL_WS = await _open_socket(_GLOBAL_WS_URL, self._timeout)
        return _GLOBAL_WS

    async def _send_and_recv(
        self, message: Dict[str, Any], *, retry_on_drop: bool = True
    ) -> Dict[str, Any]:
        """Send one message and return its paired response.

        On a dead-socket exception we drop the singleton and (optionally)
        retry once with a fresh connection. This handles the case where
        the HF Space proxy has idled the socket between user clicks.
        """
        global _GLOBAL_WS

        async def _attempt() -> Dict[str, Any]:
            global _GLOBAL_WS
            async with _get_lock():
                ws = await self._ensure_socket()
                try:
                    await ws.send(json.dumps(message))
                    raw = await asyncio.wait_for(ws.recv(), timeout=self._timeout)
                except Exception:
                    _GLOBAL_WS = None
                    try:
                        await ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                    raise
            return json.loads(raw)

        try:
            response = await _attempt()
        except Exception as exc:
            if not retry_on_drop:
                raise
            logger.warning(
                "WS %s failed (%s) — reconnecting and retrying once",
                message.get("type"), exc,
            )
            response = await _attempt()

        msg_type = response.get("type")
        if msg_type == "error":
            raise RuntimeError(f"Env server error: {response.get('data')}")
        if msg_type not in ("observation", "state"):
            raise RuntimeError(f"Unexpected WS response type {msg_type!r}: {response}")
        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reset(self, task: str, seed: Optional[int]) -> Dict[str, Any]:
        """Reset the episode. Returns the **flat observation dict**.

        Sends ``{"type": "reset"}`` on the existing socket so the env's
        ``reset()`` runs on the same ``CascadeEnvironment`` instance.
        """
        data: Dict[str, Any] = {"task_id": task}
        if seed is not None:
            data["seed"] = int(seed)
        response = await self._send_and_recv({"type": "reset", "data": data})
        envelope = response.get("data") or {}
        obs: Dict[str, Any] = envelope.get("observation") or envelope
        logger.info(
            "ENV reset: task=%s seed=%s -> step=%s budget=%s max_steps=%s nodes=%s",
            task, seed,
            obs.get("step"), obs.get("budget_remaining"),
            obs.get("max_steps"), len(obs.get("nodes", [])),
        )
        return obs

    async def step(
        self,
        action: str,
        target: Optional[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one action. Returns ``{"observation": {...}, "reward": float, "done": bool}``."""
        response = await self._send_and_recv({
            "type": "step",
            "data": {
                "action_type": action,
                "target_node_id": target,
                "parameters": parameters or {},
                "metadata": {},
            },
        })
        envelope = response.get("data") or {}
        obs: Dict[str, Any] = envelope.get("observation") or {}
        reward_raw = envelope.get("reward")
        reward = float(reward_raw) if reward_raw is not None else float(obs.get("reward", 0.0))
        done = bool(envelope.get("done", obs.get("done", False)))
        logger.info(
            "ENV step: action=%s target=%s -> step=%s budget=%s reward=%s done=%s nodes=%s",
            action, target,
            obs.get("step"), obs.get("budget_remaining"),
            reward, done, len(obs.get("nodes", [])),
        )
        return {"observation": obs, "reward": reward, "done": done}

    async def get_state(self) -> Dict[str, Any]:
        """Fetch the env's CascadeState dict via the same shared socket."""
        try:
            response = await self._send_and_recv({"type": "state"})
            return response.get("data") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_state failed: %s — returning empty dict", exc)
            return {}

    async def health_check(self) -> bool:
        if not self.base_url:
            return False
        try:
            response = await self._http.get("/health")
            response.raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False

    async def close(self) -> None:
        """Per-session close is a no-op for the global socket.

        The HTTP client (used only by health_check) is closed here.
        """
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001
            pass
