"""
cascade_env_client.py — Type-safe OpenEnv client for CascadeGuard
=================================================================
Wraps openenv.core.env_client.EnvClient with CascadeGuard-specific
action / observation types.

Usage (async)
-------------
    import asyncio
    from cascade_guard.cascade_env_client import CascadeEnvClient
    from cascade_guard.models import CascadeAction

    async def main():
        async with CascadeEnvClient("https://your-space.hf.space") as env:
            obs = await env.reset(task_id="task_hard")
            result = await env.step(CascadeAction(action_type="wait"))
            print(result.reward, result.done)

    asyncio.run(main())

Usage (sync)
------------
    from cascade_guard.cascade_env_client import CascadeEnvClient

    client = CascadeEnvClient("http://localhost:8000")
    obs = client.reset_sync(task_id="task_osm_london")
    obs2 = client.step_sync({"action_type": "wait"})
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from openenv.core.env_client import EnvClient  # type: ignore
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    EnvClient = object  # type: ignore[assignment,misc]

from cascade_guard.models import CascadeAction, CascadeObservation


class CascadeEnvClient(EnvClient if _OPENENV_AVAILABLE else object):  # type: ignore[misc]
    """
    Type-safe OpenEnv client for CascadeGuard.

    Inherits the full async WebSocket + HTTP client from openenv-core when
    available. Also provides lightweight sync helpers (reset_sync / step_sync)
    that call the REST endpoints directly via `requests` for quick scripts.
    """

    action_class      = CascadeAction
    observation_class = CascadeObservation

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        if _OPENENV_AVAILABLE:
            super().__init__(base_url=base_url)

    # ------------------------------------------------------------------
    # Sync REST helpers (no WebSocket needed)
    # ------------------------------------------------------------------

    def _http(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        try:
            import requests as _requests
        except ImportError as exc:
            raise RuntimeError("Install 'requests' to use sync helpers.") from exc

        url  = f"{self.base_url}{path}"
        resp = _requests.request(method, url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def reset_sync(
        self,
        task_id: str = "task_hard",
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous reset — returns raw observation dict."""
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        payload.update(kwargs)
        return self._http("POST", "/reset", json=payload)

    def step_sync(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous step — returns raw observation dict."""
        return self._http("POST", "/step", json=action)

    def state_sync(self) -> Dict[str, Any]:
        """Synchronous state — returns CascadeState dict."""
        return self._http("GET", "/state")

    def graph_sync(self, city: str = "london") -> Dict[str, Any]:
        """Fetch real-world node graph for `city` — for UI / analysis."""
        return self._http("GET", f"/graph?city={city}")

    def cities_sync(self) -> Dict[str, Any]:
        """List available cities with node counts."""
        return self._http("GET", "/cities")

    def tasks_sync(self) -> Dict[str, Any]:
        """List available task IDs with descriptions."""
        return self._http("GET", "/tasks")

    def health_sync(self) -> Dict[str, Any]:
        """Ping the server health endpoint."""
        return self._http("GET", "/health")
