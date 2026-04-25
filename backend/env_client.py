from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class EnvClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = (
            base_url
            or os.getenv("OPENENV_SERVER_URL")
            or "https://samarthdave0305-cascade-failure-env.hf.space"
        ).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = await self._client.request(method, path, json=json_body)
        response.raise_for_status()
        return response.json()

    async def reset(self, task: str, seed: Optional[int]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_id": task}
        if seed is not None:
            payload["seed"] = int(seed)
        data = await self._request("POST", "/reset", payload)
        return data.get("observation", data)

    async def step(
        self,
        action: str,
        target: Optional[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action": {
                "action_type": action,
                "target_node_id": target,
                "parameters": parameters or {},
                "metadata": {},
            }
        }
        data = await self._request("POST", "/step", payload)
        if "observation" in data:
            return data
        return {
            "observation": data,
            "reward": data.get("reward", 0.0),
            "done": data.get("done", False),
        }

    async def get_state(self) -> Dict[str, Any]:
        return await self._request("GET", "/state")

    async def health_check(self) -> bool:
        if not self.base_url:
            return False
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
            return True
        except Exception:
            return False
