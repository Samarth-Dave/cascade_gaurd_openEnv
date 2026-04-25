from __future__ import annotations

import dataclasses
from typing import Optional

from openenv.core.env_client import EnvClient, StepResult

from .models import CascadeAction, CascadeObservation, CascadeState


class CascadeGuardEnv(EnvClient):
    action_type = CascadeAction
    observation_type = CascadeObservation

    async def reset(
        self,
        task_id: str = "task_easy",
        seed: Optional[int] = None,
        scenario_split: str = "train",
        scenario_index: int = 0,
    ) -> StepResult:
        params = {
            "task_id": task_id,
            "scenario_split": scenario_split,
            "scenario_index": scenario_index,
        }
        if seed is not None:
            params["seed"] = int(seed)
        return await super().reset(params=params)

    async def step(self, action: CascadeAction) -> StepResult:
        return await super().step(action)

    async def state(self):
        return await super().state()

    #  ADD THIS
    def _parse_result(self, data: dict) -> StepResult:
        observation = CascadeObservation(**data["observation"])
        raw_reward = data.get("reward")
        return StepResult(
            observation=observation,
            reward=float(raw_reward) if raw_reward is not None else 0.0,
            done=bool(data.get("done", False))
        )

    #  ADD THIS
    def _parse_state(self, data: dict):
        if dataclasses.is_dataclass(CascadeState):
            valid_fields = {f.name for f in dataclasses.fields(CascadeState)}
        elif hasattr(CascadeState, "model_fields"):
            # Pydantic v2
            valid_fields = set(CascadeState.model_fields.keys())
        else:
            # Pydantic v1 fallback
            valid_fields = set(getattr(CascadeState, "__fields__", {}).keys())

        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return CascadeState(**filtered)

    #  ADD THIS
    def _step_payload(self, action: CascadeAction):
        return action.dict()


__all__ = ["CascadeGuardEnv", "CascadeAction", "CascadeObservation"]