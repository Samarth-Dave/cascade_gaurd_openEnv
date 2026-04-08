from __future__ import annotations

from typing import Optional

from openenv.core.env_client import EnvClient, StepResult

from cascade_guard.models import CascadeAction, CascadeObservation, CascadeState


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
        return StepResult(
            observation=observation,
            reward=data.get("reward", 0.0),
            done=data.get("done", False)
        )

    #  ADD THIS
    def _parse_state(self, data: dict):
        return CascadeState(**data)

    #  ADD THIS
    def _step_payload(self, action: CascadeAction):
        return action.dict()


__all__ = ["CascadeGuardEnv", "CascadeAction", "CascadeObservation"]