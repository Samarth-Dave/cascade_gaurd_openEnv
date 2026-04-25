from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from models import AgentAction, EpisodeHistoryItem, EpisodeOutcome, EpisodeState, TrajectoryItem, WsEvent


@dataclass
class SessionRecord:
    session_id: str
    requested_task: str
    env_task: str
    seed: Optional[int]
    created_at: datetime
    updated_at: datetime
    budget_initial: float = 0.0
    episode_score: float = 0.0
    done: bool = False
    raw_observation: Dict[str, Any] = field(default_factory=dict)
    raw_state: Dict[str, Any] = field(default_factory=dict)
    current_state: Optional[EpisodeState] = None
    trajectory: List[TrajectoryItem] = field(default_factory=list)
    event_queue: asyncio.Queue[WsEvent] = field(default_factory=asyncio.Queue)

    def to_history_item(self) -> EpisodeHistoryItem:
        current = self.current_state or EpisodeState(
            session_id=self.session_id,
            task=self.requested_task,  # type: ignore[arg-type]
        )
        budget_used = max(self.budget_initial - current.budget, 0.0)
        cascade_peak = max((item.cascade_depth for item in self.trajectory), default=current.cascade_depth)
        outcome: EpisodeOutcome
        if not self.done:
            outcome = "IN_PROGRESS"
        elif self.episode_score >= 0.6:
            outcome = "CONTAINED"
        else:
            outcome = "FAILED"
        return EpisodeHistoryItem(
            session_id=self.session_id,
            task=self.requested_task,  # type: ignore[arg-type]
            score=round(self.episode_score, 4),
            budget_used=round(budget_used, 3),
            steps=current.step,
            cascade_peak=cascade_peak,
            outcome=outcome,
            started_at=self.created_at,
            updated_at=self.updated_at,
            trajectory=self.trajectory,
        )


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._completed_order: List[str] = []
        self._lock = asyncio.Lock()

    async def create(self, requested_task: str, env_task: str, seed: Optional[int]) -> SessionRecord:
        async with self._lock:
            now = datetime.now(timezone.utc)
            session = SessionRecord(
                session_id=str(uuid4()),
                requested_task=requested_task,
                env_task=env_task,
                seed=seed,
                created_at=now,
                updated_at=now,
            )
            self._sessions[session.session_id] = session
            return session

    async def get(self, session_id: str) -> Optional[SessionRecord]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def update(
        self,
        session_id: str,
        *,
        raw_observation: Optional[Dict[str, Any]] = None,
        raw_state: Optional[Dict[str, Any]] = None,
        current_state: Optional[EpisodeState] = None,
        episode_score: Optional[float] = None,
        budget_initial: Optional[float] = None,
        done: Optional[bool] = None,
    ) -> SessionRecord:
        async with self._lock:
            session = self._sessions[session_id]
            if raw_observation is not None:
                session.raw_observation = raw_observation
            if raw_state is not None:
                session.raw_state = raw_state
            if current_state is not None:
                session.current_state = current_state
            if episode_score is not None:
                session.episode_score = episode_score
            if budget_initial is not None and session.budget_initial == 0.0:
                session.budget_initial = budget_initial
            if done is not None:
                session.done = done
                if done and session_id not in self._completed_order:
                    self._completed_order.insert(0, session_id)
            session.updated_at = datetime.now(timezone.utc)
            return session

    async def append_trajectory(self, session_id: str, item: TrajectoryItem) -> SessionRecord:
        async with self._lock:
            session = self._sessions[session_id]
            session.trajectory.append(item)
            session.updated_at = datetime.now(timezone.utc)
            return session

    async def publish(self, session_id: str, event: WsEvent) -> None:
        session = await self.get(session_id)
        if session is None:
            return
        await session.event_queue.put(event)

    async def history(self) -> List[EpisodeHistoryItem]:
        async with self._lock:
            return [self._sessions[session_id].to_history_item() for session_id in self._completed_order]
