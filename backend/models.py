from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Task = Literal[
    "task_easy",
    "task_medium",
    "task_hard",
    "task_blackout",
    "task_cyberattack",
]

NodeStatus = Literal["healthy", "degraded", "critical", "isolated"]
EpisodeOutcome = Literal["CONTAINED", "FAILED", "IN_PROGRESS"]
WsEventType = Literal[
    "step_result",
    "failure_event",
    "cascade_trigger",
    "repair_complete",
    "episode_done",
]
WsLevel = Literal["INFO", "WARN", "CRIT", "OK"]


class HealthResponse(BaseModel):
    status: str
    env_connected: bool
    model_loaded: bool
    # Lifecycle: "idle" | "loading" | "ready" | "error".
    model_load_state: str = "idle"
    model_load_error: Optional[str] = None


class SectorHealth(BaseModel):
    power: float = 0.0
    water: float = 0.0
    hospital: float = 0.0
    telecom: float = 0.0


class InfraNode(BaseModel):
    id: str
    label: str
    sector: str
    health: float = 0.0
    health_pct: int = 0
    status: NodeStatus = "healthy"
    is_critical: bool = False
    is_operational: bool = True
    is_hardened: bool = False
    lat: float = 0.0
    lng: float = 0.0
    service_radius_km: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InfraEdge(BaseModel):
    id: str
    source: str
    target: str
    kind: Literal["dependency", "cascade", "recovery"] = "dependency"
    status: Literal["active", "inactive"] = "active"


class GraderBreakdown(BaseModel):
    hospital_health: float
    cascade_penalty: float
    budget_efficiency: float
    infrastructure_stability: float
    centrality_protection: float


class EpisodeState(BaseModel):
    session_id: str
    task: Task
    step: int = 0
    max_steps: int = 30
    budget: float = 0.0
    episode_score: float = 0.0
    cascade_depth: int = 0
    done: bool = False
    sector_health: SectorHealth = Field(default_factory=SectorHealth)
    nodes: List[InfraNode] = Field(default_factory=list)
    edges: List[InfraEdge] = Field(default_factory=list)


class EpisodeResetRequest(BaseModel):
    task: Task
    seed: Optional[int] = 42


class EpisodeResetResponse(BaseModel):
    session_id: str
    state: EpisodeState
    nodes: List[InfraNode]
    edges: List[InfraEdge]
    sector_health: SectorHealth
    budget: float
    max_steps: int
    task: Task


class EpisodeStepRequest(BaseModel):
    session_id: str
    action: str
    target: Optional[str] = None
    # Optional second target for multi-target actions:
    #   reroute(source, target)
    #   cross_sector_bridge(sector_a, sector_b)
    #   redistribute_load(node_a, node_b)
    target2: Optional[str] = None
    # Optional explicit parameters dict — overrides target/target2 derivation
    # when callers want full control over the action payload.
    parameters: Optional[Dict[str, Any]] = None


class AgentAction(BaseModel):
    action: str
    target: Optional[str] = None


class TrajectoryItem(BaseModel):
    step: int
    action: str
    target: Optional[str] = None
    reward_step: float
    score_after: float
    cascade_depth: int
    thinking: Optional[str] = None
    created_at: datetime


class StepResult(BaseModel):
    new_state: EpisodeState
    reward_step: float
    episode_score: float
    done: bool
    grader_breakdown: GraderBreakdown
    nodes: List[InfraNode]
    edges: List[InfraEdge]
    agent_thinking: Optional[str] = None
    agent_action: Optional[AgentAction] = None


class EpisodeHistoryItem(BaseModel):
    session_id: str
    task: Task
    score: float
    budget_used: float
    steps: int
    cascade_peak: int
    outcome: EpisodeOutcome
    started_at: datetime
    updated_at: datetime
    trajectory: List[TrajectoryItem] = Field(default_factory=list)


class RewardCurvePoint(BaseModel):
    episode: int
    grpo_score: float
    random_baseline: float
    zero_shot_baseline: float


class BaselineComparison(BaseModel):
    random: float
    zero_shot: float
    grpo_trained: float


class DatasetStats(BaseModel):
    total: int
    task_distribution: Dict[str, int]
    action_distribution: Dict[str, int]
    avg_score: float


class GraphResponse(BaseModel):
    source: str
    city: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    available_cities: List[str]
    session_id: Optional[str] = None


class WsEvent(BaseModel):
    type: WsEventType
    level: WsLevel
    timestamp: str
    message: str
    payload: Dict[str, Any] = Field(default_factory=dict)
