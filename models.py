from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from openenv.core.env_server.types import Action, Observation, State


class NodeStatus(BaseModel):
    node_id: str
    sector: str  # one of: power, water, hospital, telecom
    health: float  # 0.0 to 1.0
    load: float  # 0.0 to 1.0
    is_operational: bool
    is_hardened: bool
    observation_delayed: bool
    is_critical: bool
    # Geo fields (optional — populated when task has coordinates)
    lat: float = 0.0              # Latitude in decimal degrees
    lon: float = 0.0              # Longitude in decimal degrees
    service_radius_km: float = 0.0  # Service coverage radius in km
    observation_confidence: float = 1.0  # 0.0–1.0; drops under telecom outage
    coverage_quality: float = 1.0  # How well this node is backed up by neighbours


class EdgeStatus(BaseModel):
    source_id: str
    target_id: str
    active: bool


class ObservationDiagnostics(BaseModel):
    """Optional diagnostic-only features for analysis and policy debugging."""

    critical_failures: List[str] = []
    at_risk_nodes: List[str] = []
    dependency_alerts: List[str] = []
    recommended_recovery_order: List[str] = []
    system_pressure: float = 0.0
    leakage_safe: bool = True


class CascadeObservation(Observation):
    """OpenEnv Observation for the CascadeGuard environment.

    Inherits `done`, `reward`, and `metadata` from the base Observation.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    step: int = 0
    max_steps: int = 10
    budget_remaining: float = 0.0
    weather_forecast: str = "clear"  # clear | storm_warning | extreme_storm
    nodes: List[NodeStatus] = []
    edges: List[EdgeStatus] = []
    active_failures: List[str] = []
    pending_recoveries: List[str] = []
    sector_summary: Dict[str, float] = {}
    diagnostics: Optional[ObservationDiagnostics] = None
    task_id: str = ""
    info: str = ""
    # Geo / coverage fields
    radius_coverage_events: List[str] = []  # human-readable radius-coverage protections this step
    narrative_phase: str = "pre_storm"  # pre_storm | storm_onset | crisis_peak | recovery | stabilization
    consecutive_waits: int = 0  # how many consecutive wait actions the agent has taken
    upcoming_stress_level: str = "none"  # none | soon | imminent


class CascadeAction(Action):
    """OpenEnv Action for the CascadeGuard environment.

    Inherits `metadata` from the base Action.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    action_type: str  # harden | shed_load | coordinate | recover | isolate | wait
    target_node_id: Optional[str] = None
    parameters: Dict[str, Any] = {}

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        allowed = {"harden", "shed_load", "coordinate", "recover", "isolate", "wait"}
        if v not in allowed:
            raise ValueError(
                f"action_type must be one of {allowed}, got {v!r}"
            )
        return v


class CascadeState(State):
    """OpenEnv State for the CascadeGuard environment.

    Inherits `episode_id` and `step_count` from the base State.
    """

    task_id: str = ""
    budget_remaining: float = 0.0
    sector_health: Dict[str, float] = {}
    score: float = 0.5                      # episode score [0, 1], computed by grader
    total_nodes: int = 0                    # actual node count for grader denominator
    # Included for grader access when running via Docker (not available in obs)
    failure_history: List[List[str]] = []       # per-step list of failed node IDs
    hospital_health_log: List[float] = []       # per-step min hospital health
    cascade_depth_log: List[int] = []           # per-step cascade depth proxy
    dependency_order_log: List[int] = []        # 1 correct recover, 0 wrong recover
    action_history: List[str] = []              # per-step action type history


# OpenEnv expected names (aliases)
CascadeGuardAction = CascadeAction
CascadeGuardObservation = CascadeObservation
