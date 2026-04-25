from __future__ import annotations

import copy
import json
import math
import random
import uuid
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server import Environment

from models import (
    CascadeAction,
    CascadeObservation,
    CascadeState,
    EdgeStatus,
    NodeStatus,
    ObservationDiagnostics,
)
from tasks import TASK_CONFIGS, materialize_task_config, resolve_seed
from server.graders import grade
try:
    from geo_utils import (
        compute_coverage_matrix,
        compute_cascade_mitigation,
        compute_obs_confidence,
        coverage_events_summary,
        DEFAULT_NODE_COORDS,
    )
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HARDEN_COST: float = 2.0
COORDINATE_COST: float = 1.0
FAILURE_THRESHOLD_UNHARDENED: float = 0.3
FAILURE_THRESHOLD_HARDENED: float = 0.1
RECOVERY_STEPS: int = 2
RECOVERY_HEALTH: float = 0.9
CASCADE_DAMAGE: float = 0.18
WAIT_BASE_PENALTY: float = 0.04
WAIT_UNDER_PRESSURE_PENALTY: float = 0.22
CRITICAL_SHED_LOAD_PENALTY: float = 1.2
DEPENDENCY_RECOVERY_BONUS: float = 0.15
CASCADE_IMPACT_PENALTY: float = 0.12
WEATHER_DAMAGE: Dict[str, float] = {
    "clear": 0.0,
    "storm_warning": 0.05,
    "extreme_storm": 0.12,
}
STEP_REWARD_MIN: float = -1.0
STEP_REWARD_MAX: float = 1.0
STEP_REWARD_EPS: float = 1e-4


class _NodeState:
    """Internal mutable node state (server-side ground truth)."""

    def __init__(self, node_id: str, sector: str, is_critical: bool, real_name: str = "") -> None:
        self.node_id = node_id
        self.sector = sector
        self.is_critical = is_critical
        self.real_name: str = real_name   # e.g. "KEM Hospital Parel", "BT Tower"
        self.health: float = 1.0
        self.load: float = 0.5
        self.is_operational: bool = True
        self.is_hardened: bool = False
        self.in_recovery: bool = False

    def copy(self) -> "_NodeState":
        n = _NodeState(self.node_id, self.sector, self.is_critical, self.real_name)
        n.health = self.health
        n.load = self.load
        n.is_operational = self.is_operational
        n.is_hardened = self.is_hardened
        n.in_recovery = self.in_recovery
        return n


class CascadeEnvironment(Environment):
    """CascadeGuard OpenEnv environment — cross-sector infrastructure RL.

    v3 + v2 upgrades:
    - Geo-aware radius coverage physics (cascade damage mitigated by backup nodes within radius)
    - Dynamic partial observability via telecom coverage confidence
    - Narrative phase tracker (pre_storm → storm_onset → crisis_peak → recovery → stabilization)
    - Structured invalid action reasons in metadata
    - Progressive wait penalty: scales with consecutive_waits counter
    - Cascade debt pressure: unresolved failures accumulate extra downstream pressure
    - Observation confidence: health readings include noise when telecom coverage is weak
    """

    def __init__(self) -> None:
        self._task_id: str = "task_easy"
        self._step: int = 0
        self._node_states: Dict[str, _NodeState] = {}
        self._edge_states: Dict[str, bool] = {}
        self._budget_total: float = 0.0
        self._budget: float = 0.0
        self._pending_recoveries: Dict[str, int] = {}
        self._sector_delays: Dict[str, int] = {}
        self._failure_history: List[Set[str]] = []
        self._hospital_health_log: List[float] = []
        self._rng: random.Random = random.Random()
        self._stress_schedule: Dict[int, dict] = {}
        self._episode_id: str = str(uuid.uuid4())
        self._weather_forecast: str = "clear"
        self._max_steps: int = 10
        self._config: dict = {}
        self._done: bool = False
        self._last_obs: Optional[CascadeObservation] = None
        self._zero_sector_streak: int = 0
        self._scada_start: Optional[int] = None
        self._scada_event: Optional[dict] = None
        self._delayed_sector_active: Dict[str, bool] = {}
        self._coord_clear: Dict[str, int] = {}
        self._partial_obs_nodes: Set[str] = set()
        self._obs_buffer: Dict[str, List[tuple]] = {}
        self._cascade_depth_log: List[int] = []
        self._dependency_order_log: List[int] = []
        self._action_history: List[str] = []
        self._reward_mode: str = "clamped"
        self._training_mode: bool = False
        self._reward_noise_scale: float = 0.0
        self._isolated_nodes: Set[str] = set()
        self._isolation_expiry: Dict[str, int] = {}
        self._last_reward_components: Dict[str, float] = {}
        # v2: new state tracking
        self._consecutive_waits: int = 0
        self._narrative_phase: str = "pre_storm"
        self._unresolved_failure_steps: Dict[str, int] = {}  # node_id -> steps since failure
        self._coverage_matrix: Dict[str, list] = {}  # geo coverage, populated on reset
        self._node_coords: Dict[str, tuple] = {}     # {node_id: (lat, lon, radius_km)}
        self._radius_protection_log: List[float] = []  # per-step mitigation fraction
        # v2.2: expanded action space state
        self._prioritize_cooldown: int = 0          # counts down from 3
        self._mutual_aid_used: bool = False
        self._lockdown_remaining: int = 0
        self._active_bridges: Dict[tuple, int] = {}  # {(sector_a, sector_b): ttl}
        self._fast_repair_nodes: Set[str] = set()
        self._scada_patched: Set[str] = set()        # node_ids with SCADA cleared
        self._reroute_targets: Dict[str, int] = {}   # {node_id: steps_remaining}
        self._prioritize_immune: Set[str] = set()    # 1-step stress-immunity set
        self._load_reduce_nodes: Dict[str, int] = {} # {node_id: steps_remaining} -0.15 fp mod
        self._neighbor_stability: Dict[str, int] = {} # {node_id: steps_remaining} +0.15
        self._sacrifice_nodes_log: List[str] = []    # controlled_cascade sacrifice IDs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> CascadeObservation:  # type: ignore[override]
        """BUG 1 FIX: accepts seed, episode_id, task_id, **kwargs so the
        openenv framework can pass task_id without a TypeError."""
        # Handle OpenEnv reset payload styles, including params={"task_id": ...}.
        if task_id is None and "task_id" in kwargs:
            task_id = kwargs.get("task_id")
        if task_id is None:
            params = kwargs.get("params")
            if isinstance(params, dict):
                task_id = params.get("task_id")

        if task_id is not None:
            if task_id not in TASK_CONFIGS:
                raise ValueError(
                    f"Unknown task_id {task_id!r}. Valid options: {list(TASK_CONFIGS)}"
                )
            self._task_id = task_id

        params = kwargs.get("params") if isinstance(kwargs.get("params"), dict) else {}
        if seed is None and "seed" in params:
            try:
                seed = int(params.get("seed"))
            except (TypeError, ValueError):
                seed = None
        scenario_split = str(params.get("scenario_split", kwargs.get("scenario_split", "train")))
        scenario_index = int(params.get("scenario_index", kwargs.get("scenario_index", 0)))

        reward_mode = kwargs.get("reward_mode", "clamped")
        if isinstance(params, dict):
            reward_mode = params.get("reward_mode", reward_mode)
        self._reward_mode = str(reward_mode)
        training_mode = kwargs.get("training_mode", False)
        if isinstance(params, dict):
            training_mode = params.get("training_mode", training_mode)
        self._training_mode = bool(training_mode)
        self._reward_noise_scale = 0.05 if self._reward_mode == "grpo" else 0.0
        resolved_seed = seed if seed is not None else resolve_seed(
            self._task_id,
            split=scenario_split,
            scenario_index=scenario_index,
            fallback_seed=int(TASK_CONFIGS[self._task_id]["seed"]),
        )

        cfg = materialize_task_config(self._task_id, seed=int(resolved_seed), split=scenario_split)
        self._config = cfg
        self._max_steps = cfg["max_steps"]
        self._budget_total = cfg["budget"]
        self._budget = cfg["budget"]
        self._step = 0
        self._done = False
        self._weather_forecast = "clear"
        self._pending_recoveries = {}
        self._failure_history = []
        self._hospital_health_log = []
        self._cascade_depth_log = []
        self._dependency_order_log = []
        self._action_history = []
        self._isolated_nodes = set()
        self._isolation_expiry = {}
        self._last_reward_components = {}
        self._zero_sector_streak = 0
        self._episode_id = episode_id or str(uuid.uuid4())
        # v2: reset new counters
        self._consecutive_waits = 0
        self._narrative_phase = "pre_storm"
        self._unresolved_failure_steps = {}
        self._radius_protection_log = []
        # v2.2: reset expanded action state
        self._prioritize_cooldown = 0
        self._mutual_aid_used = False
        self._lockdown_remaining = 0
        self._active_bridges = {}
        self._fast_repair_nodes = set()
        self._scada_patched = set()
        self._reroute_targets = {}
        self._prioritize_immune = set()
        self._load_reduce_nodes = {}
        self._neighbor_stability = {}
        self._sacrifice_nodes_log = []

        rng_seed = int(resolved_seed)
        self._rng = random.Random(rng_seed)

        self._node_states = {}
        for n in cfg["nodes"]:
            ns = _NodeState(
                n["node_id"],
                n["sector"],
                n["is_critical"],
                real_name=n.get("real_name", ""),   # real-world name from OSM/task data
            )
            ns.load = round(self._rng.uniform(0.45, 0.65), 3)
            self._node_states[n["node_id"]] = ns

        self._edge_states = {}
        for e in cfg["edges"]:
            key = f"{e['source_id']}->{e['target_id']}"
            self._edge_states[key] = True

        self._delayed_sector_active = {
            s: True for s in cfg.get("delayed_sectors", [])
        }
        self._coord_clear = {}
        self._partial_obs_nodes = set(cfg.get("partial_obs_nodes", []))

        self._stress_schedule = cfg["stress_schedule"]
        self._scada_start = None
        self._scada_event = None
        for step_n, event in self._stress_schedule.items():
            if event.get("type") == "scada_anomaly" and event.get("recurring"):
                self._scada_start = step_n
            self._scada_event = dict(event)
            break

        self._obs_buffer = {nid: [] for nid in self._node_states}

        # Build geo coverage matrix from task node definitions
        self._node_coords = {}
        if GEO_AVAILABLE:
            for n in cfg["nodes"]:
                nid = n["node_id"]
                lat = n.get("lat", DEFAULT_NODE_COORDS.get(nid, (0.0, 0.0, 0.0))[0])
                lon = n.get("lon", DEFAULT_NODE_COORDS.get(nid, (0.0, 0.0, 0.0))[1])
                radius = n.get("service_radius_km", DEFAULT_NODE_COORDS.get(nid, (0.0, 0.0, 0.0))[2])
                self._node_coords[nid] = (lat, lon, radius)
            self._coverage_matrix = compute_coverage_matrix(self._node_coords)
        else:
            self._coverage_matrix = {}

        obs = self._build_observation()
        self._last_obs = obs
        return obs

    def step(self, action: CascadeAction, timeout_s: float | None = None, **kwargs) -> CascadeObservation:  # type: ignore[override]
        if self._done:
            obs = self._last_obs
            obs.reward = 0.0
            obs.done = True
            obs.metadata = {"reason": "episode_already_done"}
            return obs

        self._expire_isolations()

        # v2.2: lockdown guard — skip cascade and recovery this step when active
        if self._lockdown_remaining > 0:
            self._lockdown_remaining -= 1
            # Still build observation and return; no cascade/recovery ticks fire.
            obs = self._build_observation()
            self._last_obs = obs
            obs.reward = 0.0
            obs.done = self._check_done()
            self._step += 1
            obs.metadata = {"reason": "lockdown_active", "lockdown_remaining": self._lockdown_remaining}
            return obs

        # v2.2: decrement prioritize cooldown at the top of each step
        if self._prioritize_cooldown > 0:
            self._prioritize_cooldown -= 1

        # v2.2: decrement active bridge TTLs and remove expired ones
        self._active_bridges = {k: v - 1 for k, v in self._active_bridges.items() if v > 1}

        # v2.2: decrement reroute supply windows
        self._reroute_targets = {k: v - 1 for k, v in self._reroute_targets.items() if v > 1}

        # v2.2: decrement load-reduce modifiers
        self._load_reduce_nodes = {k: v - 1 for k, v in self._load_reduce_nodes.items() if v > 1}

        # v2.2: decrement neighbor stability boost windows
        self._neighbor_stability = {k: v - 1 for k, v in self._neighbor_stability.items() if v > 1}

        # v2.2: clear 1-step prioritize immunity set from previous step
        self._prioritize_immune = set()

        reward: float = 0.0
        action_bonus: float = 0.0
        action_valid: bool = True
        parse_error: bool = False
        action_type: str = "wait"
        action_target: Optional[str] = None
        coord_useful: bool = False
        load_shed_protects: bool = False
        critical_shed_load: bool = False
        wait_under_pressure: bool = False
        dependency_recovery_score: float = 0.0
        cascade_impact_count: int = 0
        recovery_correct_order: bool = False
        recovery_wrong_order: bool = False
        completed_recoveries: List[str] = []
        newly_failed: List[str] = []

        # ----------------------------------------------------------------
        # 1. Validate / parse action
        # ----------------------------------------------------------------
        try:
            action_type = action.action_type
            action_target = action.target_node_id
        except Exception:
            parse_error = True
            action_valid = False
            action_type = "wait"

        # ----------------------------------------------------------------
        # 2. Apply action effects (with constraint checks)
        # ----------------------------------------------------------------
        invalid_reason: str = ""
        if not parse_error:
            if action_type == "harden":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif self._node_states[target].is_hardened:
                    action_valid = False
                    invalid_reason = "already_hardened"
                elif self._budget < HARDEN_COST:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    self._node_states[target].is_hardened = True
                    self._budget -= HARDEN_COST
                    # Proactive harden reward — boosted for v2
                    upcoming = self._steps_until_next_damaging_stress()
                    if upcoming is not None and upcoming <= 4:
                        action_bonus += 0.12
                        reward += 0.12

            elif action_type == "shed_load":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                else:
                    ns = self._node_states[target]
                    if ns.load <= 0.5:
                        action_valid = False
                        invalid_reason = "no_load_to_shed"
                    elif ns.is_critical and (ns.load - 0.2) < 0.2:
                        action_valid = False
                        invalid_reason = "unsafe_critical_load_shed"
                    else:
                        prev_load = ns.load
                        ns.load = max(0.1, ns.load - 0.2)
                        if prev_load > 0.85:
                            load_shed_protects = True
                        if ns.sector == "hospital" or ns.is_critical:
                            critical_shed_load = True

            elif action_type == "coordinate":
                target = action_target
                if target is not None and target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif self._budget < COORDINATE_COST:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    self._budget -= COORDINATE_COST
                    if target is None:
                        partial_sectors = {
                            self._node_states[nid].sector
                            for nid in self._partial_obs_nodes
                            if nid in self._node_states
                        }
                        for sector in list(self._delayed_sector_active.keys()):
                            if self._delayed_sector_active[sector]:
                                coord_useful = True
                                self._coord_clear[sector] = 3
                                self._delayed_sector_active[sector] = False
                        for sector in partial_sectors:
                            self._coord_clear[sector] = 3
                        if partial_sectors:
                            coord_useful = True
                    else:
                        sector = self._node_states[target].sector
                        if sector in self._delayed_sector_active and self._delayed_sector_active[sector]:
                            coord_useful = True
                        elif any(
                            self._node_states[n].sector == sector
                            for n in self._partial_obs_nodes
                            if n in self._node_states
                        ):
                            coord_useful = True
                        self._coord_clear[sector] = 3
                        if sector in self._delayed_sector_active:
                            self._delayed_sector_active[sector] = False

            elif action_type == "isolate":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                else:
                    ns = self._node_states[target]
                    if ns.is_operational:
                        action_valid = False
                        invalid_reason = "node_is_operational"
                    elif target in self._isolated_nodes:
                        action_valid = False
                        invalid_reason = "already_isolated"
                    else:
                        self._isolated_nodes.add(target)
                        self._isolation_expiry[target] = self._step + 3
                        action_bonus += 0.05
                        reward += 0.05

            elif action_type == "recover":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                else:
                    ns = self._node_states[target]
                    if ns.health > 0.0 and ns.is_operational:
                        action_valid = False
                        invalid_reason = "node_not_failed"
                    elif target in self._pending_recoveries:
                        action_valid = False
                        invalid_reason = "already_in_recovery"
                    else:
                        upstream_ok = self._check_upstream_operational(target)
                        if not upstream_ok:
                            recovery_wrong_order = True
                            action_valid = False
                            invalid_reason = "blocked_by_upstream_dependency"
                        else:
                            recovery_correct_order = True
                            self._pending_recoveries[target] = RECOVERY_STEPS
                            ns.in_recovery = True
                            dependency_recovery_score = self._dependency_recovery_score(target)

            elif action_type == "wait":
                pass

            # ----------------------------------------------------------------
            # v2.2: New action handlers
            # ----------------------------------------------------------------
            elif action_type == "reroute":
                source = action.parameters.get("source") or action.target_node_id
                target = action.parameters.get("target")
                if source is None or source not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_source"
                elif target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif not self._node_states[source].is_operational:
                    action_valid = False
                    invalid_reason = "source_not_operational"
                elif self._budget < 0.6:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    # BFS to confirm a path exists from source to target in operational graph
                    if not self._bfs_operational_path(source, target):
                        action_valid = False
                        invalid_reason = "no_alternate_path"
                    else:
                        self._reroute_targets[target] = 2
                        self._budget -= 0.6
                        # Hospital reroute bonus
                        if self._node_states[target].sector == "hospital":
                            action_bonus += 0.05
                            reward += 0.05

            elif action_type == "prioritize":
                target = action.target_node_id
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif self._prioritize_cooldown > 0:
                    action_valid = False
                    invalid_reason = "cooldown_active"
                else:
                    ns = self._node_states[target]
                    ns.health = min(1.0, ns.health + 0.15)
                    self._prioritize_immune.add(target)
                    self._prioritize_cooldown = 3

            elif action_type == "deploy_repair_crew":
                target = action.target_node_id
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif target not in self._pending_recoveries:
                    action_valid = False
                    invalid_reason = "not_in_repair"
                elif self._budget < 1.5 * 0.8:  # 1.5x standard recover cost lower bound
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    self._fast_repair_nodes.add(target)
                    self._budget -= 1.5 * 0.8  # 1.5 × standard recover cost

            elif action_type == "emergency_shutdown":
                target = action.target_node_id
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif self._node_states[target].health >= 0.30:
                    action_valid = False
                    invalid_reason = "health_above_threshold"
                elif self._budget < 0.2:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    ns = self._node_states[target]
                    ns.is_operational = False
                    ns.load = 0.0
                    self._budget -= 0.2

            elif action_type == "cross_sector_bridge":
                sector_a = action.parameters.get("sector_a") or action.target_node_id
                sector_b = action.parameters.get("sector_b")
                if sector_a is None or sector_b is None:
                    action_valid = False
                    invalid_reason = "missing_sector_parameter"
                elif sector_a == sector_b:
                    action_valid = False
                    invalid_reason = "same_sector"
                elif self._budget < 1.5:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    self._active_bridges[(sector_a, sector_b)] = 3
                    self._budget -= 1.5

            elif action_type == "patch_scada":
                target = action.target_node_id
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                elif not self._has_active_scada(target):
                    action_valid = False
                    invalid_reason = "no_active_scada"
                elif self._budget < 0.8:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    self._scada_patched.add(target)
                    self._budget -= 0.8
                    action_bonus += 0.05
                    reward += 0.05

            elif action_type == "redistribute_load":
                node_a = action.parameters.get("node_a") or action.target_node_id
                node_b = action.parameters.get("node_b")
                if node_a is None or node_a not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_node_a"
                elif node_b is None or node_b not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_node_b"
                elif not self._node_states[node_a].is_operational or not self._node_states[node_b].is_operational:
                    action_valid = False
                    invalid_reason = "node_not_operational"
                elif self._node_states[node_a].sector != self._node_states[node_b].sector:
                    action_valid = False
                    invalid_reason = "cross_sector_not_allowed"
                elif self._budget < 0.5:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    ns_a = self._node_states[node_a]
                    ns_b = self._node_states[node_b]
                    transfer = 0.2
                    ns_a.load = max(0.0, min(1.0, ns_a.load - transfer))
                    ns_b.load = max(0.0, min(1.0, ns_b.load + transfer))
                    self._load_reduce_nodes[node_a] = 2
                    self._budget -= 0.5

            elif action_type == "request_mutual_aid":
                sector = action.parameters.get("sector") or action.target_node_id
                if sector is None:
                    action_valid = False
                    invalid_reason = "missing_sector_parameter"
                elif self._mutual_aid_used:
                    action_valid = False
                    invalid_reason = "already_used_this_episode"
                elif self._budget < 2.5:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    for ns in self._node_states.values():
                        if ns.sector == sector:
                            ns.health = min(1.0, ns.health + 0.2)
                    self._mutual_aid_used = True
                    self._budget -= 2.5
                    action_bonus += 0.05
                    reward += 0.05

            elif action_type == "controlled_cascade":
                target = action.target_node_id
                if target is None or target not in self._node_states:
                    action_valid = False
                    invalid_reason = "unknown_target"
                else:
                    ns = self._node_states[target]
                    ns.health = max(0.0, ns.health - 0.5)
                    self._sacrifice_nodes_log.append(target)
                    # Give neighbors a stability boost
                    neighbors = self._get_node_neighbors(target)
                    for neighbor_id in neighbors:
                        self._neighbor_stability[neighbor_id] = 2
                        n_ns = self._node_states[neighbor_id]
                        n_ns.health = min(1.0, n_ns.health + 0.15)

            elif action_type == "multi_sector_lockdown":
                if self._budget < 2.0:
                    action_valid = False
                    invalid_reason = "insufficient_budget"
                else:
                    self._lockdown_remaining = 2
                    self._budget -= 2.0


        # ----------------------------------------------------------------
        # BUG 2 FIX: Capture baseline BEFORE any stress events fire.
        # Previously captured AFTER stress, giving wrong newly_failed detection.
        # ----------------------------------------------------------------
        prev_health_map: Dict[str, float] = {
            nid: ns.health for nid, ns in self._node_states.items()
        }
        prev_op: Dict[str, bool] = {
            nid: ns.is_operational for nid, ns in self._node_states.items()
        }

        # ----------------------------------------------------------------
        # 3. Apply stress_schedule events for this step
        # ----------------------------------------------------------------
        if self._step in self._stress_schedule:
            event = self._stress_schedule[self._step]
            self._apply_stress_event(event)

        # Update consecutive_waits counter (v2)
        if action_type == "wait":
            self._consecutive_waits += 1
        else:
            self._consecutive_waits = 0

        self._action_history.append(action_type)

        # Apply recurring SCADA anomaly — skip nodes that have been patched
        if (
            self._scada_start is not None
            and self._scada_event is not None
            and self._step >= self._scada_start
        ):
            t = self._scada_event.get("target")
            if t and t in self._node_states and t not in self._scada_patched:
                ns = self._node_states[t]
                if ns.health > 0.0:
                    effect = float(self._scada_event.get("effect", 0.0))
                    ns.health = max(0.0, ns.health + effect)

        # v2.2: apply reroute health bonus to reroute targets
        for node_id in self._reroute_targets:
            if node_id in self._node_states:
                self._node_states[node_id].health = min(1.0, self._node_states[node_id].health + 0.1)


        # ----------------------------------------------------------------
        # BUG 3 FIX: Update operational status BEFORE cascade propagation.
        # Previously is_operational stayed True from last step even after a
        # stress event dropped health below threshold - so cascade never
        # propagated from a just-faulted node in the same step.
        # ----------------------------------------------------------------
        self._apply_threshold_update()

        # ----------------------------------------------------------------
        # 4. Cascade simulation: downstream -= 0.3 when upstream fails.
        #    Now uses up-to-date is_operational from the threshold update above.
        # ----------------------------------------------------------------
        self._propagate_cascade()

        # ----------------------------------------------------------------
        # 5. Apply weather to non-hardened nodes
        # ----------------------------------------------------------------
        weather_dmg = WEATHER_DAMAGE.get(self._weather_forecast, 0.0)
        if weather_dmg > 0.0:
            for ns in self._node_states.values():
                if not ns.is_hardened and ns.health > 0.0:
                    ns.health = max(0.0, ns.health - weather_dmg)

        # Apply deterministic demand drift so load-shedding remains meaningful.
        self._apply_background_load_dynamics()

        # ----------------------------------------------------------------
        # 6. Advance pending_recoveries (counter -> 0 means health restored)
        # ----------------------------------------------------------------
        finished = []
        for node_id in list(self._pending_recoveries.keys()):
            # v2.2: fast-repair crew halves remaining steps (round up), applied once
            if node_id in self._fast_repair_nodes:
                current = self._pending_recoveries[node_id]
                import math as _math
                self._pending_recoveries[node_id] = _math.ceil(current / 2)
                self._fast_repair_nodes.discard(node_id)
            self._pending_recoveries[node_id] -= 1
            if self._pending_recoveries[node_id] <= 0:
                finished.append(node_id)


        for node_id in finished:
            del self._pending_recoveries[node_id]
            ns = self._node_states[node_id]
            ns.health = RECOVERY_HEALTH
            ns.is_operational = True
            ns.in_recovery = False
            completed_recoveries.append(node_id)

        # ----------------------------------------------------------------
        # 7. Load overload: sustained overload causes gradual health loss
        # ----------------------------------------------------------------
        for ns in self._node_states.values():
            if ns.load > 0.9 and ns.health > 0.0:
                ns.health = max(0.0, ns.health - 0.06)

        # ----------------------------------------------------------------
        # 8. Final threshold check - covers cascade/weather/overload damage
        # ----------------------------------------------------------------
        self._apply_threshold_update()

        # ----------------------------------------------------------------
        # BUG 2 FIX: newly_failed uses prev_op snapshot (pre-step operational
        # status) so any node that was up before this step but is now down
        # (for any reason) is counted.
        # ----------------------------------------------------------------
        newly_failed = [
            nid for nid, ns in self._node_states.items()
            if not ns.is_operational and prev_op.get(nid, True)
        ]
        cascade_impact_count = self._count_cascade_impact(newly_failed)

        urgent_pressure = (
            len(newly_failed) > 0
            or len(self._pending_recoveries) > 0
            or any(ns.load > 0.85 for ns in self._node_states.values())
            or self._weather_forecast != "clear"
            or any(not ns.is_operational and ns.is_critical for ns in self._node_states.values())
        )
        if action_type == "wait" and urgent_pressure:
            wait_under_pressure = True

        if action_type == "recover":
            self._dependency_order_log.append(1 if recovery_correct_order else 0)

        # ----------------------------------------------------------------
        # 9. Compute reward
        # ----------------------------------------------------------------
        reward += self._compute_reward(
            prev_health_map=prev_health_map,
            action_valid=action_valid,
            action_type=action_type,
            parse_error=parse_error,
            newly_failed=newly_failed,
            completed_recoveries=completed_recoveries,
            recovery_correct_order=recovery_correct_order,
            recovery_wrong_order=recovery_wrong_order,
            load_shed_protects=load_shed_protects,
            coord_useful=coord_useful,
            critical_shed_load=critical_shed_load,
            wait_under_pressure=wait_under_pressure,
            dependency_recovery_score=dependency_recovery_score,
            cascade_impact_count=cascade_impact_count,
            actionable_wait=self._is_actionable_wait(action_type),
        )

        # ----------------------------------------------------------------
        # 10. Check done conditions
        # ----------------------------------------------------------------
        done = self._check_done()
        if done:
            reward += self._compute_terminal_reward()

        # ----------------------------------------------------------------
        # 11. Update narrative phase (v2)
        # ----------------------------------------------------------------
        self._narrative_phase = self._compute_narrative_phase()

        # ----------------------------------------------------------------
        # 12. Log failure_history and hospital_health_log
        # ----------------------------------------------------------------
        current_failures: Set[str] = {
            nid for nid, ns in self._node_states.items() if not ns.is_operational
        }
        self._failure_history.append(current_failures)

        hosp_nodes = [ns for ns in self._node_states.values() if ns.sector == "hospital"]
        min_hosp_health = min((ns.health for ns in hosp_nodes), default=1.0)
        self._hospital_health_log.append(min_hosp_health)
        # Keep depth proxy bounded by node count for stable grader semantics.
        cascade_depth_proxy = min(cascade_impact_count, len(self._node_states))
        self._cascade_depth_log.append(cascade_depth_proxy)

        # Advance coordinate clear counters
        expired = []
        for sector in list(self._coord_clear.keys()):
            self._coord_clear[sector] -= 1
            if self._coord_clear[sector] <= 0:
                expired.append(sector)
        for sector in expired:
            del self._coord_clear[sector]
            if sector in self._delayed_sector_active:
                self._delayed_sector_active[sector] = True

        # ----------------------------------------------------------------
        # 12. Increment step
        # ----------------------------------------------------------------
        self._step += 1

        # ----------------------------------------------------------------
        # 13. Build observation and return
        # ----------------------------------------------------------------
        obs = self._build_observation()
        self._last_obs = obs
        self._done = done

        raw_reward = reward
        clipped_reward = max(STEP_REWARD_MIN, min(STEP_REWARD_MAX, reward))
        reward_span = STEP_REWARD_MAX - STEP_REWARD_MIN
        if reward_span <= 0:
            normalized_reward = 0.5
        else:
            normalized_reward = (clipped_reward - STEP_REWARD_MIN) / reward_span
        normalized_reward = max(STEP_REWARD_EPS, min(1.0 - STEP_REWARD_EPS, normalized_reward))

        # Uncapped mode: expose raw reward without [0,1] normalisation.
        # This allows training loops to see the full magnitude of failure vs success
        # (Mercor bonus sub-theme). Default is "clamped" which is OpenEnv-compliant.
        if self._reward_mode == "uncapped":
            exposed_reward = raw_reward          # can exceed [0, 1]
        elif self._reward_mode == "grpo":
            exposed_reward = raw_reward * 3.0
        else:
            exposed_reward = normalized_reward   # default: clamped to (eps, 1-eps)

        # Update unresolved failure pressure (v2 cascade debt)
        self._update_failure_pressure(newly_failed)
        reward_components = dict(getattr(self, "_last_reward_components", {}))
        reward_components["r_action_bonus"] = round(action_bonus, 4)

        info: Dict[str, Any] = {
            "step": self._step,
            "newly_failed": newly_failed,
            "cascade_impact_count": cascade_impact_count,
            "completed_recoveries": completed_recoveries,
            "action_valid": action_valid,
            "invalid_reason": invalid_reason,
            "wait_under_pressure": wait_under_pressure,
            "critical_shed_load": critical_shed_load,
            "dependency_recovery_score": round(dependency_recovery_score, 3),
            "weather": self._weather_forecast,
            "task_id": self._task_id,
            "raw_reward": round(raw_reward, 4),
            "clipped_reward": round(clipped_reward, 4),
            "normalized_reward": round(normalized_reward, 4),
            "reward_mode": self._reward_mode,
            "reward_clipped": raw_reward != clipped_reward,
            "consecutive_waits": self._consecutive_waits,
            "narrative_phase": self._narrative_phase,
            "isolated_nodes": sorted(self._isolated_nodes),
            "legal_actions_count": len(self.get_legal_actions()),
            "state_features": self.get_state_features(),
        }
        info.update(reward_components)

        obs.reward = exposed_reward
        obs.done = done
        obs.metadata = info
        return obs

    @property
    def state(self) -> CascadeState:
        sector_health: Dict[str, float] = self._compute_sector_summary()
        budget_total = float(getattr(self, "_budget_total", self._budget))

        # Compute episode score using the grader, including centrality-aware bonus
        failure_history_sets = [set(s) for s in self._failure_history]
        episode_score = grade(
            task_id=self._task_id,
            failure_history=failure_history_sets,
            hospital_health_log=self._hospital_health_log,
            final_sector_summary=sector_health,
            budget_spent=budget_total - self._budget,
            budget_total=budget_total,
            total_nodes=len(self._node_states),
            cascade_depth_log=self._cascade_depth_log,
            dependency_order_log=self._dependency_order_log,
            action_history=self._action_history,
            high_centrality_nodes=self._compute_high_centrality_nodes(),   # P2
        )

        # Keep score strictly in the evaluator-accepted open interval.
        episode_score = max(0.01, min(0.99, float(episode_score)))

        return CascadeState(
            episode_id=self._episode_id,
            step_count=self._step,
            task_id=self._task_id,
            budget_remaining=self._budget,
            sector_health=sector_health,
            score=episode_score,
            # BUG 5 FIX: include total_nodes so graders use the correct denominator
            total_nodes=len(self._node_states),
            failure_history=[list(s) for s in self._failure_history],
            hospital_health_log=list(self._hospital_health_log),
            cascade_depth_log=list(self._cascade_depth_log),
            dependency_order_log=list(self._dependency_order_log),
            action_history=list(self._action_history),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def get_legal_actions(self) -> List[dict]:
        """Return all currently valid actions as structured dictionaries."""
        legal: List[dict] = []

        # Tier 1 — Basic
        if self._budget >= HARDEN_COST:
            for nid, ns in self._node_states.items():
                if ns.is_operational and not ns.is_hardened:
                    legal.append({"action_type": "harden", "target_node_id": nid})

        for nid, ns in self._node_states.items():
            if (
                not ns.is_operational
                and nid not in self._pending_recoveries
                and self._check_upstream_operational(nid)
            ):
                legal.append({"action_type": "recover", "target_node_id": nid})

        legal.append({"action_type": "wait", "target_node_id": None})

        # Tier 2 — Intermediate
        for nid, ns in self._node_states.items():
            if ns.is_operational and ns.load > 0.5:
                if not (ns.is_critical and (ns.load - 0.2) < 0.2):
                    legal.append({"action_type": "shed_load", "target_node_id": nid})

        if self._budget >= COORDINATE_COST:
            has_delayed = any(self._delayed_sector_active.values()) or bool(self._partial_obs_nodes)
            if has_delayed:
                added_sectors: Set[str] = set()
                for nid, ns in self._node_states.items():
                    sector = ns.sector
                    sector_delayed = (
                        self._delayed_sector_active.get(sector, False)
                        or nid in self._partial_obs_nodes
                    )
                    if sector_delayed and sector not in added_sectors:
                        legal.append({"action_type": "coordinate", "target_node_id": nid})
                        added_sectors.add(sector)
                legal.append({"action_type": "coordinate", "target_node_id": None})

        for nid, ns in self._node_states.items():
            if not ns.is_operational and nid not in self._isolated_nodes:
                legal.append({"action_type": "isolate", "target_node_id": nid})

        # reroute: source must be operational and a BFS path must exist
        if self._budget >= 0.6:
            op_nodes = [nid for nid, ns in self._node_states.items() if ns.is_operational]
            for src in op_nodes:
                for tgt, tns in self._node_states.items():
                    if tgt != src and self._bfs_operational_path(src, tgt):
                        legal.append({
                            "action_type": "reroute",
                            "target_node_id": src,
                            "parameters": {"source": src, "target": tgt},
                        })

        # prioritize: cooldown must be 0
        if self._prioritize_cooldown == 0:
            for nid in self._node_states:
                legal.append({"action_type": "prioritize", "target_node_id": nid})

        # deploy_repair_crew: node must be in active recovery
        if self._budget >= 1.5 * 0.8:
            for nid in self._pending_recoveries:
                legal.append({"action_type": "deploy_repair_crew", "target_node_id": nid})

        # Tier 3 — Hard
        # emergency_shutdown: health < 0.30
        if self._budget >= 0.2:
            for nid, ns in self._node_states.items():
                if ns.health < 0.30:
                    legal.append({"action_type": "emergency_shutdown", "target_node_id": nid})

        # cross_sector_bridge: any two different sectors
        if self._budget >= 1.5:
            sectors = list({ns.sector for ns in self._node_states.values()})
            for i, sa in enumerate(sectors):
                for sb in sectors[i + 1:]:
                    legal.append({
                        "action_type": "cross_sector_bridge",
                        "target_node_id": sa,
                        "parameters": {"sector_a": sa, "sector_b": sb},
                    })

        # patch_scada: node must have active unpatched SCADA anomaly
        if self._budget >= 0.8:
            for nid in self._node_states:
                if self._has_active_scada(nid):
                    legal.append({"action_type": "patch_scada", "target_node_id": nid})

        # redistribute_load: both nodes operational, same sector
        if self._budget >= 0.5:
            op_nodes_list = [
                (nid, ns) for nid, ns in self._node_states.items() if ns.is_operational
            ]
            for i, (nid_a, ns_a) in enumerate(op_nodes_list):
                for nid_b, ns_b in op_nodes_list[i + 1:]:
                    if ns_a.sector == ns_b.sector:
                        legal.append({
                            "action_type": "redistribute_load",
                            "target_node_id": nid_a,
                            "parameters": {"node_a": nid_a, "node_b": nid_b},
                        })

        # request_mutual_aid: once per episode, per sector
        if not self._mutual_aid_used and self._budget >= 2.5:
            for sector in {ns.sector for ns in self._node_states.values()}:
                legal.append({
                    "action_type": "request_mutual_aid",
                    "target_node_id": sector,
                    "parameters": {"sector": sector},
                })

        # Tier 4 — Expert
        # controlled_cascade: any node (agent chooses sacrifice)
        for nid in self._node_states:
            legal.append({"action_type": "controlled_cascade", "target_node_id": nid})

        # multi_sector_lockdown
        if self._budget >= 2.0:
            legal.append({"action_type": "multi_sector_lockdown", "target_node_id": None})

        return legal

    def get_state_features(self) -> List[float]:
        """Return a compact fixed-length state vector for training/debugging."""
        n_nodes = len(self._node_states)
        failed = [ns for ns in self._node_states.values() if not ns.is_operational]
        hosp = [ns for ns in self._node_states.values() if ns.sector == "hospital"]
        weather_feature = (
            1.0 if self._weather_forecast == "clear"
            else 0.5 if self._weather_forecast == "storm_warning"
            else 0.0
        )
        return [
            self._step / max(self._max_steps, 1),
            self._budget / max(self._budget_total, 1.0),
            len(failed) / max(n_nodes, 1),
            len(self._pending_recoveries) / max(n_nodes, 1),
            min(1.0, self._consecutive_waits / 10.0),
            min(1.0, len(self._unresolved_failure_steps) * 0.1),
            min((ns.health for ns in hosp), default=1.0),
            weather_feature,
        ]

    def _bfs_operational_path(self, source: str, target: str) -> bool:
        """BFS over the operational graph to check if a path exists from source to target."""
        if source == target:
            return True
        # Build adjacency list from both directions of the edge graph
        adjacency: Dict[str, List[str]] = {nid: [] for nid in self._node_states}
        for key in self._edge_states:
            src, tgt = key.split("->")
            if src in adjacency and tgt in adjacency:
                adjacency[src].append(tgt)
                adjacency[tgt].append(src)  # undirected for reroute purposes
        visited: Set[str] = {source}
        queue: List[str] = [source]
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor == target:
                    return True
                if neighbor not in visited and self._node_states.get(neighbor) and self._node_states[neighbor].is_operational:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def _has_active_scada(self, node_id: str) -> bool:
        """Return True if node_id has an active, unpatched SCADA anomaly drain."""
        if node_id in self._scada_patched:
            return False
        if (
            self._scada_start is None
            or self._scada_event is None
            or self._step < self._scada_start
        ):
            return False
        return self._scada_event.get("target") == node_id

    def _get_node_neighbors(self, node_id: str) -> List[str]:
        """Return all adjacent node IDs (both incoming and outgoing edges)."""
        neighbors: List[str] = []
        for key in self._edge_states:
            src, tgt = key.split("->")
            if src == node_id and tgt in self._node_states:
                neighbors.append(tgt)
            elif tgt == node_id and src in self._node_states:
                neighbors.append(src)
        return neighbors

    def _expire_isolations(self) -> None:
        expired = [
            nid for nid, expires_at in self._isolation_expiry.items()
            if expires_at <= self._step
        ]
        for nid in expired:
            self._isolated_nodes.discard(nid)
            self._isolation_expiry.pop(nid, None)

    def _apply_threshold_update(self) -> None:
        """BUG 3 FIX: Update is_operational for all nodes based on current
        health values. Called (a) after stress events and (b) after cascade/
        weather/overload so cascade uses up-to-date operational status."""
        for ns in self._node_states.values():
            ns.health = max(0.0, min(1.0, ns.health))
            threshold = FAILURE_THRESHOLD_HARDENED if ns.is_hardened else FAILURE_THRESHOLD_UNHARDENED
            if ns.health == 0.0 or (ns.health < threshold and not ns.in_recovery):
                if ns.health < threshold:
                    ns.health = 0.0
                ns.is_operational = False
            elif not ns.in_recovery:
                ns.is_operational = True

    def _steps_until_next_damaging_stress(self) -> Optional[int]:
        """Return steps until next stress event that deals direct health damage."""
        future_damaging = [
            s for s, ev in self._stress_schedule.items()
            if s > self._step and ev.get("type") in ("equipment_fault", "scada_anomaly")
        ]
        # Also include weather events that cause damage
        future_weather = [
            s for s, ev in self._stress_schedule.items()
            if s > self._step and ev.get("type") == "weather"
            and ev.get("effect") in ("storm_warning", "extreme_storm")
        ]
        all_future = future_damaging + future_weather
        if not all_future:
            return None
        return min(all_future) - self._step

    def _apply_stress_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "weather":
            self._weather_forecast = event["effect"]
        elif etype == "equipment_fault":
            target = event.get("target")
            if target and target in self._node_states:
                ns = self._node_states[target]
                ns.health = max(0.0, ns.health + event["effect"])
        elif etype == "load_surge":
            effect = float(event.get("effect", 0.2))
            targets = event.get("targets")
            if isinstance(targets, list):
                target_nodes = [str(nid) for nid in targets if nid in self._node_states]
            else:
                target_nodes = [
                    nid
                    for nid, ns in self._node_states.items()
                    if ns.is_operational and ns.sector != "hospital"
                ]
            for nid in target_nodes:
                ns = self._node_states[nid]
                ns.load = max(0.0, min(1.0, ns.load + effect))

    def inject_attack(self, attack_event: Optional[dict]) -> None:
        """
        Inject an adversarial attack into the environment before the next
        step resolves. Called by the multi-agent training loop AFTER the
        defender picks its action, but BEFORE env.step() commits physics.

        attack_event format:
            {"type": "equipment_fault", "target": "NODE_ID", "effect": -0.15}

        Passing None is safe (no-op). Use with adversarial_attacker.
        """
        if attack_event is None:
            return
        self._apply_stress_event(attack_event)
        # Immediately update operational status so cascade in next step() sees it
        self._apply_threshold_update()

    def _compute_high_centrality_nodes(self) -> List[str]:
        """
        Return node IDs with degree >= 3 as a proxy for backbone / high-centrality.
        Degree = count of edge endpoints incident to the node (both directions).
        Used by grade_hard / grade_cyberattack centrality_protection_score.
        """
        degree: Dict[str, int] = {nid: 0 for nid in self._node_states}
        for key in self._edge_states:
            src, tgt = key.split("->")
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1
        return [nid for nid, deg in degree.items() if deg >= 3]

    def _propagate_cascade(self) -> None:
        """Propagate health damage from failed upstream nodes.
        BUG 3: Now called AFTER _apply_threshold_update() so nodes that
        JUST failed this step (due to stress events) correctly damage their
        downstream dependents in the same step."""
        dependents: Dict[str, List[str]] = {nid: [] for nid in self._node_states}
        for key in self._edge_states:
            src, tgt = key.split("->")
            if src in self._node_states and tgt in self._node_states:
                dependents[tgt].append(src)

        for tgt_id, sources in dependents.items():
            tgt_ns = self._node_states[tgt_id]
            if tgt_ns.health == 0.0:
                continue
            for src_id in sources:
                src_ns = self._node_states[src_id]
                if not src_ns.is_operational and src_id not in self._isolated_nodes:
                    tgt_ns.health = max(0.0, tgt_ns.health - CASCADE_DAMAGE)
                    tgt_ns.load = min(1.0, tgt_ns.load + 0.2)

    def _check_upstream_operational(self, node_id: str) -> bool:
        for key in self._edge_states:
            src, tgt = key.split("->")
            if tgt == node_id and src in self._node_states:
                if not self._node_states[src].is_operational:
                    return False
        return True

    def _compute_sector_summary(self) -> Dict[str, float]:
        sector_totals: Dict[str, List[float]] = {}
        for ns in self._node_states.values():
            sector_totals.setdefault(ns.sector, []).append(ns.health)
        return {
            s: (sum(vals) / len(vals))
            for s, vals in sector_totals.items()
        }

    def _dependency_recovery_score(self, node_id: str) -> float:
        """Higher score for recovering nodes that unlock failed dependents."""
        dependent_nodes = [
            key.split("->")[1]
            for key in self._edge_states
            if key.split("->")[0] == node_id
        ]
        unlocked = sum(
            1
            for nid in dependent_nodes
            if nid in self._node_states and not self._node_states[nid].is_operational
        )
        is_root = not any(
            key.split("->")[1] == node_id
            for key in self._edge_states
        )
        steps_unresolved = self._unresolved_failure_steps.get(node_id, 0)
        urgency_bonus = min(0.3, 0.05 * steps_unresolved)

        if unlocked == 0 and not is_root:
            return min(1.0, max(urgency_bonus, 0.05))

        base = 0.3 + 0.2 * unlocked
        if is_root and unlocked == 0:
            base = 0.4
        return min(1.0, base + urgency_bonus)

    def _count_cascade_impact(self, newly_failed: List[str]) -> int:
        if not newly_failed:
            return 0
        failed_set = set(newly_failed)
        impacted = 0
        for key in self._edge_states:
            src, tgt = key.split("->")
            if src in failed_set and tgt in failed_set:
                impacted += 1
        return impacted

    def _build_dependency_feedback(
        self,
        observed_nodes: List[NodeStatus],
        observed_edges: List[EdgeStatus],
        pending_rec_list: List[str],
    ) -> tuple[List[str], List[str], List[str], List[str], float]:
        """Build diagnostics from observable channels only (leakage-safe)."""
        node_map = {n.node_id: n for n in observed_nodes}
        critical_failures = [
            n.node_id for n in observed_nodes if n.is_critical and not n.is_operational
        ]

        at_risk_scored: List[tuple[float, str]] = []
        for ns in observed_nodes:
            if not ns.is_operational:
                continue
            downstream = sum(1 for e in observed_edges if e.source_id == ns.node_id)
            risk = (1.0 - ns.health) + max(0.0, ns.load - 0.7)
            if ns.is_critical:
                risk += 0.4
            risk += 0.05 * downstream
            if risk >= 0.45:
                at_risk_scored.append((risk, ns.node_id))
        at_risk_scored.sort(reverse=True)
        at_risk_nodes = [nid for _, nid in at_risk_scored[:5]]

        dependency_alerts: List[str] = []
        for e in observed_edges:
            src = node_map.get(e.source_id)
            tgt = node_map.get(e.target_id)
            if src is None or tgt is None:
                continue
            if (not src.is_operational) and tgt.is_operational:
                dependency_alerts.append(f"{src.node_id} blocks {tgt.node_id}")
        dependency_alerts = dependency_alerts[:6]

        recover_candidates: List[tuple[int, int, str]] = []
        for ns in observed_nodes:
            nid = ns.node_id
            if ns.is_operational or nid in pending_rec_list:
                continue
            failed_upstream = sum(
                1
                for e in observed_edges
                if e.target_id == nid
                and e.source_id in node_map
                and not node_map[e.source_id].is_operational
            )
            failed_dependents = sum(
                1
                for e in observed_edges
                if e.source_id == nid
                and e.target_id in node_map
                and not node_map[e.target_id].is_operational
            )
            recover_candidates.append((failed_upstream, -failed_dependents, nid))
        recover_candidates.sort()
        recommended_recovery_order = [nid for _, _, nid in recover_candidates[:5]]

        failed_count = sum(1 for n in observed_nodes if not n.is_operational)
        overloaded = sum(1 for n in observed_nodes if n.load > 0.85)
        system_pressure = min(
            1.0,
            0.15 * failed_count
            + 0.1 * overloaded
            + (0.2 if self._weather_forecast != "clear" else 0.0),
        )

        return critical_failures, at_risk_nodes, dependency_alerts, recommended_recovery_order, round(system_pressure, 3)

    def _build_observation(self) -> CascadeObservation:
        # Get telecom operational status for observation confidence computation
        telecom_operational = []
        if GEO_AVAILABLE and self._node_coords:
            for nid, ns in self._node_states.items():
                if ns.sector == "telecom" and ns.is_operational and nid in self._node_coords:
                    lat, lon, radius = self._node_coords[nid]
                    telecom_operational.append((nid, lat, lon, radius))

        # Fill rolling observation buffer BEFORE building delayed observations.
        # This keeps delay windows episode-local and avoids stale ordering artifacts.
        for nid, ns in self._node_states.items():
            buf = self._obs_buffer.setdefault(nid, [])
            buf.append((ns.health, ns.load))
            if len(buf) > 4:
                buf.pop(0)

        node_list: List[NodeStatus] = []
        for nid, ns in self._node_states.items():
            is_delayed = self._is_node_delayed(nid)
            if is_delayed:
                buf = self._obs_buffer.get(nid, [])
                if len(buf) >= 2:
                    health_obs, load_obs = buf[-2]
                else:
                    health_obs, load_obs = ns.health, ns.load
            else:
                health_obs, load_obs = ns.health, ns.load

            # v2: observation confidence based on telecom coverage
            obs_confidence = 1.0
            if GEO_AVAILABLE and self._node_coords and nid in self._node_coords:
                n_lat, n_lon, _ = self._node_coords[nid]
                obs_confidence = compute_obs_confidence(nid, telecom_operational, n_lat, n_lon)
                # Low confidence = add slight noise (keeps expected value correct, adds variance)
                if obs_confidence < 0.7 and is_delayed:
                    noise = self._rng.gauss(0, (1.0 - obs_confidence) * 0.05)
                    health_obs = max(0.0, min(1.0, health_obs + noise))

            op_obs = ns.is_operational
            if is_delayed:
                delayed_threshold = (
                    FAILURE_THRESHOLD_HARDENED if ns.is_hardened else FAILURE_THRESHOLD_UNHARDENED
                )
                op_obs = health_obs >= delayed_threshold and not ns.in_recovery

            # v2: geo fields from node_coords
            n_lat, n_lon, n_radius = (0.0, 0.0, 0.0)
            if GEO_AVAILABLE and nid in self._node_coords:
                n_lat, n_lon, n_radius = self._node_coords[nid]

            node_list.append(
                NodeStatus(
                    node_id=nid,
                    sector=ns.sector,
                    health=health_obs,
                    load=load_obs,
                    is_operational=op_obs,
                    is_hardened=ns.is_hardened,
                    observation_delayed=is_delayed,
                    is_critical=ns.is_critical,
                    real_name=ns.real_name,             # real-world infrastructure name
                    lat=n_lat,
                    lon=n_lon,
                    service_radius_km=n_radius,
                    observation_confidence=round(obs_confidence, 3),
                )
            )

        edge_list: List[EdgeStatus] = []
        for key in self._edge_states:
            src, tgt = key.split("->")
            src_op = self._node_states[src].is_operational if src in self._node_states else True
            tgt_op = self._node_states[tgt].is_operational if tgt in self._node_states else True
            edge_list.append(EdgeStatus(source_id=src, target_id=tgt, active=src_op and tgt_op))

        sector_summary = self._compute_sector_summary()
        active_failures = [nid for nid, ns in self._node_states.items() if not ns.is_operational]
        pending_rec_list = list(self._pending_recoveries.keys())
        (
            critical_failures,
            at_risk_nodes,
            dependency_alerts,
            recommended_recovery_order,
            system_pressure,
        ) = self._build_dependency_feedback(node_list, edge_list, pending_rec_list)

        # v2: radius coverage events for this step
        radius_events: List[str] = []
        if GEO_AVAILABLE and self._coverage_matrix and active_failures:
            op_map = {nid: ns.is_operational for nid, ns in self._node_states.items()}
            sector_map = {nid: ns.sector for nid, ns in self._node_states.items()}
            radius_events = coverage_events_summary(
                self._coverage_matrix, op_map, active_failures, sector_map
            )

        narrative = getattr(self, "_narrative_phase", "pre_storm")
        consec_waits = getattr(self, "_consecutive_waits", 0)
        upcoming_stress = self._steps_until_next_damaging_stress()
        if upcoming_stress is not None and upcoming_stress <= 4:
            stress_level = "imminent"
        elif upcoming_stress is not None and upcoming_stress <= 8:
            stress_level = "soon"
        else:
            stress_level = "none"

        info_str = (
            f"[{narrative.upper()}] step={self._step}/{self._max_steps} | "
            f"task={self._task_id} | weather={self._weather_forecast} "
            f"| budget={self._budget:.1f} | failures={len(active_failures)} "
            f"| pressure={system_pressure:.2f} | waits={consec_waits} "
            f"| stress={stress_level}"
        )

        return CascadeObservation(
            step=self._step,
            max_steps=self._max_steps,
            budget_remaining=self._budget,
            weather_forecast=self._weather_forecast,
            nodes=node_list,
            edges=edge_list,
            active_failures=active_failures,
            pending_recoveries=pending_rec_list,
            sector_summary=sector_summary,
            diagnostics=ObservationDiagnostics(
                critical_failures=critical_failures,
                at_risk_nodes=at_risk_nodes,
                dependency_alerts=dependency_alerts,
                recommended_recovery_order=recommended_recovery_order,
                system_pressure=system_pressure,
                leakage_safe=True,
            ),
            task_id=self._task_id,
            info=info_str,
            radius_coverage_events=radius_events,
            narrative_phase=narrative,
            consecutive_waits=consec_waits,
            upcoming_stress_level=stress_level,
        )

    def _is_actionable_wait(self, action_type: str) -> bool:
        if action_type != "wait":
            return False

        recoverable_failed = any(
            (not ns.is_operational)
            and (nid not in self._pending_recoveries)
            and self._check_upstream_operational(nid)
            for nid, ns in self._node_states.items()
        )
        overload_present = any(ns.is_operational and ns.load > 0.85 for ns in self._node_states.values())
        delayed_visible = bool(self._partial_obs_nodes) or any(self._delayed_sector_active.values())
        coord_opportunity = delayed_visible and self._budget >= COORDINATE_COST
        isolatable_failed = any(
            (not ns.is_operational) and nid not in self._isolated_nodes
            for nid, ns in self._node_states.items()
        )
        return recoverable_failed or overload_present or coord_opportunity or isolatable_failed

    def _is_node_delayed(self, node_id: str) -> bool:
        if node_id in self._partial_obs_nodes:
            if node_id in self._node_states:
                sector = self._node_states[node_id].sector
                if sector in self._coord_clear:
                    return False
            return True
        if node_id in self._node_states:
            sector = self._node_states[node_id].sector
            if sector in self._delayed_sector_active and self._delayed_sector_active[sector]:
                return True
        return False

    def _apply_background_load_dynamics(self) -> None:
        """Small stochastic load drift to emulate changing demand over time."""
        weather_boost = {
            "clear": 0.0,
            "storm_warning": 0.02,
            "extreme_storm": 0.04,
        }.get(self._weather_forecast, 0.0)
        system_load_target = 0.58
        if self._weather_forecast != "clear":
            system_load_target = 0.70
        if any(not ns.is_operational for ns in self._node_states.values()):
            system_load_target = 0.75

        for ns in self._node_states.values():
            if not ns.is_operational:
                continue

            # Mean reversion prevents runaway overload spirals while keeping dynamics stochastic.
            reversion = (system_load_target - ns.load) * 0.12
            drift = self._rng.uniform(-0.02, 0.03) + weather_boost + reversion
            if ns.sector == "hospital":
                drift += 0.01
            ns.load = max(0.1, min(1.0, ns.load + drift))

    def _compute_reward(
        self,
        prev_health_map: Dict[str, float],
        action_valid: bool,
        action_type: str,
        parse_error: bool,
        newly_failed: List[str],
        completed_recoveries: List[str],
        recovery_correct_order: bool,
        recovery_wrong_order: bool,
        load_shed_protects: bool,
        coord_useful: bool,
        critical_shed_load: bool,
        wait_under_pressure: bool,
        dependency_recovery_score: float,
        cascade_impact_count: int,
        actionable_wait: bool,
    ) -> float:
        r: float = 0.0
        components: Dict[str, float] = {
            "r_wait_pressure": 0.0,
            "r_cascade_debt": 0.0,
            "r_newly_failed": 0.0,
            "r_recovery": 0.0,
            "r_hospital": 0.0,
            "r_harden_bonus": 0.0,
            "r_grader_shaped_bonus": 0.0,
            "r_reward_noise": 0.0,
        }

        # Base time-step cost
        r -= 0.01

        if action_type == "wait":
            r -= WAIT_BASE_PENALTY

        # v2: Progressive wait penalty — escalates with consecutive waits
        # This breaks the GRPO "always wait" collapse by making repeated inaction
        # increasingly costly. After 3 consecutive waits, penalty doubled.
        if wait_under_pressure:
            consecutive = getattr(self, "_consecutive_waits", 0)
            pressure_scale = min(2.0, 1.0 + 0.15 * consecutive)  # caps at 2x after ~7 waits
            wait_penalty = -WAIT_UNDER_PRESSURE_PENALTY * pressure_scale
            components["r_wait_pressure"] = wait_penalty
            r += wait_penalty

        if actionable_wait:
            r -= 0.07

        if parse_error:
            r -= 0.05

        if not action_valid and not parse_error and not recovery_wrong_order:
            r -= 0.10

        if recovery_wrong_order:
            r -= 0.20

        if critical_shed_load:
            r -= CRITICAL_SHED_LOAD_PENALTY

        # Hardened node survives a failure event this step
        for nid, ns in self._node_states.items():
            if ns.is_hardened and ns.is_operational:
                prev = prev_health_map.get(nid, 1.0)
                if prev < FAILURE_THRESHOLD_UNHARDENED and ns.health >= FAILURE_THRESHOLD_HARDENED:
                    components["r_harden_bonus"] += 0.15
                    r += 0.15

        # ----------------------------------------------------------------
        # BUG 4 FIX: Unified hospital penalty - single consistent pass.
        # Old code had two separate loops that could interact incorrectly.
        # New code: one loop categorises each hospital node into exactly
        # one bucket with a single penalty/reward.
        # ----------------------------------------------------------------
        for ns in self._node_states.values():
            if ns.sector != "hospital":
                continue
            if not ns.is_operational:
                # Hospital fully failed: heavy penalty (once per failure)
                if ns.node_id in newly_failed:
                    components["r_hospital"] -= 0.50
                    r -= 0.50   # newly failed this step
                # (already-failed hospitals give no +0.08 and no extra penalty)
            elif ns.health > 0.4:
                # Reward stability only when hospital improved vs pre-step state.
                prev_h = prev_health_map.get(ns.node_id, ns.health)
                if ns.health > prev_h + 1e-6:
                    components["r_hospital"] += 0.08
                    r += 0.08
            else:
                # Hospital operational but health between 0 and 0.4: danger zone
                components["r_hospital"] -= 0.30
                r -= 0.30   # reduced from -0.50 to avoid double-hit instability

        # Node fails during this step (all sectors, including any hospital)
        components["r_newly_failed"] = -0.25 * len(newly_failed)
        r += components["r_newly_failed"]

        # Recovery completes
        components["r_recovery"] = 0.30 * len(completed_recoveries)
        r += components["r_recovery"]

        # v2: extra reward for valid recovery (imitation bootstrap signal)
        if recovery_correct_order:
            r += 0.10
            # Additional bonus for recovering critical-path nodes
            if recovery_correct_order:
                r += 0.05  # small boost so recovery is clearly better than wait

        if dependency_recovery_score > 0.0:
            r += DEPENDENCY_RECOVERY_BONUS * dependency_recovery_score

        if cascade_impact_count > 0:
            r -= CASCADE_IMPACT_PENALTY * cascade_impact_count

        # v2: Cascade debt penalty — unresolved failures accumulate pressure
        cascade_debt_penalty = self._compute_cascade_debt_penalty()
        components["r_cascade_debt"] = -cascade_debt_penalty
        r -= cascade_debt_penalty

        if load_shed_protects:
            r += 0.10

        if coord_useful:
            r += 0.10

        grader_bonus = self._compute_grader_shaped_bonus()
        components["r_grader_shaped_bonus"] = grader_bonus
        r += grader_bonus

        if self._reward_mode == "grpo" and self._reward_noise_scale > 0.0:
            noise = self._rng.gauss(0.0, self._reward_noise_scale)
            components["r_reward_noise"] = noise
            r += noise

        self._last_reward_components = {
            key: round(value, 4) for key, value in components.items()
        }
        return r

    def _compute_grader_shaped_bonus(self) -> float:
        """Small per-step bonus mirroring grader components in GRPO mode."""
        if self._reward_mode != "grpo":
            return 0.0

        bonus = 0.0
        hosp_nodes = [ns for ns in self._node_states.values() if ns.sector == "hospital"]
        if hosp_nodes:
            avg_hosp = sum(ns.health for ns in hosp_nodes) / len(hosp_nodes)
            bonus += 0.03 * avg_hosp

        current_failed = sum(1 for ns in self._node_states.values() if not ns.is_operational)
        if current_failed == 0:
            bonus += 0.02

        sector_summary = self._compute_sector_summary()
        all_healthy = all(v >= 0.7 for v in sector_summary.values())
        if all_healthy and self._budget > 0.5 * self._budget_total:
            bonus += 0.01

        return bonus

    def _compute_terminal_reward(self) -> float:
        r: float = 0.0
        sector_summary = self._compute_sector_summary()
        if all(v >= 0.5 for v in sector_summary.values()):
            r += 1.0
        if any(v == 0.0 for v in sector_summary.values()):
            r -= 1.0
        return r

    def _check_done(self) -> bool:
        # Condition 1: time limit
        if self._step >= self._max_steps - 1:
            return True

        if self._training_mode:
            return False

        # Condition 2: catastrophic sustained blackout (3 consecutive steps)
        sector_summary = self._compute_sector_summary()
        if any(v == 0.0 for v in sector_summary.values()):
            self._zero_sector_streak += 1
        else:
            self._zero_sector_streak = 0
        if self._step >= 6 and self._zero_sector_streak >= 4:
            return True

        # Condition 3: early success
        all_op = all(ns.is_operational for ns in self._node_states.values())
        weather_clear = self._weather_forecast == "clear"
        last_stress_step = max(self._stress_schedule.keys()) if self._stress_schedule else 0
        all_events_fired = self._step > last_stress_step
        no_pending = len(self._pending_recoveries) == 0

        # BUG 6 FIX: Require minimum steps so the episode runs long enough
        # for the agent to actually encounter stress events and learn from them.
        # min_steps = max(last_stress_step + 2, max_steps // 3)
        # Examples:
        #   task_easy  (max=10, last_stress=2):  min_steps = max(4, 3)  = 4
        #   task_medium(max=20, last_stress=12): min_steps = max(14, 6) = 14
        #   task_hard  (max=30, last_stress=15): min_steps = max(17, 10)= 17
        min_steps_for_early_exit = max(last_stress_step + 3, int(math.ceil(self._max_steps * 0.6)))

        if (
            self._step >= min_steps_for_early_exit
            and all_op
            and weather_clear
            and all_events_fired
            and no_pending
        ):
            return True

        return False
    def _compute_narrative_phase(self) -> str:
        """Determine the story phase of the episode for display and diagnostics."""
        sector_summary = self._compute_sector_summary()
        failed_count = sum(1 for ns in self._node_states.values() if not ns.is_operational)
        pending_count = len(self._pending_recoveries)
        weather = self._weather_forecast
        step_frac = self._step / max(self._max_steps, 1)

        if step_frac < 0.15 and weather == "clear" and failed_count == 0:
            return "pre_storm"
        elif weather in ("storm_warning", "extreme_storm") and failed_count <= 2:
            return "storm_onset"
        elif failed_count >= 3 or any(v < 0.3 for v in sector_summary.values()):
            return "crisis_peak"
        elif pending_count > 0 or failed_count > 0:
            return "recovery"
        else:
            return "stabilization"

    def _update_failure_pressure(self, newly_failed: list) -> None:
        """Track how many steps each failed node has been unresolved (cascade debt)."""
        # Increment counter for all currently failed nodes
        for nid, ns in self._node_states.items():
            if not ns.is_operational and nid not in self._pending_recoveries:
                self._unresolved_failure_steps[nid] = self._unresolved_failure_steps.get(nid, 0) + 1
            else:
                # Resolved or recovered — clear counter
                self._unresolved_failure_steps.pop(nid, None)

    def _compute_cascade_debt_penalty(self) -> float:
        """Return additional penalty from long-unresolved failures (cascade debt).

        After 3+ steps of inaction on a failed node, downstream pressure builds:
        each extra step of non-recovery adds -0.04 per deep-debt node.
        Capped to prevent runaway penalties.
        """
        debt_penalty = 0.0
        for nid, steps_unresolved in self._unresolved_failure_steps.items():
            if steps_unresolved >= 3:
                # Extra -0.04 per step beyond 3, capped at -0.20 per node
                extra = min(0.20, 0.04 * (steps_unresolved - 2))
                debt_penalty += extra
        return min(0.40, debt_penalty)  # global cap
