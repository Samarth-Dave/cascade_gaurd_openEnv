from __future__ import annotations

import copy
import json
import math
import random
import uuid
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server import Environment

from cascade_guard.models import (
    CascadeAction,
    CascadeObservation,
    CascadeState,
    EdgeStatus,
    NodeStatus,
    ObservationDiagnostics,
)
from cascade_guard.tasks import TASK_CONFIGS, materialize_task_config, resolve_seed
from cascade_guard.server.graders import grade

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HARDEN_COST: float = 2.0
COORDINATE_COST: float = 1.0
FAILURE_THRESHOLD_UNHARDENED: float = 0.3
FAILURE_THRESHOLD_HARDENED: float = 0.1
RECOVERY_STEPS: int = 2
RECOVERY_HEALTH: float = 0.8
CASCADE_DAMAGE: float = 0.3
WAIT_BASE_PENALTY: float = 0.04
WAIT_UNDER_PRESSURE_PENALTY: float = 0.22
CRITICAL_SHED_LOAD_PENALTY: float = 1.2
DEPENDENCY_RECOVERY_BONUS: float = 0.15
CASCADE_IMPACT_PENALTY: float = 0.12
WEATHER_DAMAGE: Dict[str, float] = {
    "clear": 0.0,
    "storm_warning": 0.1,
    "extreme_storm": 0.25,
}
STEP_REWARD_MIN: float = -1.0
STEP_REWARD_MAX: float = 1.0
STEP_REWARD_EPS: float = 1e-4


class _NodeState:
    """Internal mutable node state (server-side ground truth)."""

    def __init__(self, node_id: str, sector: str, is_critical: bool) -> None:
        self.node_id = node_id
        self.sector = sector
        self.is_critical = is_critical
        self.health: float = 1.0
        self.load: float = 0.5
        self.is_operational: bool = True
        self.is_hardened: bool = False
        self.in_recovery: bool = False

    def copy(self) -> "_NodeState":
        n = _NodeState(self.node_id, self.sector, self.is_critical)
        n.health = self.health
        n.load = self.load
        n.is_operational = self.is_operational
        n.is_hardened = self.is_hardened
        n.in_recovery = self.in_recovery
        return n


class CascadeEnvironment(Environment):
    """CascadeGuard OpenEnv environment - cross-sector infrastructure RL.

    VERSION 3 - Fixes all 6 structural bugs:

    Bug 1  FIXED: reset() now accepts seed, episode_id, task_id, **kwargs
           so openenv framework passes the task_id correctly.

    Bug 2  FIXED: prev_health_map captured BEFORE stress events fire,
           giving accurate pre-step baseline for reward/newly_failed detection.

        Bug 3  FIXED: _apply_threshold_update() called BEFORE cascade propagation
            -> cascade now correctly spreads from nodes that JUST failed this step.

        Bug 4  FIXED: Unified hospital penalty - no double-counting between
           newly_failed loop and per-step degraded-hospital loop.

    Bug 5  FIXED: state() includes total_nodes so graders use correct denominator.

    Bug 6  FIXED: _check_done() requires min steps (last_stress + 2 OR max//3)
           before early-success can trigger. Episodes run to meaningful length.
    """

    def __init__(self) -> None:
        self._task_id: str = "task_easy"
        self._step: int = 0
        self._node_states: Dict[str, _NodeState] = {}
        self._edge_states: Dict[str, bool] = {}
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
        self._delayed_sector_active: Dict[str, bool] = {}
        self._coord_clear: Dict[str, int] = {}
        self._partial_obs_nodes: Set[str] = set()
        self._obs_buffer: Dict[str, List[tuple]] = {}
        self._cascade_depth_log: List[int] = []
        self._dependency_order_log: List[int] = []
        self._action_history: List[str] = []

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
        resolved_seed = seed if seed is not None else resolve_seed(
            self._task_id,
            split=scenario_split,
            scenario_index=scenario_index,
            fallback_seed=int(TASK_CONFIGS[self._task_id]["seed"]),
        )

        cfg = materialize_task_config(self._task_id, seed=int(resolved_seed), split=scenario_split)
        self._config = cfg
        self._max_steps = cfg["max_steps"]
        self._budget_total = cfg["budget"]  # Store initial budget for computing budget_spent
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
        self._zero_sector_streak = 0
        self._episode_id = episode_id or str(uuid.uuid4())

        rng_seed = int(resolved_seed)
        self._rng = random.Random(rng_seed)

        self._node_states = {}
        for n in cfg["nodes"]:
            ns = _NodeState(n["node_id"], n["sector"], n["is_critical"])
            # Deterministic per-episode load variability creates realistic triage pressure.
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
        for step_n, event in self._stress_schedule.items():
            if event.get("type") == "scada_anomaly" and event.get("recurring"):
                self._scada_start = step_n

        self._obs_buffer = {nid: [] for nid in self._node_states}

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

        reward: float = 0.0
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
        if not parse_error:
            if action_type == "harden":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                elif self._node_states[target].is_hardened:
                    action_valid = False
                elif self._budget < HARDEN_COST:
                    action_valid = False
                else:
                    self._node_states[target].is_hardened = True
                    self._budget -= HARDEN_COST
                    # Proactive harden reward
                    upcoming = self._steps_until_next_damaging_stress()
                    if upcoming is not None and upcoming <= 4:
                        reward += 0.06

            elif action_type == "shed_load":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                else:
                    ns = self._node_states[target]
                    if ns.load <= 0.5:
                        action_valid = False
                    elif ns.is_critical and (ns.load - 0.2) < 0.2:
                        action_valid = False
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
                elif self._budget < COORDINATE_COST:
                    action_valid = False
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

            elif action_type == "recover":
                target = action_target
                if target is None or target not in self._node_states:
                    action_valid = False
                else:
                    ns = self._node_states[target]
                    if ns.health > 0.0 and ns.is_operational:
                        action_valid = False
                    elif target in self._pending_recoveries:
                        action_valid = False
                    else:
                        upstream_ok = self._check_upstream_operational(target)
                        if not upstream_ok:
                            recovery_wrong_order = True
                            action_valid = False
                        else:
                            recovery_correct_order = True
                            self._pending_recoveries[target] = RECOVERY_STEPS
                            ns.in_recovery = True
                            dependency_recovery_score = self._dependency_recovery_score(target)

            elif action_type == "wait":
                pass

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

        self._action_history.append(action_type)

        # Apply recurring SCADA anomaly
        if self._scada_start is not None and self._step >= self._scada_start:
            for step_n, event in self._stress_schedule.items():
                if event.get("type") == "scada_anomaly" and event.get("recurring"):
                    t = event.get("target")
                    if t and t in self._node_states:
                        ns = self._node_states[t]
                        if ns.health > 0.0:
                            ns.health = max(0.0, ns.health + event["effect"])
                    break

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
        # 7. Load overload: load > 0.85 -> health -= 0.1
        # ----------------------------------------------------------------
        for ns in self._node_states.values():
            if ns.load > 0.85 and ns.health > 0.0:
                ns.health = max(0.0, ns.health - 0.1)

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
        # 11. Log failure_history and hospital_health_log
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

        info: Dict[str, Any] = {
            "step": self._step,
            "newly_failed": newly_failed,
            "cascade_impact_count": cascade_impact_count,
            "completed_recoveries": completed_recoveries,
            "action_valid": action_valid,
            "wait_under_pressure": wait_under_pressure,
            "critical_shed_load": critical_shed_load,
            "dependency_recovery_score": round(dependency_recovery_score, 3),
            "weather": self._weather_forecast,
            "task_id": self._task_id,
            "raw_reward": round(raw_reward, 4),
            "clipped_reward": round(clipped_reward, 4),
            "normalized_reward": round(normalized_reward, 4),
            "reward_clipped": raw_reward != clipped_reward,
        }

        obs.reward = normalized_reward
        obs.done = done
        obs.metadata = info
        return obs

    @property
    def state(self) -> CascadeState:
        sector_health: Dict[str, float] = self._compute_sector_summary()
        
        # Compute episode score using the grader
        failure_history_sets = [set(s) for s in self._failure_history]
        episode_score = grade(
            task_id=self._task_id,
            failure_history=failure_history_sets,
            hospital_health_log=self._hospital_health_log,
            final_sector_summary=sector_health,
            budget_spent=self._budget_total - self._budget,
            budget_total=self._budget_total,
            total_nodes=len(self._node_states),
            cascade_depth_log=self._cascade_depth_log,
            dependency_order_log=self._dependency_order_log,
            action_history=self._action_history,
        )

# 🔥 ADD THIS LINE
        episode_score = max(0.1, min(0.99, float(episode_score)))
        
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
                if not src_ns.is_operational:
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
        if unlocked == 0:
            return 0.0
        return min(1.0, 0.3 + 0.2 * unlocked)

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

            op_obs = ns.is_operational
            if is_delayed:
                delayed_threshold = (
                    FAILURE_THRESHOLD_HARDENED if ns.is_hardened else FAILURE_THRESHOLD_UNHARDENED
                )
                op_obs = health_obs >= delayed_threshold and not ns.in_recovery

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
                )
            )

        for nid, ns in self._node_states.items():
            buf = self._obs_buffer.setdefault(nid, [])
            buf.append((ns.health, ns.load))
            if len(buf) > 4:
                buf.pop(0)

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

        info_str = (
            f"step={self._step} | task={self._task_id} | weather={self._weather_forecast} "
            f"| budget={self._budget:.1f} | failures={len(active_failures)} | pressure={system_pressure:.2f}"
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
        return recoverable_failed or overload_present or coord_opportunity

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
            "storm_warning": 0.04,
            "extreme_storm": 0.08,
        }.get(self._weather_forecast, 0.0)

        for ns in self._node_states.values():
            if not ns.is_operational:
                continue

            drift = self._rng.uniform(-0.03, 0.05) + weather_boost
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

        # Base time-step cost
        r -= 0.01

        if action_type == "wait":
            r -= WAIT_BASE_PENALTY

        if wait_under_pressure:
            r -= WAIT_UNDER_PRESSURE_PENALTY

        if actionable_wait:
            r -= 0.12

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
                    r -= 0.50   # newly failed this step
                # (already-failed hospitals give no +0.08 and no extra penalty)
            elif ns.health > 0.4:
                # Reward stability only when hospital improved vs pre-step state.
                prev_h = prev_health_map.get(ns.node_id, ns.health)
                if ns.health > prev_h + 1e-6:
                    r += 0.08
            else:
                # Hospital operational but health between 0 and 0.4: danger zone
                r -= 0.30   # reduced from -0.50 to avoid double-hit instability

        # Node fails during this step (all sectors, including any hospital)
        r -= 0.25 * len(newly_failed)

        # Recovery completes
        r += 0.30 * len(completed_recoveries)

        if recovery_correct_order:
            r += 0.10

        if dependency_recovery_score > 0.0:
            r += DEPENDENCY_RECOVERY_BONUS * dependency_recovery_score

        if cascade_impact_count > 0:
            r -= CASCADE_IMPACT_PENALTY * cascade_impact_count

        if load_shed_protects:
            r += 0.10

        if coord_useful:
            r += 0.05

        return r

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
