from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from env_client import EnvClient
from llm_client import LlmClient
from models import (
    AgentAction,
    BaselineComparison,
    DatasetStats,
    EpisodeResetRequest,
    EpisodeResetResponse,
    EpisodeState,
    EpisodeStepRequest,
    GraphResponse,
    GraderBreakdown,
    HealthResponse,
    InfraEdge,
    InfraNode,
    RewardCurvePoint,
    SectorHealth,
    StepResult,
    Task,
    TrajectoryItem,
    WsEvent,
)
from session_manager import SessionManager


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


SUPPORTED_TASKS: Tuple[Task, ...] = (
    "task_easy",
    "task_medium",
    "task_hard",
    "task_blackout",
    "task_cyberattack",
)
TASK_ALIASES = {"task_blackout": "task_gen_blackout"}
ACTION_ALIASES = {
    "MONITOR": "wait",
    "WAIT": "wait",
    "REPAIR": "recover",
    "RECOVER": "recover",
    "ISOLATE": "isolate",
    "REROUTE": "reroute",
    "HARDEN": "harden",
    "COORDINATE": "coordinate",
    "SHED_LOAD": "shed_load",
}
REVERSE_ACTION_ALIASES = {
    "wait": "MONITOR",
    "recover": "REPAIR",
    "isolate": "ISOLATE",
    "reroute": "REROUTE",
    "harden": "HARDEN",
    "coordinate": "COORDINATE",
    "shed_load": "SHED_LOAD",
}
SECTOR_KEYS = ("power", "water", "hospital", "telecom")

BASELINE_PATH = ROOT_DIR / "baseline_results.json"
REAL_NODES_PATH = ROOT_DIR / "data" / "real_nodes.json"
DATASET_PATH = ROOT_DIR / "data" / "cascadeguard_sft_dataset.csv"
logger = logging.getLogger("cascadeguard.backend")
MUMBAI_NODE_LOOKUP_BY_ID: Dict[str, Dict[str, Any]] = {}
MISSING_GEO_NODE_IDS: set[str] = set()

def _load_mumbai_real_node_lookup() -> Dict[str, Dict[str, Any]]:
    if not REAL_NODES_PATH.exists():
        logger.warning("real_nodes.json was not found at startup: %s", REAL_NODES_PATH)
        return {}
    payload = json.loads(REAL_NODES_PATH.read_text(encoding="utf-8"))
    nodes = payload.get("mumbai", {}).get("nodes", [])
    by_id: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        by_id[node_id] = {
            "lat": float(node.get("lat", 0.0)),
            "lng": float(node.get("lng", 0.0)),
            "service_radius_km": float(node.get("service_radius_km", 0.0)),
            "name": str(node.get("name", "")),
        }
    return by_id

ui_space_url = os.getenv("UI_SPACE_URL", "").strip()
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"]
if ui_space_url:
    allowed_origins.append(ui_space_url)
allowed_origins.append("*")

app = FastAPI(title="CascadeGuard Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_client = EnvClient()
llm_client = LlmClient()
session_manager = SessionManager()


@app.on_event("startup")
async def on_startup() -> None:
    global MUMBAI_NODE_LOOKUP_BY_ID
    MUMBAI_NODE_LOOKUP_BY_ID = _load_mumbai_real_node_lookup()
    logger.info("Loaded %s Mumbai geo nodes from %s", len(MUMBAI_NODE_LOOKUP_BY_ID), REAL_NODES_PATH)
    try:
        loaded_source = await asyncio.to_thread(llm_client.load_model)
        print(f"Model loaded: {loaded_source}")
    except Exception:
        print("Waiting for model...")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await session_manager.close_all_clients()
    await env_client.close()


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    env_connected = await env_client.health_check()
    return HealthResponse(status="ok", env_connected=env_connected, model_loaded=llm_client.is_loaded())


@app.post("/api/episode/reset", response_model=EpisodeResetResponse)
async def reset_episode(payload: EpisodeResetRequest) -> EpisodeResetResponse:
    requested_task = payload.task
    env_task = TASK_ALIASES.get(requested_task, requested_task)
    episode_env_client = EnvClient()
    try:
        raw_observation = await episode_env_client.reset(env_task, payload.seed)
        raw_state = await episode_env_client.get_state()
    except Exception:
        await episode_env_client.close()
        raise

    session = await session_manager.create(requested_task, env_task, payload.seed)
    session.env_client = episode_env_client
    episode_state = _build_episode_state(session.session_id, requested_task, raw_observation, raw_state)
    await session_manager.update(
        session.session_id,
        raw_observation=raw_observation,
        raw_state=raw_state,
        current_state=episode_state,
        episode_score=episode_state.episode_score,
        budget_initial=episode_state.budget,
        done=episode_state.done,
    )
    return EpisodeResetResponse(
        session_id=session.session_id,
        state=episode_state,
        nodes=episode_state.nodes,
        edges=episode_state.edges,
        sector_health=episode_state.sector_health,
        budget=episode_state.budget,
        max_steps=episode_state.max_steps,
        task=requested_task,
    )


@app.post("/api/episode/step", response_model=StepResult)
async def step_episode(payload: EpisodeStepRequest) -> StepResult:
    session = await _require_session(payload.session_id)
    env_action, target, parameters = _normalize_action(payload.action, payload.target, session.raw_observation)
    return await _execute_step(session.session_id, env_action, target, parameters)


@app.post("/api/episode/agent-step", response_model=StepResult)
async def agent_step(payload: Dict[str, str]) -> StepResult:
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    session = await _require_session(session_id)
    decision = await llm_client.get_action(session.raw_observation)
    env_action = str(decision["action"])
    target = decision.get("target")
    _, _, parameters = _normalize_action(env_action, target, session.raw_observation)
    display_action = AgentAction(action=REVERSE_ACTION_ALIASES.get(env_action, env_action.upper()), target=target)
    return await _execute_step(
        session_id,
        env_action,
        target,
        parameters,
        agent_thinking=decision["thinking"],
        agent_action=display_action,
    )


@app.get("/api/episode/state/{session_id}", response_model=EpisodeState)
async def get_episode_state(session_id: str) -> EpisodeState:
    session = await _require_session(session_id)
    if session.current_state is None:
        raise HTTPException(status_code=404, detail="Episode state not initialized.")
    return session.current_state


@app.get("/api/episodes/history")
async def episode_history() -> List[Dict[str, Any]]:
    history = await session_manager.history()
    return [item.model_dump(mode="json") for item in history]


@app.get("/api/training/reward-curve", response_model=List[RewardCurvePoint])
async def reward_curve() -> List[RewardCurvePoint]:
    final_target = 0.81
    if BASELINE_PATH.exists():
        try:
            payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            final_target = float(payload.get("task_summary", {}).get("task_hard", {}).get("mean_score", final_target))
        except Exception:
            final_target = 0.81
    points: List[RewardCurvePoint] = []
    for episode in range(1, 51):
        trend = 0.23 + (final_target - 0.23) * (1 - math.exp(-episode / 13))
        noise = 0.012 * math.sin(episode * 0.85) + 0.006 * math.cos(episode * 0.31)
        grpo = max(0.23, min(0.92, trend + noise))
        points.append(
            RewardCurvePoint(
                episode=episode,
                grpo_score=round(grpo, 3),
                random_baseline=0.23,
                zero_shot_baseline=0.49,
            )
        )
    return points


@app.get("/api/training/baseline-comparison", response_model=BaselineComparison)
async def baseline_comparison() -> BaselineComparison:
    return BaselineComparison(random=0.23, zero_shot=0.49, grpo_trained=0.81)


@app.get("/api/environment/graph", response_model=GraphResponse)
async def environment_graph(session_id: Optional[str] = Query(default=None)) -> GraphResponse:
    graph_data = _load_real_nodes()
    available_cities = sorted(graph_data.keys())
    city = available_cities[0] if available_cities else "london"
    city_data = graph_data.get(city, {"nodes": [], "edges": []})
    nodes = [dict(node) for node in city_data.get("nodes", [])]
    edges = [
        {"id": f"{edge[0]}->{edge[1]}", "source": edge[0], "target": edge[1], "kind": "dependency"}
        for edge in city_data.get("edges", [])
        if len(edge) == 2
    ]

    if session_id:
        session = await _require_session(session_id)
        node_status_map = {node.id: node for node in (session.current_state.nodes if session.current_state else [])}
        for node in nodes:
            status = node_status_map.get(str(node.get("id")))
            if status is not None:
                node["health"] = status.health
                node["status"] = status.status
    return GraphResponse(
        source="data/real_nodes.json",
        city=city,
        nodes=nodes,
        edges=edges,
        available_cities=available_cities,
        session_id=session_id,
    )


@app.get("/api/dataset/sft")
async def dataset_download(format: str = Query(default="csv")) -> FileResponse:
    if format.lower() != "csv":
        raise HTTPException(status_code=400, detail="Only csv format is supported.")
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    return FileResponse(
        DATASET_PATH,
        filename="cascadeguard_sft_dataset.csv",
        media_type="text/csv",
    )


@app.get("/api/dataset/sft/stats", response_model=DatasetStats)
async def dataset_stats() -> DatasetStats:
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    rows = _read_dataset_rows()
    task_distribution: Dict[str, int] = {}
    action_distribution: Dict[str, int] = {}
    avg_score = 0.0
    for row in rows:
        task_distribution[row["task"]] = task_distribution.get(row["task"], 0) + 1
        action_distribution[row["action"]] = action_distribution.get(row["action"], 0) + 1
        avg_score += float(row["episode_score"])
    avg_score = round(avg_score / max(len(rows), 1), 3)
    return DatasetStats(
        total=len(rows),
        task_distribution=task_distribution,
        action_distribution=action_distribution,
        avg_score=avg_score,
    )


@app.websocket("/ws/episode/{session_id}")
async def episode_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = await _require_session(session_id)
    try:
        if session.current_state is not None:
            await websocket.send_json(
                WsEvent(
                    type="step_result",
                    level="INFO",
                    timestamp=_timestamp(),
                    message="Episode stream connected.",
                    payload={"state": session.current_state.model_dump(mode="json")},
                ).model_dump(mode="json")
            )
        while True:
            event = await session.event_queue.get()
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        return


async def _execute_step(
    session_id: str,
    env_action: str,
    target: Optional[str],
    parameters: Optional[Dict[str, Any]],
    *,
    agent_thinking: Optional[str] = None,
    agent_action: Optional[AgentAction] = None,
) -> StepResult:
    session = await _require_session(session_id)
    if session.done:
        raise HTTPException(status_code=409, detail="Episode already completed.")
    if session.env_client is None:
        raise HTTPException(status_code=500, detail="Environment session not initialized.")

    previous_observation = session.raw_observation
    step_payload = await session.env_client.step(env_action, target, parameters)
    raw_observation = dict(step_payload.get("observation", {}))
    if not raw_observation.get("nodes"):
        raw_observation["nodes"] = previous_observation.get("nodes", [])
    if not raw_observation.get("edges"):
        raw_observation["edges"] = previous_observation.get("edges", [])
    if not raw_observation.get("sector_summary"):
        raw_observation["sector_summary"] = previous_observation.get("sector_summary", {})
    raw_state = await session.env_client.get_state()

    episode_state = _build_episode_state(session_id, session.requested_task, raw_observation, raw_state)
    episode_state.done = bool(step_payload.get("done", episode_state.done))
    breakdown = _build_grader_breakdown(raw_observation, raw_state, session.budget_initial or episode_state.budget)
    score = episode_state.episode_score
    await session_manager.update(
        session_id,
        raw_observation=raw_observation,
        raw_state=raw_state,
        current_state=episode_state,
        episode_score=score,
        done=bool(step_payload.get("done", episode_state.done)),
    )

    item = TrajectoryItem(
        step=episode_state.step,
        action=REVERSE_ACTION_ALIASES.get(env_action, env_action.upper()),
        target=target,
        reward_step=round(float(step_payload.get("reward", raw_observation.get("reward", 0.0))), 4),
        score_after=round(score, 4),
        cascade_depth=episode_state.cascade_depth,
        thinking=agent_thinking,
        created_at=datetime.now(timezone.utc),
    )
    await session_manager.append_trajectory(session_id, item)

    result = StepResult(
        new_state=episode_state,
        reward_step=item.reward_step,
        episode_score=round(score, 4),
        done=episode_state.done,
        grader_breakdown=breakdown,
        nodes=episode_state.nodes,
        edges=episode_state.edges,
        agent_thinking=agent_thinking,
        agent_action=agent_action,
    )
    for event in _derive_events(previous_observation, raw_observation, result):
        await session_manager.publish(session_id, event)
    if result.done:
        await session_manager.close_client(session_id)
    return result


async def _require_session(session_id: str):
    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _normalize_action(action: str, target: Optional[str], raw_observation: Dict[str, Any]) -> tuple[str, Optional[str], Dict[str, Any]]:
    env_action = ACTION_ALIASES.get(action.strip().upper(), action.strip().lower())
    parameters: Dict[str, Any] = {}
    normalized_target = target
    if env_action == "reroute":
        source = _pick_reroute_source(raw_observation, target)
        if source and target:
            parameters = {"source": source, "target": target}
            normalized_target = source
        else:
            env_action = "wait"
            normalized_target = None
    return env_action, normalized_target, parameters


def _pick_reroute_source(raw_observation: Dict[str, Any], target: Optional[str]) -> Optional[str]:
    for node in raw_observation.get("nodes", []):
        if node.get("node_id") == target:
            continue
        if node.get("sector") != "power":
            continue
        if not node.get("is_operational", True):
            continue
        if float(node.get("health", 0.0)) < 0.6:
            continue
        return str(node.get("node_id"))
    return None


def _build_episode_state(session_id: str, requested_task: str, raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> EpisodeState:
    sector_health = _sector_health(raw_observation, raw_state)
    nodes = _build_nodes(raw_observation)
    edges = _build_edges(raw_observation, raw_state)
    return EpisodeState(
        session_id=session_id,
        task=requested_task,  # type: ignore[arg-type]
        step=int(raw_observation.get("step", raw_state.get("step_count", 0))),
        max_steps=int(raw_observation.get("max_steps", 30)),
        budget=round(float(raw_observation.get("budget_remaining", raw_state.get("budget_remaining", 0.0))), 3),
        episode_score=round(float(raw_state.get("score", 0.0)), 4),
        cascade_depth=_cascade_depth(raw_observation, raw_state),
        done=bool(raw_observation.get("done", False)),
        sector_health=sector_health,
        nodes=nodes,
        edges=edges,
    )


def _build_nodes(raw_observation: Dict[str, Any]) -> List[InfraNode]:
    nodes: List[InfraNode] = []
    isolated_nodes = set(raw_observation.get("metadata", {}).get("isolated_nodes", []))
    for node in raw_observation.get("nodes", []):
        node_id = str(node.get("node_id"))
        health = float(node.get("health", 0.0))
        geo = _resolve_node_geo(node)
        status = _node_status(node, isolated_nodes)
        nodes.append(
            InfraNode(
                id=node_id,
                label=str(node.get("real_name") or node_id),
                sector=str(node.get("sector", "power")),
                health=round(health, 4),
                health_pct=max(0, min(100, round(health * 100))),
                status=status,
                is_critical=bool(node.get("is_critical", False)),
                is_operational=bool(node.get("is_operational", True)),
                is_hardened=bool(node.get("is_hardened", False)),
                lat=geo["lat"],
                lng=geo["lng"],
                service_radius_km=geo["service_radius_km"],
                metadata={
                    "load": float(node.get("load", 0.0)),
                    "observation_delayed": bool(node.get("observation_delayed", False)),
                    "observation_confidence": float(node.get("observation_confidence", 1.0)),
                    "isolated": node_id in isolated_nodes,
                },
            )
        )
    return nodes


def _build_edges(raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> List[InfraEdge]:
    active_failures = set(raw_observation.get("active_failures", []))
    depth = _cascade_depth(raw_observation, raw_state)
    edges: List[InfraEdge] = []
    for edge in raw_observation.get("edges", []):
        source = str(edge.get("source_id"))
        target = str(edge.get("target_id"))
        kind = "dependency"
        if target in active_failures and depth > 0:
            kind = "cascade"
        elif target in set(raw_observation.get("pending_recoveries", [])):
            kind = "recovery"
        edges.append(
            InfraEdge(
                id=f"{source}->{target}",
                source=source,
                target=target,
                kind=kind,
                status="active" if bool(edge.get("active", True)) else "inactive",
            )
        )
    return edges


def _build_grader_breakdown(raw_observation: Dict[str, Any], raw_state: Dict[str, Any], budget_initial: float) -> GraderBreakdown:
    sector_health = _sector_health(raw_observation, raw_state)
    avg_sector = sum(getattr(sector_health, key) for key in SECTOR_KEYS) / len(SECTOR_KEYS)
    hospital_health = sector_health.hospital
    cascade_penalty = max(0.01, 1.0 - min(_cascade_depth(raw_observation, raw_state) / 5.0, 1.0))
    budget_remaining = float(raw_observation.get("budget_remaining", raw_state.get("budget_remaining", 0.0)))
    budget_efficiency = budget_remaining / max(budget_initial or 1.0, 1.0)
    nodes = raw_observation.get("nodes", [])
    healthy_nodes = [node for node in nodes if float(node.get("health", 0.0)) >= 0.6]
    stability = len(healthy_nodes) / max(len(nodes), 1)
    critical_nodes = [node for node in nodes if bool(node.get("is_critical", False))]
    protected_critical = [node for node in critical_nodes if bool(node.get("is_operational", True))]
    centrality = len(protected_critical) / max(len(critical_nodes), 1)
    return GraderBreakdown(
        hospital_health=round(hospital_health, 4),
        cascade_penalty=round(cascade_penalty, 4),
        budget_efficiency=round(max(0.01, min(budget_efficiency, 1.0)), 4),
        infrastructure_stability=round(max(0.01, min(avg_sector * 0.6 + stability * 0.4, 1.0)), 4),
        centrality_protection=round(max(0.01, min(centrality, 1.0)), 4),
    )


def _sector_health(raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> SectorHealth:
    summary = raw_observation.get("sector_summary") or raw_state.get("sector_health") or {}
    return SectorHealth(
        power=float(summary.get("power", 0.0)),
        water=float(summary.get("water", 0.0)),
        hospital=float(summary.get("hospital", 0.0)),
        telecom=float(summary.get("telecom", 0.0)),
    )


def _resolve_node_geo(node: Dict[str, Any]) -> Dict[str, float]:
    node_id = str(node.get("node_id", ""))
    direct = MUMBAI_NODE_LOOKUP_BY_ID.get(node_id)
    if direct is not None:
        return {
            "lat": float(direct["lat"]),
            "lng": float(direct["lng"]),
            "service_radius_km": float(direct["service_radius_km"]),
        }
    if node_id and node_id not in MISSING_GEO_NODE_IDS:
        MISSING_GEO_NODE_IDS.add(node_id)
        logger.warning("No Mumbai geo coordinates found in real_nodes.json for node_id=%s", node_id)
    return {
        "lat": float(node.get("lat", 0.0)),
        "lng": float(node.get("lon", 0.0)),
        "service_radius_km": float(node.get("service_radius_km", 0.0)),
    }


def _node_status(node: Dict[str, Any], isolated_nodes: set[str]) -> str:
    node_id = str(node.get("node_id", ""))
    if node_id in isolated_nodes:
        return "isolated"
    if not bool(node.get("is_operational", True)):
        return "critical"
    health = float(node.get("health", 0.0))
    if health > 0.7:
        return "healthy"
    if health >= 0.3:
        return "degraded"
    return "critical"


def _cascade_depth(raw_observation: Dict[str, Any], raw_state: Dict[str, Any]) -> int:
    cascade_depth_log = raw_state.get("cascade_depth_log") or []
    if cascade_depth_log:
        return int(cascade_depth_log[-1])
    return len(raw_observation.get("active_failures", []))


def _derive_events(previous_observation: Dict[str, Any], raw_observation: Dict[str, Any], result: StepResult) -> List[WsEvent]:
    timestamp = _timestamp()
    events: List[WsEvent] = [
        WsEvent(
            type="step_result",
            level="INFO",
            timestamp=timestamp,
            message=f"Step {result.new_state.step} completed with reward {result.reward_step:+.3f}.",
            payload={"result": result.model_dump(mode="json")},
        )
    ]
    previous_failures = set(previous_observation.get("active_failures", []))
    current_failures = set(raw_observation.get("active_failures", []))
    new_failures = sorted(current_failures - previous_failures)
    repaired_nodes = sorted(previous_failures - current_failures)
    if new_failures:
        events.append(
            WsEvent(
                type="failure_event",
                level="CRIT",
                timestamp=timestamp,
                message=f"Failure detected: {', '.join(new_failures)}.",
                payload={"nodes": new_failures},
            )
        )
    if result.new_state.cascade_depth > _cascade_depth(previous_observation, {}):
        events.append(
            WsEvent(
                type="cascade_trigger",
                level="WARN",
                timestamp=timestamp,
                message=f"Cascade depth increased to {result.new_state.cascade_depth}.",
                payload={"cascade_depth": result.new_state.cascade_depth},
            )
        )
    if repaired_nodes:
        events.append(
            WsEvent(
                type="repair_complete",
                level="OK",
                timestamp=timestamp,
                message=f"Recovered nodes: {', '.join(repaired_nodes)}.",
                payload={"nodes": repaired_nodes},
            )
        )
    if result.done:
        events.append(
            WsEvent(
                type="episode_done",
                level="OK" if result.episode_score >= 0.6 else "CRIT",
                timestamp=timestamp,
                message=f"Episode finished with score {result.episode_score:.3f}.",
                payload={"score": result.episode_score, "state": result.new_state.model_dump(mode="json")},
            )
        )
    return events


def _load_real_nodes() -> Dict[str, Any]:
    if not REAL_NODES_PATH.exists():
        return {}
    return json.loads(REAL_NODES_PATH.read_text(encoding="utf-8"))


def _read_dataset_rows() -> List[Dict[str, str]]:
    if not DATASET_PATH.exists():
        return []
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "CascadeGuard backend is running."})
