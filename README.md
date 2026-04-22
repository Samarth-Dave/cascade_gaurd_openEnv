---
title: CascadeGuard Environment
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - infrastructure
---

# CascadeGuard

CascadeGuard is a cascading-failure resilience system built around two pieces:

- an OpenEnv backend that simulates cross-sector infrastructure crises
- a Vite + React frontend that visualizes live episodes and sends operator actions

The backend models failures across power, water, hospital, and telecom systems. The frontend connects to the backend over WebSocket, renders the live graph and sector health, and lets the user step the environment or dispatch actions.

## Repository Layout

This git repository contains the environment and server:

```text
cascade_gaurd_openEnv/
  server/
    app.py                    FastAPI/OpenEnv server entrypoint
    cascade_environment.py    environment dynamics and reward logic
  models.py                   Pydantic action, observation, and state models
  tasks.py                    task registry and scenario definitions
  data/                       real-world node and edge data
  inference.py                baseline inference runner
  test_ws.py                  websocket protocol smoke test
  ui/
    README.md                 repo-side frontend documentation
```

The frontend source now lives inside this repository under:

```text
ui/
```
 
It is the active UI that replaced the old standalone HTML prototype.

## System Overview

### Backend

The backend exposes an OpenEnv-compatible environment with:

- typed actions via `CascadeAction`
- typed observations via `CascadeObservation`
- typed state via `CascadeState`
- reset and step semantics over WebSocket
- supporting REST endpoints for health, graph, city, task, and node metadata

### Frontend

The frontend is a React app that:

- connects to `ws://localhost:8000/ws`
- sends `reset` and `step` payloads
- renders the city map, nodes, edges, logs, and reward curve
- uses live OpenEnv data when available
- falls back to a scripted simulation if the backend is unreachable

## Backend Contract

### WebSocket Endpoint

The frontend connects to:

```text
ws://localhost:8000/ws
```

### Reset Message

```json
{
  "type": "reset",
  "data": {
    "task_id": "task_hard"
  }
}
```

### Step Message

```json
{
  "type": "step",
  "data": {
    "action_type": "recover",
    "target_node_id": "POWER_GEN_1",
    "parameters": {}
  }
}
```

### Observation Shape

The server returns observations containing:

- `step`
- `max_steps`
- `budget_remaining`
- `weather_forecast`
- `nodes`
- `edges`
- `active_failures`
- `pending_recoveries`
- `sector_summary`
- `diagnostics`
- `task_id`
- `info`
- `reward`
- `done`

Key node fields include:

- `node_id`
- `sector`
- `health`
- `load`
- `is_operational`
- `is_hardened`
- `observation_delayed`
- `is_critical`
- `real_name`
- `lat`
- `lon`
- `service_radius_km`

### Supported Core Actions

Common actions used by the UI:

- `recover`
- `harden`
- `shed_load`
- `coordinate`
- `wait`

The model layer also supports advanced actions such as:

- `isolate`
- `reroute`
- `prioritize`
- `deploy_repair_crew`
- `emergency_shutdown`
- `cross_sector_bridge`
- `patch_scada`
- `redistribute_load`
- `request_mutual_aid`
- `controlled_cascade`
- `multi_sector_lockdown`

## REST Endpoints

The FastAPI server also exposes:

- `GET /health`
- `GET /graph?city=london`
- `GET /cities`
- `GET /tasks`
- `GET /node_data?city=london&node_id=...`

## Prerequisites

### Backend

- Python 3.10+
- `fastapi`
- `uvicorn`
- `openenv-core`
- other Python dependencies from `pyproject.toml`

### Frontend

- Node.js 18+ recommended
- npm

## Backend Setup

From the repo root:

```powershell
pip install -e .
```

If you already have the Python dependencies installed, you can go straight to running the server.

## Frontend Setup

From the repo root:

```powershell
cd ui
npm.cmd install
```

If PowerShell blocks `npm.ps1`, keep using `npm.cmd`.

## How To Run Everything Locally

Open two terminals.

### Terminal 1 - backend

From `cascade_gaurd_openEnv`:

```powershell
python server/app.py
```

The backend starts on:

```text
http://localhost:8000
```

### Terminal 2 - frontend

From `ui`:

```powershell
cd ui
npm.cmd run dev
```

The frontend starts on:

```text
http://localhost:8080
```

## Local Test Flow

1. Start the backend.
2. Start the frontend.
3. Open `http://localhost:8080`.
4. Launch a city from the modal.
5. Confirm the UI banner shows a live OpenEnv connection.
6. Use `Step`, `Reset`, or the action buttons to interact with the environment.

If the backend is not reachable, the UI automatically switches to the scripted simulation fallback.

## Frontend Environment Variables

The frontend defaults to local URLs, so no env file is required for local development.

Optional overrides:

- `VITE_ENV_BASE`
- `VITE_ENV_WS`

Example:

```powershell
$env:VITE_ENV_BASE="http://localhost:8000"
$env:VITE_ENV_WS="ws://localhost:8000/ws"
npm.cmd run dev
```

## Verification Commands

### Backend health

```powershell
curl http://localhost:8000/health
```

### WebSocket smoke test

From the repo root:

```powershell
python test_ws.py
```

This validates that the server accepts a `reset` and a `step` call on the WebSocket protocol.

### Frontend production build

From the frontend folder:

```powershell
cd ui
npm.cmd run build
```

## Inference and Training

### Baseline inference

From the repo root:

```powershell
python inference.py
```

### Multi-seed evaluation

```powershell
$env:EVAL_MODE="multiseed"
$env:EVAL_SPLIT="holdout"
python inference.py
```

Switch `EVAL_SPLIT` as needed for your scenario split.

### GRPO training

The training code lives under `training/`. Use the notebook and training scripts there for GRPO experiments.

## Frontend Scripts

From `ui/`:

- `npm.cmd run dev`
- `npm.cmd run build`
- `npm.cmd run build:dev`
- `npm.cmd run lint`
- `npm.cmd run test`
- `npm.cmd run test:watch`

## Troubleshooting

### PowerShell blocks npm

Use:

```powershell
npm.cmd install
npm.cmd run dev
```

### Frontend falls back to simulation

Check:

- backend is running on port `8000`
- frontend is pointing to `ws://localhost:8000/ws`
- no stale custom `VITE_ENV_BASE` or `VITE_ENV_WS` values are set

### WebSocket connection refused

Start the backend first, wait until it is listening on port `8000`, then refresh the frontend.

### Vite build fails because dependencies are missing

Run:

```powershell
cd ..\cascade-guard-core-main
npm.cmd install
```

### Python server import errors

Reinstall the backend in editable mode:

```powershell
pip install -e .
```

## Additional Docs

- UI-specific docs: [ui/README.md](ui/README.md)
- Run notes: [HOW_TO_RUN.md](HOW_TO_RUN.md)
