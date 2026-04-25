---
title: CascadeGuard
emoji: "⚡"
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - infrastructure
---

# CascadeGuard

CascadeGuard is a cascading-failure resilience system for critical infrastructure.
It combines:

- an OpenEnv environment server (simulation engine)
- a FastAPI orchestration backend (sessions, API, websocket stream, LLM policy calls)
- a React dashboard (live episode controls, graph, analytics, dataset browser)

The system models failures across power, water, hospital, and telecom sectors and lets a user or agent execute containment actions step-by-step.

## What Is In This Repository

```text
cascade_gaurd_openEnv/
├── server/                 OpenEnv environment server (port 8000)
├── backend/                FastAPI orchestrator + static UI serving (port 7860)
├── ui/                     React/Vite dashboard source
├── training/               GRPO training scripts and notebook
├── data/                   real_nodes.json and SFT dataset CSV
├── tasks.py                scenario definitions + OSM task registration
├── inference.py            baseline/multiseed evaluator runner
└── scripts/                utility and verification scripts
```

## Architecture

Data flow during normal use:

1. UI calls backend REST endpoints under /api/* and subscribes to /ws/episode/{session_id}
2. Backend creates per-session environment clients and forwards actions to environment server /reset, /step, /state
3. Backend enriches state, computes grader breakdown, and streams events
4. UI renders graph, sector health, score, logs, history, and analytics

Production behavior:

- backend/main.py serves both API and built UI static files from one container on port 7860
- in this mode, browser/API are same-origin

## Quick Start (Local)

Prerequisites:

- Python 3.11+
- Node.js 20+ (or 18+)
- npm

### 1. Install Python Dependencies

From repo root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r backend/requirements.txt
```

### 2. Install UI Dependencies

```powershell
cd ui
npm install
cd ..
```

### 3. Run Services

Use three terminals.

Terminal A (environment server, port 8000):

```powershell
uvicorn cascade_guard.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Terminal B (backend orchestrator, port 7860):

```powershell
cd backend
$env:OPENENV_SERVER_URL="http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Terminal C (UI dev server, port 5173):

```powershell
cd ui
$env:VITE_API_URL="http://localhost:7860"
$env:VITE_WS_URL="ws://localhost:7860"
npm run dev
```

Open http://localhost:5173.

Notes:

- If OPENENV_SERVER_URL is not set, backend defaults to https://samarthdave0305-cascade-failure-env.hf.space
- ui/.env in this repo already points to backend port 7860 for local dev

## Main Backend API (backend/main.py)

REST endpoints:

- GET /api/health
- POST /api/episode/reset
- POST /api/episode/step
- POST /api/episode/agent-step
- GET /api/episode/state/{session_id}
- GET /api/episodes/history
- GET /api/training/reward-curve
- GET /api/training/baseline-comparison
- GET /api/environment/graph
- GET /api/dataset/sft?format=csv
- GET /api/dataset/sft/stats

WebSocket:

- WS /ws/episode/{session_id}

Manual step contract:

```json
POST /api/episode/step
{
  "session_id": "...",
  "action": "REPAIR|ISOLATE|REROUTE|MONITOR|...",
  "target": "NODE_ID or null"
}
```

## Environment Server API (server/app.py)

- GET /health
- GET /graph?city=...
- GET /cities
- GET /tasks
- GET /actions
- GET /node_data?city=...&node_id=...

## Tasks and Scenarios

UI selectable tasks (current backend Task type):

- task_easy
- task_medium
- task_hard
- task_blackout (mapped to task_gen_blackout internally)
- task_cyberattack

Environment task templates in tasks.py additionally include:

- task_gen_blackout
- task_surge_demand
- task_real_city
- task_osm_london
- task_osm_mumbai
- task_osm_bangalore
- task_osm_delhi
- task_osm_nyc
- task_osm_tokyo

OSM tasks are loaded dynamically from data/real_nodes.json at import time.

## UI Pages

- /simulate: live episode controls, graph, manual actions, terminal log
- /analytics: reward curve and baseline comparison cards
- /history: completed episode summaries and trajectories
- /dataset: filterable SFT browser + CSV download
- /about: architecture summary and links

## Training, Evaluation, and Data

Artifacts:

- data/cascadeguard_sft_dataset.csv
- data/real_nodes.json
- baseline_results.json

Training and evaluation entrypoints:

- training/train_grpo.py
- training/CascadeGuard_GRPO_Colab.ipynb
- inference.py

Common examples:

```powershell
python inference.py
```

```powershell
$env:EVAL_MODE="multiseed"
$env:EVAL_SPLIT="holdout"
python inference.py
```

## Deployment Notes

- backend/Dockerfile builds UI in a Node stage, copies dist into backend/static, and serves everything from FastAPI on port 7860
- root Dockerfile builds the OpenEnv environment server image for port 8000

## Verification and Troubleshooting

Health checks:

```powershell
curl http://localhost:7860/api/health
curl http://localhost:8000/health
```

Useful checks:

```powershell
python scripts/verify_training_pipeline.py
python scripts/verify_upgrade.py
```

If npm scripts are blocked in PowerShell, use npm.cmd instead of npm.

If UI requests fail, confirm VITE_API_URL and VITE_WS_URL point to backend (7860), not environment server (8000).

## Additional Docs

- backend/README.md
- ui/README.md
- HOW_TO_RUN.md
- CASCADEGUARD_MASTER_README.md
