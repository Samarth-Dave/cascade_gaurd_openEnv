---
title: CascadeGuard
emoji: ⚡
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# CascadeGuard

Single-Space deployment: FastAPI on port 7860 serves both the `/api/*` routes
AND the built React UI as static files. Browser, UI, and API all share the
same origin in production, so CORS does not apply there.

## What it does

- exposes `/api/*` endpoints for episode control, analytics, history, and dataset download
- talks to the deployed OpenEnv environment server via `OPENENV_SERVER_URL`
- loads a local or HuggingFace model at startup for `/api/episode/agent-step`
- streams live episode events over `ws/episode/{session_id}`

## Environment

Copy `.env.example` and set:

- `OPENENV_SERVER_URL`
- `MODEL_PATH`
- `MODEL_NAME`
- `UI_SPACE_URL`
- `PORT`

Judges: place your model in `./model` directory OR set `MODEL_NAME` to a HuggingFace model ID. The model loads automatically on backend startup.

## Local Run

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 7860
```

## Main Routes

- `GET /api/health`
- `POST /api/episode/reset`
- `POST /api/episode/step`
- `POST /api/episode/agent-step`
- `GET /api/episode/state/{session_id}`
- `GET /api/episodes/history`
- `GET /api/training/reward-curve`
- `GET /api/training/baseline-comparison`
- `GET /api/environment/graph`
- `GET /api/dataset/sft`
- `GET /api/dataset/sft/stats`
- `WS /ws/episode/{session_id}`
