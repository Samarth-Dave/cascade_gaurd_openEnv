# CascadeGuard UI

This folder contains the actual Vite + React frontend for CascadeGuard.

It replaces the old standalone HTML prototype and connects to the backend over:

- `http://localhost:8000`
- `ws://localhost:8000/ws`

## Run The UI Locally

From this `ui/` folder:

```powershell
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://localhost:8080
```

## Run The Backend For Live Mode

From the repo root:

```powershell
python -m uvicorn cascade_guard.server.app:app --host 0.0.0.0 --port 8000
```

Then launch the frontend and confirm the connection banner switches from simulation to live mode.

## Frontend Runtime Notes

- The frontend sends a `reset` message when an episode is launched.
- The frontend sends `step` messages for wait/dispatch actions.
- Live observations are normalized into map, timeline, event feed, and reasoning panels.
- If the backend is unavailable, the UI falls back to the scripted simulation.
- In deployment, the backend serves the built `ui/dist` bundle directly.

## Main Documentation

For the full combined guide covering backend, API contract, frontend integration, local run, testing, and troubleshooting, see [README.md](../README.md).
