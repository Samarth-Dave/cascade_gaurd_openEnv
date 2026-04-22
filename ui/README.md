# CascadeGuard UI

This folder contains the repository copy of the frontend documentation for the `harsh` branch.

In this workspace, the actual Vite frontend source lives in the sibling folder:

```text
../cascade-guard-core-main
```

That frontend is the current UI for CascadeGuard. It replaces the old standalone HTML prototype and connects to the backend over:

- `http://localhost:8000`
- `ws://localhost:8000/ws`

## Run The UI Locally

From the frontend project folder:

```powershell
cd ..\cascade-guard-core-main
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://localhost:8080
```

## Run The Backend For Live Mode

From this repo root:

```powershell
python server/app.py
```

Then launch the frontend and confirm the connection banner switches from simulation to live mode.

## Frontend Runtime Notes

- The frontend sends a `reset` message when an episode is launched.
- The frontend sends `step` messages for wait/dispatch actions.
- Live observations are normalized into map, timeline, event feed, and reasoning panels.
- If the backend is unavailable, the UI falls back to the scripted simulation.

## Main Documentation

For the full combined guide covering backend, API contract, frontend integration, local run, testing, and troubleshooting, see [README.md](../README.md).
