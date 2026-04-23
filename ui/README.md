# CascadeGuard Frontend

This is the Vite + React + Tailwind frontend for CascadeGuard.

It is the current UI for the app and replaces the old standalone HTML prototype. The frontend connects to the CascadeGuard OpenEnv backend over the same local endpoints used by the old HTML flow:

- HTTP base: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws`

When the backend is running, the UI uses live OpenEnv observations. If the backend is unavailable, the UI falls back to the local scripted simulation so the experience still loads.

## Stack

- Vite 5
- React 18
- TypeScript
- Tailwind CSS
- Leaflet for the live map
- Vitest for tests

## Project Layout

```text
cascade-guard-core-main/
  src/
    cascade/
      components/        UI sections and map panels
      config/env.ts      frontend env var defaults
      data/              scripted episode data and types
      hooks/             live websocket integration
      lib/               live observation normalization
    pages/
      Index.tsx          app-level orchestration
  public/
  package.json
  vite.config.ts
```

## What The Frontend Does

- Opens a WebSocket to `ws://localhost:8000/ws`
- Sends a reset payload when the user launches an episode:

```json
{
  "type": "reset",
  "data": {
    "task_id": "task_hard"
  }
}
```

- Sends action payloads when the user steps or dispatches an action:

```json
{
  "type": "step",
  "data": {
    "action_type": "wait",
    "target_node_id": null,
    "parameters": {}
  }
}
```

- Normalizes backend observations into frontend state for:
  - map nodes and edges
  - sector health
  - event feed and logs
  - step timeline
  - attacker/defender reasoning panels
  - reward curve

## Requirements

- Node.js 18+ recommended
- npm
- The backend running locally on port `8000`

## Local Run

Open a terminal in `cascade-guard-core-main` and install dependencies:

```powershell
npm.cmd install
```

Start the dev server:

```powershell
npm.cmd run dev
```

By default Vite serves the app at:

```text
http://localhost:8080
```

## Backend Needed For Live Mode

The frontend expects the backend to be running separately. From the backend folder:

```powershell
cd ..\cascade_gaurd_openEnv
python server/app.py
```

Once both are up:

1. Open `http://localhost:8080`
2. Launch a city from the modal
3. Confirm the map banner shows a live OpenEnv connection instead of simulation fallback

## Environment Variables

The frontend already defaults to local development values, so you do not need an env file for local testing.

Optional overrides:

- `VITE_ENV_BASE`
- `VITE_ENV_WS`

Example:

```powershell
$env:VITE_ENV_BASE="http://localhost:8000"
$env:VITE_ENV_WS="ws://localhost:8000/ws"
npm.cmd run dev
```

## Scripts

- `npm.cmd run dev` - start local Vite dev server
- `npm.cmd run build` - production build
- `npm.cmd run build:dev` - development-mode build
- `npm.cmd run lint` - run ESLint
- `npm.cmd run test` - run Vitest once
- `npm.cmd run test:watch` - run Vitest in watch mode

## Verified Local Flow

The current frontend wiring was verified against the local backend by:

- building the frontend successfully with `npm.cmd run build`
- confirming the backend websocket accepts `reset` and `step` messages on `ws://localhost:8000/ws`

## Troubleshooting

### PowerShell blocks `npm`

If PowerShell blocks `npm.ps1`, use `npm.cmd` instead:

```powershell
npm.cmd install
npm.cmd run dev
```

### UI says simulation instead of live

Check:

- backend is running on `localhost:8000`
- websocket endpoint is reachable at `ws://localhost:8000/ws`
- frontend env vars are not pointing somewhere else

### Failed to fetch or disconnected errors

Make sure the backend starts cleanly before opening the frontend. Then refresh the page and launch the episode again.

## Related Docs

- Combined project guide in `../cascade_gaurd_openEnv/README.md`
- Mirrored UI docs for the pushed repo branch in `../cascade_gaurd_openEnv/ui/README.md`
