---
title: CascadeGuard UI
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# CascadeGuard UI

Judge-facing React dashboard for live simulation, analytics, episode history, and SFT dataset browsing.

## Local run

```powershell
cd ui
npm install
npm run dev
```

The dev server runs on `http://localhost:5173`.

## Environment

Use `.env.example` for local work:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

Use `.env.production` for Hugging Face Spaces:

```env
VITE_API_URL=https://<backend-space>.hf.space
VITE_WS_URL=wss://<backend-space>.hf.space
```

## Pages

- `/simulate` live containment controls + graph + terminal log
- `/analytics` training evidence cards + reward curve
- `/history` completed episode summaries + timelines
- `/dataset` filterable SFT browser + CSV download
- `/about` short architecture summary + deployment links

## Build

```powershell
npm run build
```
