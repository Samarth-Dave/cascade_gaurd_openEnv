---
title: CascadeGuard Backend
emoji: "\U0001F6E1"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
hardware: t4-small
models:
  - samarthdave0305/cascadeguard-trained
---

# CascadeGuard — Backend Space

This Space is a drop-in **replacement for a local `backend/` + `server/`
stack**. It runs everything CascadeGuard needs to serve the React UI in
a single process:

- **In-process `CascadeEnvironment`** — no WebSocket to a separate env
  Space, no `max_concurrent_envs=1` contention. Each session gets its
  own env instance.
- **Trained LLM from Hugging Face Hub** — the LoRA adapter pushed by
  the training Space (`samarthdave0305/cascadeguard-trained` by default)
  is downloaded on startup and merged into the base model for fast
  inference.
- **FastAPI API + WebSocket** — exposes the exact same `/api/*` and
  `/ws/episode/{session_id}` contract the UI already speaks.

Everything the React UI needs is available from this one URL.

---

## 1. One-time setup (you do this once per Space)

You need exactly three files in this Space repo:

- `app.py`           &nbsp;← from `deploy/app.py`
- `requirements.txt` &nbsp;← from `deploy/requirements.txt`
- `README.md`        &nbsp;← this file

The Space will clone the main CascadeGuard repo at startup (by default
`https://github.com/Samarth-Dave/cascade_gaurd_openEnv`) so the env code
is always up-to-date. If you forked it, override `CASCADE_REPO_URL`.

### 1.1 Create the Space

1. Go to https://huggingface.co/new-space
2. **Owner**: your account
3. **Name**: e.g. `cascadeguard-backend`
4. **License**: `mit` (or whatever your main repo uses)
5. **Select the Space SDK**: **Gradio**
6. **Hardware**: `T4 small` (16 GB VRAM) is the minimum for comfortable
   inference of any Qwen 7B/Llama 8B-class model. If you deployed the
   32B DeepSeek-R1-Distill training pipeline, bump to **A10G large**
   (24 GB VRAM) and the adapter will load in 4-bit.
7. **Public or private**: your call.
8. **Create Space**.

### 1.2 Upload the three files

From the Space's **Files** tab, click **+ Add file -> Upload files**
and drop in `app.py`, `requirements.txt`, and `README.md`. The Space
will start building immediately.

### 1.3 Set secrets (only if needed)

Open **Settings -> Variables and secrets** and add:

| Name                     | Type    | When you need it                                                   |
| ------------------------ | ------- | ------------------------------------------------------------------ |
| `HF_TOKEN`               | Secret  | Your adapter repo is private, **or** the base model is gated.      |
| `MODEL_NAME`             | Var     | Override the default `samarthdave0305/cascadeguard-trained`.       |
| `FALLBACK_MODEL_NAME`    | Var     | Override the default `Qwen/Qwen2.5-0.5B-Instruct`.                 |
| `CASCADE_REPO_URL`       | Var     | If you forked the main repo.                                       |

You don't need any secrets for the default path (public adapter + public
Qwen base).

### 1.4 Watch the build logs

First boot does three things — each takes roughly this long on T4:

1. **Install deps** (`pip install -r requirements.txt`) — 5 to 7 min.
2. **Clone the main repo + build package symlink** — <10 s.
3. **Download the model** — 2 to 8 min depending on size (the training
   Space pushed adapter files only, so the base model pulls first).

Once the logs say `Uvicorn running on http://0.0.0.0:7860` and then
`Model loaded from … on cuda`, the Space is ready.

---

## 2. Point the UI at the Space

The React UI reads its backend URL from `VITE_API_URL` and
`VITE_WS_URL`. Set them to your Space URL:

```bash
# ui/.env (or .env.local)
VITE_API_URL=https://<your-username>-cascadeguard-backend.hf.space
VITE_WS_URL=wss://<your-username>-cascadeguard-backend.hf.space
```

Then rebuild / restart the UI:

```bash
cd ui
npm run build            # for production (upload /ui/dist/ to a static Space)
# or
npm run dev              # for local dev against the remote backend
```

That's it — the UI has no other code change. `fetch("/api/health")`
becomes `fetch("https://.../api/health")` and the WebSocket connects to
`wss://.../ws/episode/{session_id}`.

---

## 3. Verifying the Space

```bash
curl -s https://<your-username>-cascadeguard-backend.hf.space/api/health | jq
```

You should see:

```json
{
  "status": "ok",
  "env_connected": true,
  "model_loaded": true,
  "model_load_state": "ready",
  "model_load_error": null
}
```

If `model_loaded` stays `false` and `model_load_state` is `"error"`,
open **Logs** on the Space and look at the first WARNING line — it will
name the repo that failed. Common causes:

| `model_load_error` contains…                 | Fix                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------- |
| `401 Client Error … access gated`            | Accept the base-model license on Hub, add `HF_TOKEN` with READ.     |
| `404 Client Error … Not Found`               | `MODEL_NAME` is wrong. Check you spelled the adapter repo correctly.|
| `CUDA out of memory`                         | Use a bigger Space hardware tier or set a smaller `MODEL_NAME`.     |
| `ImportError: bitsandbytes`                  | Hardware is CPU-only. Remove `bitsandbytes` from requirements.      |

---

## 4. Local testing (optional)

You can run this exact `app.py` locally. The only difference is that
when `/home/user/app` is not writable it falls back to cloning into
whatever `BASE_DIR` points at:

```bash
cd deploy
pip install -r requirements.txt
BASE_DIR=$PWD/_work python app.py
# -> serving at http://localhost:7860
```

Then hit `http://localhost:7860/docs` for the OpenAPI docs.

---

## 5. How the three pieces talk

```
                   ┌──────────────────────────────────┐
                   │   Browser (your laptop)          │
                   │   React UI: VITE_API_URL         │
                   │              VITE_WS_URL         │
                   └───────────────┬──────────────────┘
                                   │ HTTPS / WSS
                                   ▼
          ┌──────────────────────────────────────────────┐
          │   HF Space: cascadeguard-backend (this one)  │
          │                                              │
          │   ┌──────────────────────────────────────┐   │
          │   │  FastAPI  /api/*  /ws/*              │   │
          │   └──────────┬─────────────┬─────────────┘   │
          │              │             │                 │
          │              ▼             ▼                 │
          │   ┌──────────────────┐  ┌───────────────┐    │
          │   │ InProcessEnv     │  │ LlmClient     │    │
          │   │   CascadeEnv     │  │  (LoRA merge) │    │
          │   └──────────────────┘  └───────┬───────┘    │
          │                                 │            │
          └─────────────────────────────────┼────────────┘
                                            │ HF Hub API
                                            ▼
                   ┌──────────────────────────────────┐
                   │  samarthdave0305/                │
                   │    cascadeguard-trained (adapter)│
                   │  + base model (Qwen/DeepSeek)    │
                   └──────────────────────────────────┘
```

No shared env Space, no `max_concurrent_envs=1`, no zombie WebSocket —
the whole loop lives in this process.

---

## 6. Upgrading after retraining

Whenever the training Space pushes a new adapter:

```bash
# SSH into the Space (or Settings -> Factory reboot):
# this forces the new adapter to be pulled on next startup.
```

Or simpler: click **Settings -> Factory reboot**. The `HF_HOME` cache
is volatile on Spaces, so the next boot will re-download the adapter
and pick up the new weights automatically.

---

## 7. What this Space intentionally does NOT do

- **Training** — that's the other Space. Keep concerns separate.
- **Serving the React UI** — host it on a static Space or Vercel
  pointing at this backend. The old single-Space bundle path is still
  available in `backend/main.py` if you want to stitch them back
  together later.
- **Per-user authentication** — there is no multi-tenant auth here.
  Treat the API as internal to your demo; for a public launch put a
  lightweight gateway in front.

---

## 8. File manifest

| File               | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `app.py`           | The FastAPI app, env wrapper, LLM client, Gradio landing page.   |
| `requirements.txt` | Pinned deps that install cleanly on the HF GPU image.            |
| `README.md`        | HF Space metadata (YAML header) + this deploy guide.             |

Any other file you drop into the Space is ignored by `app.py`.
