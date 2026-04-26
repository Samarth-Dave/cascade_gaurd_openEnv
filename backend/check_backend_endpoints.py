#!/usr/bin/env python3
"""
CascadeGuard — backend API catalog + smoke test
==============================================
Original server: `backend/main.py` (FastAPI). Run the backend first, e.g.:

    cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 7860

Then (with env server + optional LLM as you use locally):

    python check_backend_endpoints.py
    python check_backend_endpoints.py --base-url http://127.0.0.1:7860

Set BASE_URL in the environment to test a remote Space: BASE_URL=https://....hf.space

-------------------------------------------------------------------------------
ENDPOINT CATALOG (must stay in sync with backend/main.py)
-------------------------------------------------------------------------------

  Method   Path                                 Notes
  -------- ------------------------------------ --------------------------------
  GET      /api/health                          env_connected, model_load_state
  POST     /api/episode/reset                   JSON: task, seed (optional)
  POST     /api/episode/step                    JSON: session_id, action, target?, target2?, parameters?
  POST     /api/episode/agent-step              JSON: { "session_id": "..." } — 503 if model still loading
  GET      /api/episode/state/{session_id}
  GET      /api/episodes/history
  GET      /api/training/reward-curve
  GET      /api/training/baseline-comparison
  GET      /api/environment/graph               Query: ?session_id=&city=
  GET      /api/dataset/sft                     Query: ?format=csv (only csv supported)
  GET      /api/dataset/sft/stats
  WS       /ws/episode/{session_id}            Initial JSON + stream of events
  GET      /                                    Without static UI: { "status": "ok", ... }
  GET      /assets/*, /*                       Only when backend/static/ exists (Docker prod)

Static mounts are optional; the smoke test only hits API routes above.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    print("Install httpx: pip install httpx", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")


def _skip(msg: str) -> None:
    print(f"  SKIP {msg}")


def _get(client: httpx.Client, path: str, expect: Tuple[int, ...] = (200,)) -> Tuple[bool, Optional[dict]]:
    r = client.get(path)
    if r.status_code not in expect:
        return False, {"status_code": r.status_code, "text": r.text[:500]}
    try:
        return True, r.json() if r.content else None
    except json.JSONDecodeError:
        return True, None


def _post_json(client: httpx.Client, path: str, body: dict, expect: Tuple[int, ...] = (200,)) -> Tuple[bool, Any]:
    r = client.post(path, json=body)
    if r.status_code not in expect:
        return False, {"status_code": r.status_code, "text": r.text[:500]}
    try:
        return True, r.json() if r.content else None
    except json.JSONDecodeError:
        return True, r.text


async def _ws_smoke(base_ws: str, session_id: str) -> bool:
    if websockets is None:
        _skip("websockets not installed; pip install websockets")
        return True
    uri = f"{base_ws}/ws/episode/{session_id}"
    try:
        async with websockets.connect(uri) as ws:  # type: ignore[union-attr]
            first = await asyncio.wait_for(ws.recv(), timeout=15.0)
            data = json.loads(first)
            if not isinstance(data, dict) or "type" not in data:
                _fail(f"WebSocket: unexpected first message: {first[:200]!r}")
                return False
            _ok(f"WebSocket first frame type={data.get('type')!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail(f"WebSocket: {exc}")
        return False


def _http_to_ws_base(base: str) -> str:
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :]
    return base


def run_checks(base: str) -> int:
    base = base.rstrip("/")
    failed = 0

    with httpx.Client(base_url=base, timeout=120.0) as client:
        # --- GET / (optional) ---
        r = client.get("/")
        if r.status_code == 200:
            _ok("GET /")
        else:
            _skip(f"GET / -> {r.status_code} (expected when static UI mounted)")

        # --- /api/health ---
        ok, data = _get(client, "/api/health")
        if not ok or not data:
            _fail(f"/api/health: {data}")
            failed += 1
        else:
            _ok(
                f"/api/health status={data.get('status')!r} "
                f"env_connected={data.get('env_connected')} "
                f"model_load_state={data.get('model_load_state')!r}"
            )
            if not data.get("env_connected"):
                print(
                    "  NOTE: env_connected=false — start the OpenEnv server and ensure "
                    "OPENENV_SERVER_URL is set; reset/step will fail until the env is up.",
                )

        # --- /api/training/* (static / analytic) ---
        for path in ("/api/training/reward-curve", "/api/training/baseline-comparison"):
            ok, data = _get(client, path)
            if not ok or data is None:
                _fail(f"{path}: {data}")
                failed += 1
            else:
                _ok(f"{path} (len={len(data) if isinstance(data, list) else 'obj'})")

        # --- /api/episodes/history (before any episode: empty list ok) ---
        ok, data = _get(client, "/api/episodes/history")
        if not ok or not isinstance(data, list):
            _fail(f"/api/episodes/history: {data}")
            failed += 1
        else:
            _ok(f"/api/episodes/history (count={len(data)})")

        # --- /api/environment/graph (no session) ---
        ok, data = _get(client, "/api/environment/graph")
        if not ok or not data:
            _fail(f"/api/environment/graph: {data}")
            failed += 1
        else:
            _ok(
                f"/api/environment/graph city={data.get('city')!r} "
                f"nodes={len(data.get('nodes', []))}"
            )

        # --- /api/dataset/sft* (may 404 if data file missing) ---
        ok, data = _get(client, "/api/dataset/sft/stats", expect=(200, 404))
        if ok and isinstance(data, dict) and "total" in data:
            _ok(f"/api/dataset/sft/stats total_rows={data.get('total')}")
        else:
            _skip("/api/dataset/sft/stats — dataset not present (404) or error")

        r = client.get("/api/dataset/sft", params={"format": "csv"})
        if r.status_code == 200 and "text/csv" in (r.headers.get("content-type") or ""):
            _ok("/api/dataset/sft?format=csv (csv download)")
        elif r.status_code == 404:
            _skip("/api/dataset/sft — 404 (cascadeguard_sft_dataset.csv missing)")
        else:
            _fail(f"/api/dataset/sft: {r.status_code} {r.text[:200]!r}")
            failed += 1

        # --- POST /api/episode/reset ---
        reset_body = {"task": "task_easy", "seed": 42}
        r = client.post("/api/episode/reset", json=reset_body)
        if r.status_code != 200:
            _fail(f"POST /api/episode/reset: {r.status_code} {r.text[:400]}")
            failed += 1
            print("  Stopping: downstream tests need a valid session_id.")
            return failed

        reset_json = r.json()
        session_id = reset_json.get("session_id")
        if not session_id:
            _fail("reset response missing session_id")
            return failed + 1
        _ok(f"POST /api/episode/reset session_id={session_id!r}")

        # --- GET /api/episode/state/{session_id} ---
        ok, data = _get(client, f"/api/episode/state/{session_id}")
        if not ok or not data:
            _fail(f"GET /api/episode/state/...: {data}")
            failed += 1
        else:
            _ok(f"GET /api/episode/state/{{id}} task={data.get('task')!r} step={data.get('step')}")

        # --- GET /api/environment/graph?session_id= ---
        ok, data = _get(client, f"/api/environment/graph?session_id={session_id}")
        if not ok or not data:
            _fail(f"/api/environment/graph?session_id=: {data}")
            failed += 1
        else:
            _ok(f"/api/environment/graph?session_id= city={data.get('city')!r}")

        # --- POST /api/episode/step (wait / MONITOR) ---
        step_body: Dict[str, Any] = {
            "session_id": session_id,
            "action": "MONITOR",
            "target": None,
        }
        ok, step_data = _post_json(client, "/api/episode/step", step_body)
        if not ok or not isinstance(step_data, dict):
            _fail(f"POST /api/episode/step: {step_data}")
            failed += 1
        else:
            _ok(
                f"POST /api/episode/step (MONITOR) reward={step_data.get('reward_step')} "
                f"done={step_data.get('done')}"
            )

        # --- POST /api/episode/agent-step ---
        r = client.post("/api/episode/agent-step", json={"session_id": session_id})
        if r.status_code == 200:
            aj = r.json()
            _ok(
                f"POST /api/episode/agent-step model loaded (score_after={aj.get('episode_score')})"
            )
        elif r.status_code == 503:
            try:
                detail = r.json().get("detail", r.text)
            except json.JSONDecodeError:
                detail = r.text
            _skip(f"POST /api/episode/agent-step 503 (model not ready): {detail!r}")
        else:
            _fail(f"POST /api/episode/agent-step: {r.status_code} {r.text[:300]}")
            failed += 1

        # --- WebSocket ---
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            ws_ok = loop.run_until_complete(
                _ws_smoke(_http_to_ws_base(base), session_id)
            )
            if not ws_ok:
                failed += 1
        finally:
            loop.close()

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", "http://127.0.0.1:7860"),
        help="Backend root URL (default: env BASE_URL or http://127.0.0.1:7860)",
    )
    args = parser.parse_args()

    print(f"Target: {args.base_url!r}\n")
    n = run_checks(args.base_url)
    if n:
        print(f"\nDone: {n} check(s) failed.")
        raise SystemExit(1)
    print("\nDone: all required checks passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
