from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_app

from cascade_guard.server.cascade_environment import CascadeEnvironment
from cascade_guard.models import CascadeAction, CascadeObservation

app = create_app(CascadeEnvironment, CascadeAction, CascadeObservation)

# ---------------------------------------------------------------------------
# Core health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "env": "cascade_guard", "version": "0.4.0"}


# ---------------------------------------------------------------------------
# Real-world graph endpoints (consumed by UI map + training clients)
# ---------------------------------------------------------------------------

_REAL_NODES_PATH = Path(__file__).parent.parent / "data" / "real_nodes.json"
_UI_DIST_PATH = Path(__file__).parent.parent / "ui" / "dist"
_UI_INDEX_PATH = _UI_DIST_PATH / "index.html"
_UI_RESERVED_PREFIXES = (
    "health",
    "graph",
    "cities",
    "tasks",
    "node_data",
    "docs",
    "redoc",
    "openapi.json",
    "ws",
)


def _load_real_nodes() -> dict:
    if _REAL_NODES_PATH.exists():
        return json.loads(_REAL_NODES_PATH.read_text(encoding="utf-8"))
    return {}


def _ui_available() -> bool:
    return _UI_INDEX_PATH.exists()


if (_UI_DIST_PATH / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_UI_DIST_PATH / "assets")), name="ui-assets")


@app.get("/graph")
async def get_graph(city: str = "london"):
    """
    Returns the full real-world node + edge graph for the given city.
    Called once on UI load to render the map.

    Returns
    -------
    {
        "nodes": [ {id, lat, lng, sector, name, service_radius_km}, ... ],
        "edges": [ [source_id, target_id], ... ],
        "center": [lat, lng],
        "city": str
    }
    """
    data = _load_real_nodes()
    if city not in data:
        available = list(data.keys())
        return JSONResponse(
            {"error": f"City '{city}' not found. Available: {available}"},
            status_code=404,
        )
    city_data = data[city]
    return JSONResponse({
        "nodes":  city_data.get("nodes", []),
        "edges":  city_data.get("edges", []),
        "center": city_data.get("center", [0.0, 0.0]),
        "city":   city,
    })


@app.get("/cities")
async def get_cities():
    """
    Returns available cities with node/edge counts and center coordinates.
    Used by the UI city selector.
    """
    data = _load_real_nodes()
    return {
        city: {
            "node_count": v.get("node_count", len(v.get("nodes", []))),
            "edge_count": v.get("edge_count", len(v.get("edges", []))),
            "center":     v.get("center", [0.0, 0.0]),
        }
        for city, v in data.items()
    }


@app.get("/tasks")
async def get_tasks():
    """
    Returns all registered task IDs with their descriptions and metadata.
    Includes dynamically loaded OSM city tasks.
    """
    from cascade_guard.tasks import TASK_CONFIGS
    return {
        tid: {
            "description": cfg.get("description", ""),
            "max_steps":   cfg.get("max_steps", 30),
            "budget":      cfg.get("budget", 15.0),
            "node_count":  len(cfg.get("nodes", [])),
            "city":        cfg.get("city", "mumbai"),
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


@app.get("/node_data")
async def get_node_data(city: str = "london", node_id: str = ""):
    """
    Returns detailed real-world metadata for a specific node.
    Used by UI tooltips and LLM agent prompts.
    """
    data = _load_real_nodes()
    if city not in data:
        return JSONResponse({"error": f"City '{city}' not found."}, status_code=404)

    nodes = {n["id"]: n for n in data[city].get("nodes", [])}
    if node_id and node_id not in nodes:
        return JSONResponse({"error": f"Node '{node_id}' not found."}, status_code=404)

    if node_id:
        return nodes[node_id]
    return {"nodes": nodes, "city": city}


@app.get("/", include_in_schema=False)
async def serve_ui_root():
    if _ui_available():
        return FileResponse(_UI_INDEX_PATH)
    return JSONResponse(
        {
            "status": "ok",
            "message": "CascadeGuard backend is running. Build ui/ to serve the frontend from this app.",
        }
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_ui_or_file(full_path: str, request: Request):
    del request

    if any(
        full_path == prefix or full_path.startswith(f"{prefix}/")
        for prefix in _UI_RESERVED_PREFIXES
    ):
        raise HTTPException(status_code=404, detail="Not found")

    if not _ui_available():
        raise HTTPException(status_code=404, detail="Frontend build not found")

    candidate = (_UI_DIST_PATH / full_path).resolve()
    ui_root = _UI_DIST_PATH.resolve()
    if not str(candidate).startswith(str(ui_root)):
        raise HTTPException(status_code=404, detail="Not found")

    if candidate.is_file():
        return FileResponse(candidate)

    return FileResponse(_UI_INDEX_PATH)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
