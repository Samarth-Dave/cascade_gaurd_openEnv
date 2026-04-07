from __future__ import annotations

from openenv.core.env_server import create_app

from cascade_guard.server.cascade_environment import CascadeEnvironment
from cascade_guard.models import CascadeAction, CascadeObservation

app = create_app(CascadeEnvironment, CascadeAction, CascadeObservation)


@app.get("/health")
async def health():
    return {"status": "ok", "env": "cascade_guard"}


# ✅ ADD THIS
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ✅ MODIFY THIS
if __name__ == "__main__":
    main()
