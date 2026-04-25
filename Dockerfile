FROM node:20-bookworm-slim AS ui-builder

WORKDIR /app/ui

COPY ui/package.json ui/package-lock.json ./
RUN npm ci

COPY ui/ ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e .

# Smoke-test: catch import errors at build time, not at runtime
RUN python -c "from cascade_guard.client import CascadeGuardEnv; print('OK cascade_guard.client')"
RUN python -c "from cascade_guard.server.cascade_environment import CascadeEnvironment; print('OK cascade_guard.server')"

COPY --from=ui-builder /app/ui/dist /app/ui/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "server/app.py"]
