# ---------------------------------------------------------------------------
# CascadeGuard — single Hugging Face Space (Docker)
# ---------------------------------------------------------------------------
# Stage 1: build the React UI
# Stage 2: install backend deps and copy the built UI into ./static
# FastAPI serves /api/* AND the React app from one container on port 7860.
# ---------------------------------------------------------------------------

FROM node:20-alpine AS ui-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build


FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY backend/ ./
COPY data/ ./data/
COPY training/ ./training/

# Built UI lives next to main.py so STATIC_DIR (./static) resolves correctly.
COPY --from=ui-builder /app/ui/dist ./static

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
