#!/usr/bin/env bash
# Starts the FastAPI backend in the background, waits for it to become
# healthy, then runs the Gradio UI in the foreground. HF Spaces (Docker
# SDK) only exposes one port per Space, so both processes share this one
# container -- Gradio on the exposed port, FastAPI on an internal one.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

echo "[entrypoint] starting FastAPI backend on port ${API_PORT}..."
uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

cleanup() {
  echo "[entrypoint] shutting down API (pid ${API_PID})..."
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[entrypoint] waiting for API health check..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${API_PORT}/api/v1/health" > /dev/null; then
    echo "[entrypoint] API is up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[entrypoint] API did not become healthy in time." >&2
    exit 1
  fi
  sleep 1
done

echo "[entrypoint] starting Gradio UI on port ${GRADIO_PORT}..."
export API_BASE="http://localhost:${API_PORT}/api/v1"
export GRADIO_PORT
python ui/gradio_app.py
