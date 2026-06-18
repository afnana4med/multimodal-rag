#!/usr/bin/env bash
# Launch the full Multimodal RAG stack with ONE command:
#   API (FastAPI/uvicorn) in the background + Streamlit UI in the foreground.
# Press Ctrl+C once to stop BOTH.
#
#   ./run.sh
set -e
cd "$(dirname "$0")"

# Use the project venv without requiring you to activate it first.
PY=".venv/bin"

echo "Starting API on http://localhost:8000 ..."
"$PY/uvicorn" api.main:app --port 8000 --log-level warning &
API_PID=$!

# Stop the API too when the UI exits / Ctrl+C.
trap 'echo; echo "Stopping API..."; kill $API_PID 2>/dev/null' EXIT INT TERM

# Wait for the API to be healthy before launching the UI.
for i in $(seq 1 60); do
  curl -sf http://localhost:8000/ >/dev/null 2>&1 && break
  sleep 0.5
done
echo "API is up. Starting Streamlit UI on http://localhost:8501 ..."

"$PY/streamlit" run ui/app.py
