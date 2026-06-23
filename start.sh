#!/usr/bin/env bash
# Container entrypoint: API in the background, Streamlit UI on the HF port (7860).
set -e

uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level warning &

# wait for the API to be healthy before starting the UI
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/ >/dev/null 2>&1 && break
  sleep 1
done

streamlit run ui/app.py \
  --server.port 7860 --server.address 0.0.0.0 --server.headless true \
  --server.enableXsrfProtection false --server.enableCORS false
