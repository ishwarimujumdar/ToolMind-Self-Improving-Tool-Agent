#!/bin/bash
set -e

echo "[ToolMind] Starting services..."

# Create data directories
mkdir -p /app/data/chroma_data

# Start OpenEnv server (for validation)
echo "[ToolMind] Starting OpenEnv server on port 7861..."
uvicorn server.app:app --host 0.0.0.0 --port 7861 &

# Start Agent API
echo "[ToolMind] Starting Agent API on port 8000..."
uvicorn api.agent_api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard
echo "[ToolMind] Starting Streamlit dashboard on port 8501..."
streamlit run frontend/streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false &

# Wait for services to start
sleep 3

# Start Nginx (foreground to keep container alive)
echo "[ToolMind] Starting Nginx on port 7860..."
nginx -g "daemon off;"
