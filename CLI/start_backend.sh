#!/usr/bin/env bash
# start_backend.sh — starts the FastAPI backend using the project venv
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_UVICORN="$SCRIPT_DIR/../venv/bin/uvicorn"

if [ ! -f "$VENV_UVICORN" ]; then
  echo "ERROR: venv not found at $VENV_UVICORN"
  echo "Run: python3 -m venv ../venv && ../venv/bin/pip install fastapi uvicorn"
  exit 1
fi

cd "$SCRIPT_DIR"
echo "Starting backend on http://localhost:8000"
"$VENV_UVICORN" backend:app --reload --port 8000
