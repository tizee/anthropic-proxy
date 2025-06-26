#!/bin/bash

# Stop server script
stopped=false

# Check PID file first
if [ -f uvicorn.pid ]; then
    PID=$(cat uvicorn.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping server with PID $PID..."
        kill $PID
        rm -f uvicorn.pid
        stopped=true
        echo "Server stopped."
    else
        echo "Stale PID file found. Removing it."
        rm -f uvicorn.pid
    fi
fi

# Fallback: Kill python server.py processes
PIDS=$(pgrep -f "python server.py")
if [ -n "$PIDS" ]; then
    echo "Found running server process(es) $PIDS. Killing as a fallback..."
    kill $PIDS || true
    stopped=true
    echo "Server process(es) killed."
fi

# Fallback: Kill uvicorn dev server processes
PIDS=$(pgrep -f "uvicorn server:app")
if [ -n "$PIDS" ]; then
    echo "Found running dev server process(es) $PIDS. Killing as a fallback..."
    kill $PIDS || true
    stopped=true
    echo "Server process(es) killed."
fi

if [ "$stopped" = "false" ]; then
    echo "No running server found."
fi