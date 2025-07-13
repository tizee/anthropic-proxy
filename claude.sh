#!/bin/bash

# Check server status
if [ -f "uvicorn.pid" ] && pgrep -F uvicorn.pid > /dev/null; then
    echo "Server is already running"
else
    echo "Starting server..."
    make run
    sleep 3  # Wait for server initialization
fi

# Launch Claude Code
ANTHROPIC_BASE_URL=http://127.0.0.1:8082 claude "$@"
