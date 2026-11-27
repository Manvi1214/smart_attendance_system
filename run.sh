#!/bin/bash
# Smart Attendance Application run script

echo "Starting Smart Attendance System Server..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

echo "Using Python: $PYTHON_EXEC"
"$PYTHON_EXEC" "$SCRIPT_DIR/app.py"

